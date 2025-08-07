import io
import json
import zipfile
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import plotly.express as px
import textwrap
from sklearn.decomposition import PCA

# --- Helper: percentile stretch and gamma correction ---
def stretch_and_gamma(arr: np.ndarray, low_p: float, high_p: float, gamma: float) -> np.ndarray:
    """
    Percentile-based stretch and optional gamma correction.
    """
    lo, hi = np.percentile(arr, (low_p, high_p))
    arr_clipped = np.clip((arr - lo) / (hi - lo + EPSILON), 0, 1)
    if gamma != 1.0:
        arr_clipped = np.power(arr_clipped, 1.0 / gamma)
    return (arr_clipped * 255.0).astype(np.uint8)

# --- Optional: capture Plotly lasso selections via streamlit-plotly-events ---
try:
    from streamlit_plotly_events import plotly_events
    HAS_PLOTLY_EVENTS = True
except ImportError:
    HAS_PLOTLY_EVENTS = False

# --- Streamlit version check for Plotly flicker workaround ---
from packaging import version

import logging
logging.basicConfig(level=logging.WARNING)

# --- Dependency Management ---
try:
    import cv2
    HAS_OPENCV = True
except ImportError:  # pragma: no cover
    HAS_OPENCV = False

try:
    import rasterio
    from rasterio.io import MemoryFile
    HAS_RASTERIO = True
except ImportError:  # pragma: no cover
    HAS_RASTERIO = False
    import tifffile as tiff

# --- Optional: streamlit-image-coordinates for white patch picker ---
try:
    from streamlit_image_coordinates import streamlit_image_coordinates as img_coords
    HAS_IMG_COORDS = True
except Exception:  # pragma: no cover
    HAS_IMG_COORDS = False

# --- Constants ---
EPSILON = np.finfo(float).eps
ECC_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-5) if HAS_OPENCV else None

# Pre-defined heuristic pigment signatures for suggestive matching.
PIGMENT_DB = [
    {"name": "Ultramarine", "metrics": {"R/G": 1.3, "B/R": 0.6, "(SWP-BP)/(SWP+BP)": 0.2}},
    {"name": "Azurite", "metrics": {"R/G": 0.8, "B/R": 1.1, "(SWP-BP)/(SWP+BP)": -0.1}},
    {"name": "Prussian blue", "metrics": {"R/G": 0.7, "B/R": 1.4, "(SWP-BP)/(SWP+BP)": -0.2}},
    {"name": "Phthalo blue", "metrics": {"R/G": 1.1, "B/R": 0.9, "(SWP-BP)/(SWP+BP)": 0.15}},
]

# -----------------------------------------------------------------------------
# Utility
# -----------------------------------------------------------------------------

def _safe_divide(numerator: Union[np.ndarray, float], denominator: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    return numerator / (denominator + EPSILON)


def _validate_tiff_metadata(fileobj: io.BytesIO) -> Dict[str, float]:
    """Read calibration-ish tags if present (best effort)."""
    fileobj.seek(0)
    try:
        with Image.open(fileobj) as img:
            tags = img.tag_v2
            def _as_float(val, default):
                if val is None:
                    return default
                if isinstance(val, (tuple, list)):
                    return float(val[0]) if len(val) > 0 else default
                try:
                    return float(val)
                except Exception:  # pragma: no cover
                    return default
            scale = _as_float(tags.get(33550, 1.0), 1.0)   # ModelPixelScaleTag
            offset = _as_float(tags.get(33922, 0.0), 0.0)  # ModelTiepointTag
        return {"scale": scale, "offset": offset}
    except Exception as e:  # pragma: no cover
        logging.info(f"No TIFF calibration metadata: {e}")
        return {}


@st.cache_data(show_spinner=False)
def _read_tiff(fileobj: io.BytesIO) -> np.ndarray:
    fileobj.seek(0)
    if HAS_RASTERIO:
        with MemoryFile(fileobj.read()) as memfile:
            with memfile.open() as src:
                arr = src.read(1)
    else:
        arr = tiff.imread(fileobj)
    arr = arr.astype(np.float32)
    if arr.ndim > 2:
        st.warning("Multi-band TIFF detected. Only the first band will be used.", icon="⚠️")
        arr = arr[0] if arr.shape[0] <= arr.shape[-1] else arr[..., 0]
    meta = _validate_tiff_metadata(fileobj)
    if meta:
        arr = arr * meta['scale'] + meta['offset']
        logging.debug(f"Applied TIFF calibration metadata: {meta}")
    return arr


def _white_patch_balance(channels: Dict[str, np.ndarray], ref_means: Dict[str, float]) -> Dict[str, np.ndarray]:
    target_mean = np.mean(list(ref_means.values()))
    balanced = {}
    for k, arr in channels.items():
        gain = _safe_divide(target_mean, ref_means[k])
        balanced[k] = np.clip(arr * gain, 0, None)
    return balanced


def _suggest_pigment(metric_dict: Dict[str, float]) -> str:
    use_keys = ["R/G", "B/R", "(SWP-BP)/(SWP+BP)"]
    db_matrix, names = [], []
    for item in PIGMENT_DB:
        db_matrix.append([item["metrics"].get(k, 0.0) for k in use_keys])
        names.append(item["name"])
    db_matrix = np.array(db_matrix)
    query_vector = np.array([metric_dict.get(k, 0.0) for k in use_keys])

    mean = db_matrix.mean(axis=0)
    std = db_matrix.std(axis=0) + EPSILON
    db_z = (db_matrix - mean) / std
    query_z = (query_vector - mean) / std

    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + EPSILON)

    best_match, best_score = "Unknown", -1.0
    for i, vec in enumerate(db_z):
        score = cosine_similarity(query_z, vec)
        if score > best_score:
            best_score, best_match = score, names[i]
    return best_match


def _generate_markdown_report(params: Dict[str, Any], roi_list: List[Dict[str, Any]]) -> str:
    lines = [
        f"# IR False-Colour Report",
        f"\n*Generated: {datetime.now().isoformat()}*",
        "\n## Processing Parameters",
    ]
    for k, v in params.items():
        val_str = json.dumps(v, indent=2, default=str)
        lines.append(f"- **{k}**: `{textwrap.shorten(val_str, width=100, placeholder='...')}`")

    lines.append("\n## Region of Interest (ROI) Metrics")
    if roi_list:
        df = pd.DataFrame(roi_list).set_index("label")
        lines.append(df.to_markdown())
    else:
        lines.append("*No ROIs were saved during this session.*")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Parameters dataclass (cleaner state passing & report generation)
# -----------------------------------------------------------------------------

@dataclass
class PCAOptions:
    scale_mode: str = "minmax"
    map: Dict[str, str] = None  # e.g. {"R": "PC1", "G": "PC2", "B": "PC3"}
    show_gray: bool = True
    whiten: bool = False


@dataclass
class RatioOptions:
    formula: str = "Default (R/G, B/R, (SWP-BP)/(SWP+BP))"
    low: float = 0.02
    high: float = 0.98
    gamma: float = 1.0


@dataclass
class AppParams:
    batch_mode: bool = False
    channel_map: Dict[str, str] = None
    stretch: Dict[str, Tuple[int, int, float]] = None
    auto_balance: bool = True
    gains: List[float] = None
    use_decor: bool = False
    saturation: float = 1.0
    pre_equalize: bool = True
    make_pca_helper: bool = False
    pca_opts: PCAOptions = field(default_factory=lambda: PCAOptions(map={"R": "PC1", "G": "PC2", "B": "PC3"}))
    make_ratio: bool = False
    ratio_opts: RatioOptions = field(default_factory=RatioOptions)


# -----------------------------------------------------------------------------
# Main application
# -----------------------------------------------------------------------------

class FalseColourApp:
    def __init__(self) -> None:
        if "rois" not in st.session_state:
            st.session_state["rois"] = []
        if "pc_debug" not in st.session_state:
            st.session_state["pc_debug"] = False

        self.params: AppParams = AppParams()
        self.raw_bands: Dict[str, np.ndarray] = {}
        self.source_bands: Dict[str, np.ndarray] = {}
        self.processed_channels: Dict[str, np.ndarray] = {}
        self.white_patch_means: Optional[Dict[str, float]] = None
        self.pca_components: Optional[List[np.ndarray]] = None
        self.pca_explained: Optional[np.ndarray] = None
        self.pca_rgb: Optional[np.ndarray] = None
        self.ratio_rgb: Optional[np.ndarray] = None
        self.dark_frame: Optional[np.ndarray] = None
        self.flat_field: Optional[np.ndarray] = None
        self.white_patch_enabled: bool = False
        self.white_patch_coords: Optional[Tuple[int, int, int, int]] = None

    # ---------------------------- Alignment ----------------------------

    def _decorrelation_stretch(self, img: np.ndarray) -> np.ndarray:
        """
        Apply a decorrelation stretch to the RGB image.
        """
        # Convert to float and reshape to (N,3)
        arr = img.astype(np.float32)
        h, w, _ = arr.shape
        flat = arr.reshape(-1, 3)
        # Center the data
        mean = np.mean(flat, axis=0)
        X = flat - mean
        # Compute covariance and eigen decomposition
        cov = np.cov(X, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        # Build scaling matrix to equalize variances
        S = np.diag(1.0 / np.sqrt(eigvals + EPSILON))
        transform = eigvecs.dot(S).dot(eigvecs.T)
        # Apply transform and re-add mean
        X2 = X.dot(transform.T)
        flat2 = X2 + mean
        arr2 = flat2.reshape(h, w, 3)
        # Clip and stretch each channel to 0-255
        result = np.zeros_like(arr2)
        for i in range(3):
            band = arr2[..., i]
            lo, hi = np.percentile(band, (1, 99))
            band = np.clip(band, lo, hi)
            band = (band - lo) * (255.0 / (hi - lo + EPSILON))
            result[..., i] = band
        return np.clip(result, 0, 255).astype(np.uint8)

    def _boost_saturation(self, img: np.ndarray, factor: float) -> np.ndarray:
        """
        Boost image saturation by a given factor via linear interpolation towards grayscale.
        """
        # Normalize to [0,1]
        img_f = img.astype(np.float32) / 255.0
        # Compute luminance (grayscale) using Rec. 601 luma coefficients
        gray = np.dot(img_f, [0.2989, 0.5870, 0.1140])[..., None]
        # Interpolate between grayscale and original by factor
        img_sat = gray + (img_f - gray) * factor
        # Clip and rescale to [0,255]
        img_sat = np.clip(img_sat, 0.0, 1.0)
        return (img_sat * 255.0).astype(np.uint8)

    def _align_bands(self, ref: np.ndarray, mov: np.ndarray) -> np.ndarray:
        """ECC alignment (if OpenCV available)."""
        if not HAS_OPENCV:
            return mov
        if ref.shape != mov.shape:
            return mov
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            _, warp = cv2.findTransformECC(ref, mov, warp, cv2.MOTION_EUCLIDEAN, ECC_CRITERIA)
            return cv2.warpAffine(mov, warp, (mov.shape[1], mov.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        except cv2.error as e:  # pragma: no cover
            logging.warning(f"ECC alignment failed: {e}")
            return mov

    def _apply_dark_flat_correction(self) -> None:
        """Apply optional dark-frame subtraction and flat-field division (normalized)."""
        if self.dark_frame is None and self.flat_field is None:
            return

        # Validate sizes
        h, w = next(iter(self.source_bands.values())).shape
        if self.dark_frame is not None and self.dark_frame.shape != (h, w):
            st.warning("Dark frame size mismatch – skipping dark correction.")
            dark = None
        else:
            dark = self.dark_frame.astype(np.float32) if self.dark_frame is not None else None

        if self.flat_field is not None and self.flat_field.shape != (h, w):
            st.warning("Flat-field size mismatch – skipping flat-field correction.")
            flat = None
        else:
            flat = self.flat_field.astype(np.float32) if self.flat_field is not None else None

        for k, v in self.source_bands.items():
            arr = v.astype(np.float32)
            if dark is not None:
                arr = arr - dark
            if flat is not None:
                denom = flat
                if dark is not None:
                    denom = denom - dark
                denom_mean = float(np.mean(denom) + EPSILON)
                arr = arr / (_safe_divide(denom, denom_mean))
            self.source_bands[k] = arr


    def _stretch_percentile_gamma(self, arr, low_p, high_p, gamma):
        """
        Apply percentile-based stretch and gamma correction to an array.
        """
        return stretch_and_gamma(arr, low_p, high_p, gamma)

    def _compute_gains(self, R: np.ndarray, G: np.ndarray, B: np.ndarray) -> list:
        """
        Compute color balance gains based on white patch, auto-balance, or manual gains.
        """
        if self.white_patch_enabled and self.white_patch_coords is not None:
            x1, y1, x2, y2 = self.white_patch_coords
            ref_means = {
                'R': float(R[y1:y2, x1:x2].mean()),
                'G': float(G[y1:y2, x1:x2].mean()),
                'B': float(B[y1:y2, x1:x1+0].mean())  # B uses same coords
            }
            target = np.mean(list(ref_means.values()))
            return [
                _safe_divide(target, ref_means['R']),
                _safe_divide(target, ref_means['G']),
                _safe_divide(target, ref_means['B']),
            ]
        if self.params.auto_balance:
            means = [float(R.mean()), float(G.mean()), float(B.mean())]
            target = np.mean([m for m in means if m > EPSILON])
            return [target / (m + EPSILON) for m in means]
        return self.params.gains or [1.0, 1.0, 1.0]
    # ------------------------------ UI --------------------------------
    def _setup_sidebar(self):
        st.sidebar.header("1. Upload Bands")
        if version.parse(st.__version__) < version.parse("1.38.0"):
            st.sidebar.warning(
                "⚠️ Please upgrade Streamlit to >=1.38.0 to avoid Plotly flicker issues",
                icon="⚠️"
            )

        uploads = st.sidebar.file_uploader(
            "Upload 3 TIFF files per composite", type=["tif", "tiff"], accept_multiple_files=True
        )
        if not uploads:
            st.info("Please upload your monochrome TIFF files to begin.")
            st.stop()

        if len(uploads) % 3 != 0:
            st.warning("Number of files is not a multiple of 3. Please upload triplets.", icon="⚠️")
            st.stop()

        self.raw_bands = {f.name: _read_tiff(f) for f in uploads}
        shapes = [b.shape for b in self.raw_bands.values()]
        if len(set(shapes)) != 1:
            st.error(f"Uploaded bands have mismatched shapes: {shapes}")
            st.stop()
        all_names = sorted(list(self.raw_bands.keys()))

        self.params.batch_mode = False
        if len(uploads) > 3:
            self.params.batch_mode = st.sidebar.checkbox(
                "Batch mode (process all triplets)", value=False,
                help="Processes all uploaded files in sorted filename groups of three."
            )
        if self.params.batch_mode:
            st.sidebar.info("Batch mode enabled. Bands will be auto-detected by name ('LWP', 'BP', 'SWP').")
            return

        with st.sidebar.expander("Preview Uploaded Bands", expanded=False):
            for name, arr in self.raw_bands.items():
                preview = self._stretch_percentile_gamma(arr, 2, 98, 1.0)
                st.image(preview, caption=name, use_container_width=True)

        st.sidebar.header("2. Channel Mapping")
        r_idx = next((i for i, n in enumerate(all_names) if 'lwp' in n.lower()), 0)
        g_idx = next((i for i, n in enumerate(all_names) if 'bp' in n.lower()), 1)
        b_idx = next((i for i, n in enumerate(all_names) if 'swp' in n.lower()), 2)
        r_name = st.sidebar.selectbox("Red channel (LWP)", all_names, index=r_idx)
        g_name = st.sidebar.selectbox("Green channel (BP)", all_names, index=g_idx)
        b_name = st.sidebar.selectbox("Blue channel (SWP)", all_names, index=b_idx)
        self.params.channel_map = {'R': r_name, 'G': g_name, 'B': b_name}

        # --- Dark / Flat-field correction hooks ---
        with st.sidebar.expander("Dark / Flat-field correction", expanded=False):
            dark_up = st.file_uploader("Dark frame (TIFF, same size)", type=["tif", "tiff"], key="dark_frame")
            flat_up = st.file_uploader("Flat-field (TIFF, same size)", type=["tif", "tiff"], key="flat_field")
            if dark_up is not None:
                self.dark_frame = _read_tiff(dark_up)
            if flat_up is not None:
                self.flat_field = _read_tiff(flat_up)

        # --- White-patch calibration controls ---
        with st.sidebar.expander("White-patch calibration", expanded=False):
            self.white_patch_enabled = st.checkbox("Enable white patch ROI calibration", value=False)
            if self.white_patch_enabled:
                h, w = next(iter(self.raw_bands.values())).shape
                x1, x2 = st.slider("WP X range (cols)", 0, w-1, (int(w*0.45), int(w*0.55)))
                y1, y2 = st.slider("WP Y range (rows)", 0, h-1, (int(h*0.45), int(h*0.55)))
                self.white_patch_coords = (x1, y1, x2, y2)
            else:
                self.white_patch_coords = None

        st.sidebar.header("3. Image Adjustments")
        with st.sidebar.expander("Per-Band Stretch & Gamma", expanded=True):
            r_low, r_high = st.slider("R percentiles", 0, 100, (2, 98), 1)
            r_gamma = st.slider("R gamma", 0.10, 3.0, 1.0, 0.01)
            g_low, g_high = st.slider("G percentiles", 0, 100, (2, 98), 1)
            g_gamma = st.slider("G gamma", 0.10, 3.0, 1.0, 0.01)
            b_low, b_high = st.slider("B percentiles", 0, 100, (2, 98), 1)
            b_gamma = st.slider("B gamma", 0.10, 3.0, 1.0, 0.01)
            self.params.stretch = {
                'R': (r_low, r_high, r_gamma), 'G': (g_low, g_high, g_gamma), 'B': (b_low, b_high, b_gamma)
            }

        with st.sidebar.expander("Color Balance", expanded=True):
            self.params.auto_balance = st.checkbox("Apply gray-world balance", value=True)
            if not self.params.auto_balance:
                r_gain = st.slider("R gain", 0.1, 3.0, 1.0, 0.01)
                g_gain = st.slider("G gain", 0.1, 3.0, 1.0, 0.01)
                b_gain = st.slider("B gain", 0.1, 3.0, 1.0, 0.01)
                self.params.gains = [r_gain, g_gain, b_gain]

        with st.sidebar.expander("Advanced Tweaks", expanded=False):
            self.params.use_decor = st.checkbox("Decorrelation stretch", value=False)
            self.params.saturation = st.slider("Saturation boost", 0.0, 3.0, 1.0, 0.01)
            self.params.pre_equalize = st.checkbox(
                "Normalize band means (pre-stretch)", value=True,
                help="Scales raw bands so their global means match before other processing."
            )
            self.params.make_pca_helper = st.checkbox(
                "Generate PCA helper composite", value=False,
                help="Creates an extra RGB image from principal components to highlight subtle variance."
            )
            if self.params.make_pca_helper:
                with st.sidebar.expander("PCA helper options", expanded=False):
                    scale_mode = st.selectbox("PC scaling", ["minmax", "zscore"], index=0,
                                               help="MinMax scales each PC to 0-1. Z-score standardizes and clips at 3 sigma.")
                    pc_choices = ["PC1", "PC2", "PC3"]
                    pc_r = st.selectbox("Red channel =", pc_choices, index=0)
                    pc_g = st.selectbox("Green channel =", pc_choices, index=1)
                    pc_b = st.selectbox("Blue channel =", pc_choices, index=2)
                    show_gray = st.checkbox("Show PCs as grayscale tabs", value=True)
                    whiten_pca = st.checkbox("Whiten PCA components", value=False)
                    self.params.pca_opts = PCAOptions(
                        scale_mode=scale_mode,
                        map={'R': pc_r, 'G': pc_g, 'B': pc_b},
                        show_gray=show_gray,
                        whiten=whiten_pca
                    )
            self.params.make_ratio = st.checkbox(
                "Generate ratio composite", value=False,
                help="Builds an RGB image from useful band ratios (R/G, B/R, (SWP-BP)/(SWP+BP))."
            )
            with st.sidebar.expander("Ratio Composite Options", expanded=False):
                ratio_choices = [
                    "Default (R/G, B/R, (SWP-BP)/(SWP+BP))",
                    "Alt1 (R/(G+B), G/(R+B), B/(R+G))",
                    "Alt2 ((R-G)/(R+G), (G-B)/(G+B), (B-R)/(B+R))"
                ]
                selected = st.selectbox("Select ratio formula", ratio_choices, index=0)
                rp_low, rp_high = st.slider("Ratio composite percentiles", 0.0, 1.0, (0.02, 0.98), 0.01)
                r_gamma = st.slider("Ratio composite gamma", 0.10, 3.0, 1.0, 0.01)
                self.params.ratio_opts = RatioOptions(formula=selected, low=rp_low, high=rp_high, gamma=r_gamma)

        st.session_state["pc_debug"] = st.sidebar.checkbox("Debug PCA/ratio internals", value=st.session_state["pc_debug"])

    # --- Main Processing and Display Logic ---
    def _process_image(self) -> np.ndarray:
        """
        Runs the full image processing pipeline based on user parameters.
        This function is the core of the application's image manipulation logic.

        Returns:
            np.ndarray: The final, processed (H, W, 3) uint8 RGB image.
        """
        # Step 1: Map uploaded files to bands
        try:
            self.source_bands = {
                'R': self.raw_bands[self.params.channel_map['R']],
                'G': self.raw_bands[self.params.channel_map['G']],
                'B': self.raw_bands[self.params.channel_map['B']],
            }
        except KeyError:
            st.error("A selected band file is no longer available. Please check uploads and mappings.")
            st.stop()

        # Optional dark / flat corrections (raw domain)
        self._apply_dark_flat_correction()

        # ... rest of the processing logic ...
        # Step 1.5: Optional pre-stretch normalization
        if self.params.pre_equalize:
            means = {k: float(v.mean()) for k, v in self.source_bands.items()}
            target_mean = np.mean(list(means.values()))
            self.source_bands = {
                k: v.astype(np.float32) * _safe_divide(target_mean, means[k])
                for k, v in self.source_bands.items()
            }

        # Step 2: Per-band stretch & gamma
        stretched_channels = {}
        for chan in ['R', 'G', 'B']:
            p = self.params.stretch[chan]
            stretched_channels[chan] = self._stretch_percentile_gamma(self.source_bands[chan], p[0], p[1], p[2])

        # Step 3: Color balance
        R, G, B = stretched_channels['R'], stretched_channels['G'], stretched_channels['B']
        gains = self._compute_gains(R, G, B)
        R = np.clip(R.astype(np.float32) * gains[0], 0, 255).astype(np.uint8)
        G = np.clip(G.astype(np.float32) * gains[1], 0, 255).astype(np.uint8)
        B = np.clip(B.astype(np.float32) * gains[2], 0, 255).astype(np.uint8)

        # Assert channel shape alignment
        if not (R.shape == G.shape == B.shape):
            st.error(f"Channel shape mismatch: R{R.shape}, G{G.shape}, B{B.shape}")
            st.stop()
        logging.debug(f"Verified channel shapes: {R.shape}")

        # Store post-gain channels for ROI stats
        self.processed_channels = {'R': R, 'G': G, 'B': B}
        rgb = np.dstack([R, G, B])

        # Step 4: Optional PCA helper composite generation
        if self.params.make_pca_helper:
            self._generate_pca_composite()

        # Optional Ratio Composite
        if self.params.make_ratio:
            self._generate_ratio_composite()

        # Step 5: Optional advanced final tweaks
        if self.params.use_decor:
            rgb = self._decorrelation_stretch(rgb)
        if self.params.saturation != 1.0:
            rgb = self._boost_saturation(rgb, self.params.saturation)
            
        return rgb
    
    def _generate_pca_composite(self):
        """Calculates and stores the PCA components and the PCA RGB helper image."""
        # Build data matrix and apply PCA via scikit-learn
        stack = np.dstack([
            self.source_bands['R'],
            self.source_bands['G'],
            self.source_bands['B']
        ]).astype(np.float32)
        h, w, _ = stack.shape
        X = stack.reshape(-1, 3)
        # Center and run PCA with optional whitening
        X_centered = X - X.mean(axis=0)
        pca = PCA(n_components=3, whiten=self.params.pca_opts.whiten)
        Y = pca.fit_transform(X_centered)
        self.pca_explained = pca.explained_variance_ratio_

        # Reshape per-component images
        pcs = [Y[:, i].reshape(h, w) for i in range(3)]

        # Scale components for visualization
        opts = self.params.pca_opts
        scale_mode = opts.scale_mode
        scaled_pcs = []
        for comp in pcs:
            if scale_mode == 'zscore':
                mean, std = comp.mean(), comp.std() + EPSILON
                comp_norm = np.clip((comp - mean) / (3 * std), -1, 1)
                comp_norm = (comp_norm + 1) / 2.0
            else:
                comp_norm = (comp - comp.min()) / (comp.max() - comp.min() + EPSILON)
            scaled_pcs.append((comp_norm * 255).astype(np.uint8))
        self.pca_components = scaled_pcs

        # Map scaled components to RGB channels
        mapping = opts.map
        idx_map = {'PC1': 0, 'PC2': 1, 'PC3': 2}
        self.pca_rgb = np.dstack([
            scaled_pcs[idx_map[mapping['R']]],
            scaled_pcs[idx_map[mapping['G']]],
            scaled_pcs[idx_map[mapping['B']]],
        ])


    def _generate_ratio_composite(self):
        """Builds a 3-channel ratio composite:
        R = R/G, G = B/R (using processed channels), B = (SWP-BP)/(SWP+BP) (using raw bands).
        Each channel is independently normalized to 0-255.
        """
        # Allow different ratio formulas
        formula = self.params.ratio_opts.formula
        # DEBUG: Show selected formula and channel ranges
        import streamlit as st
        st.write(f"🔧 Ratio formula: {formula}")
        # Define raw bands for SWP and BP
        chan_map = self.params.channel_map
        raw_bp = self.raw_bands[chan_map['G']].astype(np.float32)
        raw_swp = self.raw_bands[chan_map['B']].astype(np.float32)
        if formula.startswith("Default"):
            ch_r = _safe_divide(self.processed_channels['R'].astype(np.float32), self.processed_channels['G'].astype(np.float32))
            ch_g = _safe_divide(self.processed_channels['B'].astype(np.float32), self.processed_channels['R'].astype(np.float32))
            ch_b = _safe_divide(raw_swp - raw_bp, raw_swp + raw_bp)
        elif formula.startswith("Alt1"):
            R = self.processed_channels['R'].astype(np.float32)
            G = self.processed_channels['G'].astype(np.float32)
            B = self.processed_channels['B'].astype(np.float32)
            ch_r = _safe_divide(R, G + B)
            ch_g = _safe_divide(G, R + B)
            ch_b = _safe_divide(B, R + G)
        elif formula.startswith("Alt2"):
            R = self.processed_channels['R'].astype(np.float32)
            G = self.processed_channels['G'].astype(np.float32)
            B = self.processed_channels['B'].astype(np.float32)
            ch_r = _safe_divide(R - G, R + G)
            ch_g = _safe_divide(G - B, G + B)
            ch_b = _safe_divide(B - R, B + R)
        else:
            # Fallback to default
            ch_r = _safe_divide(self.processed_channels['R'].astype(np.float32), self.processed_channels['G'].astype(np.float32))
            ch_g = _safe_divide(self.processed_channels['B'].astype(np.float32), self.processed_channels['R'].astype(np.float32))
            ch_b = _safe_divide(raw_swp - raw_bp, raw_swp + raw_bp)

        # Apply user-defined percentile clipping and gamma to ratio channels
        ro = self.params.ratio_opts
        lo_p, hi_p = ro.low * 100, ro.high * 100

        ch_r = stretch_and_gamma((ch_r*255.0).astype(np.uint8), lo_p, hi_p, ro.gamma)
        ch_g = stretch_and_gamma((ch_g*255.0).astype(np.uint8), lo_p, hi_p, ro.gamma)
        ch_b = stretch_and_gamma((ch_b*255.0).astype(np.uint8), lo_p, hi_p, ro.gamma)

        # DEBUG: Print raw min/max for each channel before normalization
        for name, arr in [("R", ch_r), ("G", ch_g), ("B", ch_b)]:
            st.write(f"🔍 {name} channel before norm: min={float(np.nanmin(arr)):.4f}, max={float(np.nanmax(arr)):.4f}")

        self.ratio_rgb = np.dstack([ch_r, ch_g, ch_b])


    def _display_roi_analysis(self, final_rgb: np.ndarray) -> np.ndarray:
        """Handles the UI and logic for ROI selection and analysis."""
        st.subheader("Region of Interest (ROI) Analysis")
        enable_roi = st.checkbox("Enable ROI controls", value=False)
        if not enable_roi:
            return final_rgb

        h, w = final_rgb.shape[:2]
        c1, c2 = st.columns(2)
        with c1:
            x1, x2 = st.slider("X range (columns)", 0, w-1, (int(w*0.25), int(w*0.75)))
        with c2:
            y1, y2 = st.slider("Y range (rows)", 0, h-1, (int(h*0.25), int(h*0.75)))

        # Get ROI data from all relevant stages
        R, G, B = self.processed_channels['R'], self.processed_channels['G'], self.processed_channels['B']
        roi_R, roi_G, roi_B = R[y1:y2, x1:x2], G[y1:y2, x1:x2], B[y1:y2, x1:x2]

        chan_map = self.params.channel_map
        raw_roi_LWP = self.raw_bands[chan_map['R']][y1:y2, x1:x2]
        raw_roi_BP  = self.raw_bands[chan_map['G']][y1:y2, x1:x2]
        raw_roi_SWP = self.raw_bands[chan_map['B']][y1:y2, x1:x2]

        # Calculate and Display Stats
        def get_stats(a: np.ndarray) -> Dict[str, float]:
            return {
                "mean": float(a.mean()), "median": float(np.median(a)),
                "min": float(a.min()), "max": float(a.max()),
                "std": float(a.std())
            }

        st.write("Post-Stretch ROI Channel Stats")
        df = pd.DataFrame([get_stats(roi_R), get_stats(roi_G), get_stats(roi_B)], index=["R", "G", "B"])
        st.dataframe(df.style.format("{:.2f}"))

        with st.expander("Raw Pre-Stretch ROI Channel Stats"):
            df_raw = pd.DataFrame([
                get_stats(raw_roi_LWP), get_stats(raw_roi_BP), get_stats(raw_roi_SWP)
            ], index=["LWP_raw", "BP_raw", "SWP_raw"])
            st.dataframe(df_raw.style.format("{:.2f}"))

        metrics = self._calculate_roi_metrics(roi_R.mean(), roi_G.mean(), roi_B.mean(), raw_roi_LWP, raw_roi_BP, raw_roi_SWP)
        st.write("ROI Metrics")
        st.dataframe(pd.DataFrame.from_dict(metrics, orient="index", columns=["Value"]).style.format("{:.4f}"))

        suggestion = _suggest_pigment(metrics)
        st.write(f"**Heuristic pigment match:** {suggestion}")

        # ... (ROI saving logic is unchanged)
        self._handle_roi_saving(metrics, (x1, y1, x2, y2), roi_R, roi_G, roi_B, raw_roi_LWP, raw_roi_BP, raw_roi_SWP)

        display_rgb = self._draw_roi_boxes(final_rgb.copy(), (x1, y1, x2, y2))
        return display_rgb

    def _white_patch_picker(self, img: np.ndarray) -> None:
        """Interactive white-patch picker using streamlit-image-coordinates if available.
        Updates self.white_patch_enabled and self.white_patch_coords when a point is clicked.
        Fallback: informs the user to `pip install streamlit-image-coordinates`.
        """
        st.markdown("### White patch picker (click on the image)")
        if not HAS_IMG_COORDS:
            st.info("Install `streamlit-image-coordinates` to enable click picking: `pip install streamlit-image-coordinates`.")
            return
        h, w = img.shape[:2]
        half = st.slider("Patch half-size (pixels)", 1, max(2, min(h, w) // 4), 20)
        res = img_coords(Image.fromarray(img), key="wp_click_picker")
        if res is not None:
            x, y = int(res["x"]), int(res["y"])
            x1 = max(0, x - half)
            x2 = min(w, x + half)
            y1 = max(0, y - half)
            y2 = min(h, y + half)
            self.white_patch_enabled = True
            self.white_patch_coords = (x1, y1, x2, y2)
            st.success(f"White patch set to: x=[{x1},{x2}), y=[{y1},{y2})")
            # quick visual confirmation
            overlay = img.copy()
            if HAS_OPENCV:
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), max(1, h // 400))
            else:
                overlay[y1:y2, x1:x1+2] = [255, 0, 0]
                overlay[y1:y2, x2-2:x2] = [255, 0, 0]
                overlay[y1:y1+2, x1:x2] = [255, 0, 0]
                overlay[y2-2:y2, x1:x2] = [255, 0, 0]
            st.image(overlay, caption="Picked white patch overlay", use_container_width=True)

    def _calculate_roi_metrics(self, rm, gm, bm, raw_lwp, raw_bp, raw_swp) -> Dict[str, float]:
        """Calculates all defined metrics for a given ROI's mean values."""
        total_intensity = rm + gm + bm
        rn_m = _safe_divide(rm, total_intensity)
        gn_m = _safe_divide(gm, total_intensity)
        bn_m = _safe_divide(bm, total_intensity)
        metrics: Dict[str, float] = {}
        def safe_metric(name: str, func):
            try:
                metrics[name] = float(func())
            except Exception as e:
                logging.warning(f"Error computing {name}: {e}")
                metrics[name] = np.nan
        safe_metric("(R+B)/G", lambda: _safe_divide(rm + bm, gm))
        safe_metric("B/(R+G)", lambda: _safe_divide(bm, rm + gm))
        safe_metric("R/G", lambda: _safe_divide(rm, gm))
        safe_metric("G/(R+B)", lambda: _safe_divide(gm, rm + bm))
        safe_metric("B/R", lambda: _safe_divide(bm, rm))
        safe_metric("B'/(R'+G')", lambda: _safe_divide(bn_m, rn_m + gn_m))
        safe_metric("(SWP-BP)/(SWP+BP)", lambda: np.mean(_safe_divide(raw_swp - raw_bp, raw_swp + raw_bp)))
        safe_metric("(SWP-mean(LWP,BP))/(SWP+mean(LWP,BP))", lambda: np.mean(
            _safe_divide(raw_swp - (raw_lwp + raw_bp) / 2, raw_swp + (raw_lwp + raw_bp) / 2)
        ))
        return metrics
    
    def _handle_roi_saving(self, metrics, coords, r, g, b, raw_lwp, raw_bp, raw_swp):
        """Manages the UI for saving and clearing ROIs."""
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            roi_label = st.text_input("ROI label", value=f"ROI-{len(st.session_state['rois'])+1}")
        with c2:
            if st.button("Save ROI"):
                new_roi = {
                    "label": roi_label,
                    "x1": coords[0], "y1": coords[1], "x2": coords[2], "y2": coords[3],
                    "R_mean": float(r.mean()), "G_mean": float(g.mean()), "B_mean": float(b.mean()),
                    "LWP_raw_mean": float(raw_lwp.mean()), "BP_raw_mean": float(raw_bp.mean()), "SWP_raw_mean": float(raw_swp.mean()),
                }
                new_roi.update(metrics)
                st.session_state["rois"].append(new_roi)
                st.toast(f"Saved {roi_label}")

        with c3:
            if st.session_state["rois"]:
                if st.button("Clear All ROIs"):
                    st.session_state["rois"] = []
                    st.toast("Cleared all saved ROIs")

        if st.session_state["rois"]:
            st.write("Saved ROI Data")
            df_rois = pd.DataFrame(st.session_state["rois"]).set_index("label")
            st.dataframe(df_rois)
    
    def _draw_roi_boxes(self, image: np.ndarray, current_roi_coords: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Draws saved ROIs (green) and the current active ROI (red) on the image.
        Uses cv2.rectangle if available for cleaner rendering.

        Args:
            image (np.ndarray): The (H, W, 3) image to draw on. A copy should be passed.
            current_roi_coords (Tuple[int, int, int, int]): The (x1, y1, x2, y2) of the active ROI.

        Returns:
            np.ndarray: The image with ROI boxes drawn.
        """
        thickness = max(1, image.shape[0] // 400)
        
        # Draw saved ROIs (green)
        for rinfo in st.session_state["rois"]:
            pt1, pt2 = (rinfo["x1"], rinfo["y1"]), (rinfo["x2"], rinfo["y2"])
            if HAS_OPENCV:
                cv2.rectangle(image, pt1, pt2, (0, 255, 0), thickness)
            else: # Fallback pixel manipulation
                image[pt1[1]:pt2[1], pt1[0]:pt1[0]+thickness] = [0, 255, 0] # Left
                image[pt1[1]:pt2[1], pt2[0]-thickness:pt2[0]] = [0, 255, 0] # Right
                image[pt1[1]:pt1[1]+thickness, pt1[0]:pt2[0]] = [0, 255, 0] # Top
                image[pt2[1]-thickness:pt2[1], pt1[0]:pt2[0]] = [0, 255, 0] # Bottom

        # Draw current ROI (red)
        pt1, pt2 = (current_roi_coords[0], current_roi_coords[1]), (current_roi_coords[2], current_roi_coords[3])
        if HAS_OPENCV:
            cv2.rectangle(image, pt1, pt2, (255, 0, 0), thickness)
        else: # Fallback pixel manipulation
            image[pt1[1]:pt2[1], pt1[0]:pt1[0]+thickness] = [255, 0, 0]
            image[pt1[1]:pt2[1], pt2[0]-thickness:pt2[0]] = [255, 0, 0]
            image[pt1[1]:pt1[1]+thickness, pt1[0]:pt2[0]] = [255, 0, 0]
            image[pt2[1]-thickness:pt2[1], pt1[0]:pt2[0]] = [255, 0, 0]
            
        return image

    def _pc1_thresholding_ui(self, base_rgb: np.ndarray):
        """
        Build simple masks from PC1 using either quantile cuts or Otsu thresholding,
        show an overlay, and compute the same ROI-style metrics for each class.
        """
        st.subheader("PC1 Mask Builder")
        if self.pca_components is None or len(self.pca_components) == 0:
            st.info("Enable 'Generate PCA helper composite' first.")
            return

        pc1 = self.pca_components[0]  # uint8, scaled to 0-255

        method = st.radio(
            "Thresholding method",
            ["Quantiles (3 classes)", "Otsu (2 classes)"],
            horizontal=True,
            index=0
        )

        classes = None
        if method.startswith("Quantiles"):
            q_lo, q_hi = st.slider(
                "Quantile cuts",
                0.0, 1.0, (0.10, 0.90), 0.01,
                help="Pixels <= q_lo = class 0; between = class 1; >= q_hi = class 2"
            )
            thr_lo = np.quantile(pc1, q_lo)
            thr_hi = np.quantile(pc1, q_hi)
            classes = np.full(pc1.shape, 1, dtype=np.uint8)
            classes[pc1 <= thr_lo] = 0
            classes[pc1 >= thr_hi] = 2
        else:
            # --- Otsu (2-class) ---
            if HAS_OPENCV:
                # cv2.threshold returns (retval, dst)
                _, binary = cv2.threshold(pc1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                classes = (binary > 0).astype(np.uint8)
            else:
                # Lightweight numpy implementation of Otsu
                hist, _ = np.histogram(pc1.ravel(), bins=256, range=(0, 255))
                weight1 = np.cumsum(hist)
                weight2 = np.cumsum(hist[::-1])[::-1]
                cum_mean = np.cumsum(hist * np.arange(256))
                mean_total = cum_mean[-1]

                # Avoid division by zero with EPSILON
                inter_class_var = (mean_total * weight1 - cum_mean)**2 / (weight1 * (weight2 + EPSILON) + EPSILON)
                t = np.argmax(inter_class_var[:-1])  # ignore the last bin
                classes = (pc1 > t).astype(np.uint8)

        # --- Overlay classes on image ---
        overlay = base_rgb.copy()
        palette = np.array([
            [255,   0,   0],   # class 0 -> red
            [  0, 255,   0],   # class 1 -> green
            [  0,   0, 255],   # class 2 -> blue
        ], dtype=np.uint8)
        for c in np.unique(classes):
            mask = classes == c
            color = palette[c % len(palette)]
            overlay[mask] = (overlay[mask] * 0.4 + color * 0.6).astype(np.uint8)

        st.image(overlay, caption="PC1 class overlay", use_container_width=True)

        # --- Compute metrics per class (same logic as ROI metrics) ---
        st.write("Per-class metrics (computed on the current processed image)")
        results = []
        R = self.processed_channels['R']
        G = self.processed_channels['G']
        B = self.processed_channels['B']

        chan_map = self.params.channel_map
        raw_lwp = self.raw_bands[chan_map['R']]
        raw_bp  = self.raw_bands[chan_map['G']]
        raw_swp = self.raw_bands[chan_map['B']]

        for c in np.unique(classes):
            mask = classes == c

            rm = float(R[mask].mean())
            gm = float(G[mask].mean())
            bm = float(B[mask].mean())

            metrics = self._calculate_roi_metrics(
                rm, gm, bm,
                raw_lwp[mask], raw_bp[mask], raw_swp[mask]
            )
            row = {"class": int(c), "pixels": int(mask.sum())}
            row.update(metrics)
            results.append(row)

        df = pd.DataFrame(results).set_index("class")
        st.dataframe(df.style.format("{:.4f}"))

        csv_bytes = df.to_csv().encode("utf-8")
        st.download_button(
            "Download class metrics (CSV)",
            data=csv_bytes,
            file_name="pc1_class_metrics.csv",
            mime="text/csv"
        )

    def _display_pca_tabs(self, display_rgb: np.ndarray) -> None:
        if self.pca_components is None or len(self.pca_components) < 3:
            st.warning("PCA components are not available. Enable 'Generate PCA helper composite' first.")
            return

        st.write("### Interactive PCA Scatterplot Explorer")

        # Flatten PCs for plotting
        pc1_flat = self.pca_components[0].flatten()
        pc2_flat = self.pca_components[1].flatten()
        pc3_flat = self.pca_components[2].flatten()
        df_pca = pd.DataFrame({
            'PC1': pc1_flat,
            'PC2': pc2_flat,
            'PC3': pc3_flat,
        })
        dims = ['PC1', 'PC2', 'PC3']

        c1, c2, c3 = st.columns(3)
        x_axis = c1.selectbox('X axis', dims, index=0, key='pca_x')
        y_axis = c2.selectbox('Y axis', dims, index=1, key='pca_y')
        color_axis = c3.selectbox('Color by', dims, index=2, key='pca_c')

        max_allowed = min(len(df_pca), 200_000)
        if max_allowed == 0:
            st.info("No pixels to plot.")
            return
        min_n = min(1_000, max_allowed)
        default_n = min(50_000, max_allowed)
        max_pts = st.slider('Max points to plot (for performance)', min_value=min_n, max_value=max_allowed, value=default_n, step=1000)
        if len(df_pca) > max_pts:
            df_sample = df_pca.sample(n=max_pts, random_state=1)
        else:
            df_sample = df_pca
        df_sample = df_sample.copy()
        df_sample['__orig_index__'] = df_sample.index.values

        fig = px.scatter(
            df_sample,
            x=x_axis,
            y=y_axis,
            color=color_axis,
            opacity=0.6,
            title=f"{x_axis} vs {y_axis}",
            height=600,
            custom_data=['__orig_index__']
        )
        fig.update_layout(dragmode='lasso', hovermode='closest')
        # Ensure markers are visible
        fig.update_traces(marker=dict(size=4), selector=dict(mode='markers'))

        # Display the scatter plot for selection
        st.plotly_chart(fig, use_container_width=True, key="pca_chart")
        # Capture lasso selection
        if not HAS_PLOTLY_EVENTS:
            st.info("Install `streamlit-plotly-events` to enable PCA lasso selection and overlay.")
            return
        selected_points = plotly_events(
            fig,
            click_event=False,
            select_event=True,
            key="pca_plot",
            override_height=600
        )
        if not selected_points:
            return
        # Extract original pixel indices from selection
        original_indices = np.array([int(p["customdata"][0]) for p in selected_points])

        # Shared variables for both tabs
        chan_map = self.params.channel_map
        h, w = self.raw_bands[chan_map['R']].shape
        rows, cols = np.unravel_index(original_indices, (h, w))

        tab_mean, tab_overlay = st.tabs(["Mean Signature", "Overlay"])
        with tab_mean:
            lwp_mean = self.raw_bands[chan_map['R']][rows, cols].mean()
            bp_mean  = self.raw_bands[chan_map['G']][rows, cols].mean()
            swp_mean = self.raw_bands[chan_map['B']][rows, cols].mean()
            spec_fig = px.bar(
                x=['LWP', 'BP', 'SWP'],
                y=[lwp_mean, bp_mean, swp_mean],
                labels={'x': 'Band', 'y': 'Mean Raw Value'},
                title="Mean Signature"
            )
            st.plotly_chart(spec_fig, use_container_width=True)

        with tab_overlay:
            mask = np.zeros((h, w), dtype=bool)
            mask[rows, cols] = True
            overlay = display_rgb.copy()
            red_overlay = np.zeros_like(overlay)
            red_overlay[..., 0] = 255
            overlay = np.where(mask[..., None], (overlay.astype(np.float32) * 0.5 + red_overlay.astype(np.float32) * 0.5).astype(np.uint8), overlay)
            st.image(overlay, use_container_width=True)
    def _run_batch_mode(self):
        """Process triplets in alphabetical order, package results as a zip."""
        st.subheader("Batch mode results")
        names = sorted(self.raw_bands.keys())
        if len(names) % 3 != 0:
            st.error("Batch mode expects a multiple of 3 files.")
            st.stop()
        triplets = [names[i:i+3] for i in range(0, len(names), 3)]
        results_pngs: List[Tuple[str, bytes]] = []
        for t in triplets:
            # Auto detect from names
            lwp = next((n for n in t if 'lwp' in n.lower()), t[0])
            bp  = next((n for n in t if 'bp'  in n.lower()), t[1])
            swp = next((n for n in t if 'swp' in n.lower()), t[2])
            self.params.channel_map = {'R': lwp, 'G': bp, 'B': swp}
            final_rgb = self._process_image()
            pil_img = Image.fromarray(final_rgb)
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            results_pngs.append((f"{t[0]}_{t[1]}_{t[2]}.png", buf.getvalue()))
            st.image(final_rgb, caption=f"{t}", use_container_width=True)
        # Zip for download
        zbuf = io.BytesIO()
        with zipfile.ZipFile(zbuf, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
            for name, data in results_pngs:
                z.writestr(name, data)
        st.download_button("Download all composites (zip)", data=zbuf.getvalue(), file_name="batch_false_colour.zip", mime="application/zip")

    # ----------------------------- Run -----------------------------
    def run(self):
        st.set_page_config(page_title="IR False-Colour Composer", layout="wide")
        st.title("Infrared False-Colour Composer")
        st.markdown("Upload three monochrome infrared bands, map them to RGB channels, and use the sidebar controls to generate and analyze a false-colour composite.")
        self._setup_sidebar()
        if self.params.batch_mode:
            self._run_batch_mode()
            return
        final_rgb = self._process_image()
        display_rgb = self._display_roi_analysis(final_rgb.copy())
        with st.expander("White patch: click picker (post-processing)", expanded=False):
            self._white_patch_picker(display_rgb)
        st.subheader("Results")
        tabs_list = ["False-colour", "R channel", "G channel", "B channel"]
        if getattr(self, 'ratio_rgb', None) is not None:
            tabs_list.append("Ratio Composite")
        if self.pca_rgb is not None:
            tabs_list.append("PCA Composite")
            if self.params.pca_opts.show_gray:
                tabs_list.extend(["PC1", "PC2", "PC3"])
            tabs_list.append("PC1 Mask Builder")
            tabs_list.append("PCA Scatter Analysis")
        tabs = st.tabs(tabs_list)
        # Show the false colour and channels first
        tabs[0].image(display_rgb, caption="Processed false-colour composite", use_container_width=True)
        tabs[1].image(self.processed_channels['R'], caption="Stretched R channel", use_container_width=True)
        tabs[2].image(self.processed_channels['G'], caption="Stretched G channel", use_container_width=True)
        tabs[3].image(self.processed_channels['B'], caption="Stretched B channel", use_container_width=True)
        idx = 4
        if getattr(self, 'ratio_rgb', None) is not None:
            with tabs[idx]:
                st.image(self.ratio_rgb, caption="Ratio composite", use_container_width=True)
            idx += 1
        if self.pca_rgb is not None:
            with tabs[idx]:
                st.image(self.pca_rgb, caption="PCA helper composite", use_container_width=True)
                st.text(f"Explained variance: PC1={self.pca_explained[0]:.2%}, PC2={self.pca_explained[1]:.2%}, PC3={self.pca_explained[2]:.2%}")
            idx += 1
            if self.params.pca_opts.show_gray:
                for i in range(3):
                    with tabs[idx]:
                        st.image(self.pca_components[i], caption=f"PC{i+1} grayscale", use_container_width=True)
                    idx += 1
            with tabs[idx]:
                self._pc1_thresholding_ui(display_rgb)
            idx += 1
            with tabs[idx]:
                self._display_pca_tabs(display_rgb)
        # Sidebar downloads
        st.sidebar.header("4. Download Results")
        pil_img = Image.fromarray(display_rgb)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        st.sidebar.download_button("Download Composite PNG", data=buf.getvalue(), file_name="false_colour_composite.png", mime="image/png")
        if st.session_state["rois"]:
            df_rois = pd.DataFrame(st.session_state["rois"]).set_index("label")
            csv_bytes = df_rois.to_csv().encode("utf-8")
            st.sidebar.download_button("Download ROI Data (CSV)", data=csv_bytes, file_name="roi_metrics.csv", mime="text/csv")
        md_report = _generate_markdown_report(asdict(self.params), st.session_state.get("rois", []))
        st.sidebar.download_button("Download Markdown Report", md_report.encode('utf-8'), "report.md", "text/markdown")


if __name__ == "__main__":
    app = FalseColourApp()
    app.run()

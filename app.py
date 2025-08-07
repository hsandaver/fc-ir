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
    # Use tifffile as a fallback if rasterio is not available
    try:
        import tifffile as tiff
    except ImportError:
        st.error("Please install either 'rasterio' or 'tifffile' to read TIFF images.")
        st.stop()


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
# Inspired by the goal of the paper to differentiate pigments.
PIGMENT_DB = [
    {"name": "Ultramarine", "metrics": {"R/G": 1.3, "B/R": 0.6, "(SWP-BP)/(SWP+BP)": 0.2}},
    {"name": "Azurite", "metrics": {"R/G": 0.8, "B/R": 1.1, "(SWP-BP)/(SWP+BP)": -0.1}},
    {"name": "Prussian blue", "metrics": {"R/G": 0.7, "B/R": 1.4, "(SWP-BP)/(SWP+BP)": -0.2}},
    {"name": "Phthalo blue", "metrics": {"R/G": 1.1, "B/R": 0.9, "(SWP-BP)/(SWP+BP)": 0.15}},
]

# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------

def _safe_divide(numerator: Union[np.ndarray, float], denominator: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    """Safely divide, returning 0 where the denominator is close to zero."""
    return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=(np.abs(denominator) > EPSILON))


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
    """Reads a TIFF file into a numpy array, preferring rasterio but falling back to tifffile."""
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


def _suggest_pigment(metric_dict: Dict[str, float]) -> str:
    """Suggests a pigment by finding the closest match in a simple database using cosine similarity."""
    use_keys = ["R/G", "B/R", "(SWP-BP)/(SWP+BP)"]
    db_matrix, names = [], []
    for item in PIGMENT_DB:
        db_matrix.append([item["metrics"].get(k, 0.0) for k in use_keys])
        names.append(item["name"])
    db_matrix = np.array(db_matrix)
    query_vector = np.array([metric_dict.get(k, 0.0) for k in use_keys])

    # Standardize both the database and query vectors for scale-invariant comparison
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
    """Generates a markdown report summarizing the session's parameters and ROI data."""
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
# Dataclasses for cleaner state passing & report generation
# -----------------------------------------------------------------------------

@dataclass
class PCAOptions:
    scale_mode: str = "minmax"
    map: Dict[str, str] = None
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
    stretch: Dict[str, Tuple[float, float, float]] = None
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
# Main Application Class
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

    # ---------------------------- Image Processing Core ----------------------------

    def _decorrelation_stretch(self, img: np.ndarray) -> np.ndarray:
        """Apply a decorrelation stretch to an RGB image to enhance color separation."""
        arr = img.astype(np.float32)
        h, w, _ = arr.shape
        flat = arr.reshape(-1, 3)
        mean = np.mean(flat, axis=0)
        X = flat - mean
        cov = np.cov(X, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        S = np.diag(1.0 / np.sqrt(eigvals + EPSILON))
        transform = eigvecs.dot(S).dot(eigvecs.T)
        X2 = X.dot(transform.T)
        flat2 = X2 + mean
        arr2 = flat2.reshape(h, w, 3)
        result = np.zeros_like(arr2)
        for i in range(3):
            band = arr2[..., i]
            lo, hi = np.percentile(band, (1, 99))
            band = np.clip(band, lo, hi)
            band = (band - lo) * (255.0 / (hi - lo + EPSILON))
            result[..., i] = band
        return np.clip(result, 0, 255).astype(np.uint8)

    def _boost_saturation(self, img: np.ndarray, factor: float) -> np.ndarray:
        """Boost image saturation via linear interpolation towards grayscale."""
        img_f = img.astype(np.float32) / 255.0
        gray = np.dot(img_f, [0.2989, 0.5870, 0.1140])[..., None]
        img_sat = gray + (img_f - gray) * factor
        img_sat = np.clip(img_sat, 0.0, 1.0)
        return (img_sat * 255.0).astype(np.uint8)

    def _apply_dark_flat_correction(self) -> None:
        """Apply optional dark-frame subtraction and flat-field division, as mentioned in the paper."""
        if self.dark_frame is None and self.flat_field is None:
            return

        h, w = next(iter(self.source_bands.values())).shape
        dark = self.dark_frame.astype(np.float32) if self.dark_frame is not None and self.dark_frame.shape == (h, w) else None
        flat = self.flat_field.astype(np.float32) if self.flat_field is not None and self.flat_field.shape == (h, w) else None

        for k, v in self.source_bands.items():
            arr = v.astype(np.float32)
            if dark is not None:
                arr = arr - dark
            if flat is not None:
                denom = flat
                if dark is not None:
                    denom = denom - dark
                # Normalize the flat field by its mean to preserve overall brightness
                denom_mean = float(np.mean(denom) + EPSILON)
                arr = _safe_divide(arr, _safe_divide(denom, denom_mean))
            self.source_bands[k] = np.clip(arr, 0, None)


    def _stretch_percentile_gamma(self, arr: np.ndarray, low_p: float, high_p: float, gamma: float) -> np.ndarray:
        """Apply percentile-based contrast stretch and gamma correction to an array."""
        lo, hi = np.percentile(arr, (low_p, high_p))
        arr_clipped = np.clip(arr, lo, hi)
        stretched = _safe_divide(arr_clipped - lo, hi - lo) * 255.0
        if gamma != 1.0:
            stretched = np.power(_safe_divide(stretched, 255.0), 1.0 / gamma) * 255.0
        return np.clip(stretched, 0, 255).astype(np.uint8)

    # ------------------------------ UI Configuration --------------------------------

    def _setup_sidebar(self):
        st.sidebar.header("1. Upload Bands")
        if version.parse(st.__version__) < version.parse("1.38.0"):
            st.sidebar.warning(
                "⚠️ Upgrade Streamlit to >=1.38.0 to avoid Plotly flicker issues.", icon="⚠️"
            )

        uploads = st.sidebar.file_uploader(
            "Upload TIFF files (3 per composite)", type=["tif", "tiff"], accept_multiple_files=True
        )
        if not uploads:
            st.info("Please upload your monochrome TIFF files to begin.")
            st.stop()

        self.raw_bands = {f.name: _read_tiff(f) for f in uploads}
        shapes = [b.shape for b in self.raw_bands.values()]
        if len(set(shapes)) != 1:
            st.error(f"Uploaded bands have mismatched shapes: {shapes}")
            st.stop()
        all_names = sorted(list(self.raw_bands.keys()))

        self.params.batch_mode = False
        if len(uploads) > 3 and len(uploads) % 3 == 0:
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

        st.sidebar.header("2. Channel Mapping & Calibration")
        r_idx = next((i for i, n in enumerate(all_names) if 'lwp' in n.lower()), 0)
        g_idx = next((i for i, n in enumerate(all_names) if 'bp' in n.lower()), min(1, len(all_names)-1))
        b_idx = next((i for i, n in enumerate(all_names) if 'swp' in n.lower()), min(2, len(all_names)-1))
        r_name = st.sidebar.selectbox("Red channel (LWP)", all_names, index=r_idx)
        g_name = st.sidebar.selectbox("Green channel (BP)", all_names, index=g_idx)
        b_name = st.sidebar.selectbox("Blue channel (SWP)", all_names, index=b_idx)
        self.params.channel_map = {'R': r_name, 'G': g_name, 'B': b_name}

        with st.sidebar.expander("Dark / Flat-field correction (optional)", expanded=False):
            dark_up = st.file_uploader("Upload Dark Frame (TIFF)", type=["tif", "tiff"], key="dark_frame")
            flat_up = st.file_uploader("Upload Flat-Field Frame (TIFF)", type=["tif", "tiff"], key="flat_field")
            if dark_up: self.dark_frame = _read_tiff(dark_up)
            if flat_up: self.flat_field = _read_tiff(flat_up)

        with st.sidebar.expander("White-patch calibration", expanded=False):
            self.white_patch_enabled = st.checkbox("Calibrate from white patch ROI", value=False)
            if self.white_patch_enabled:
                h, w = next(iter(self.raw_bands.values())).shape
                x1, x2 = st.slider("WP X range", 0, w-1, (int(w*0.45), int(w*0.55)))
                y1, y2 = st.slider("WP Y range", 0, h-1, (int(h*0.45), int(h*0.55)))
                self.white_patch_coords = (x1, y1, x2, y2)

        st.sidebar.header("3. Image Adjustments")
        with st.sidebar.expander("Per-Band Stretch & Gamma", expanded=True):
            r_low, r_high = st.slider("R percentiles", 0, 100, (2, 98), 1)
            r_gamma = st.slider("R gamma", 0.1, 3.0, 1.0, 0.01)
            g_low, g_high = st.slider("G percentiles", 0, 100, (2, 98), 1)
            g_gamma = st.slider("G gamma", 0.1, 3.0, 1.0, 0.01)
            b_low, b_high = st.slider("B percentiles", 0, 100, (2, 98), 1)
            b_gamma = st.slider("B gamma", 0.1, 3.0, 1.0, 0.01)
            self.params.stretch = {'R': (r_low, r_high, r_gamma), 'G': (g_low, g_high, g_gamma), 'B': (b_low, b_high, b_gamma)}

        with st.sidebar.expander("Color Balance", expanded=True):
            self.params.auto_balance = st.checkbox("Apply gray-world balance", value=True)
            if not self.params.auto_balance and not self.white_patch_enabled:
                r_gain, g_gain, b_gain = st.slider("RGB gains", 0.1, 3.0, (1.0, 1.0, 1.0), 0.01)
                self.params.gains = [r_gain, g_gain, b_gain]

        with st.sidebar.expander("Advanced Processing Tools", expanded=False):
            self.params.use_decor = st.checkbox("Decorrelation stretch", value=False, help="Enhances subtle color differences.")
            self.params.saturation = st.slider("Saturation boost", 0.0, 3.0, 1.0, 0.01)
            self.params.pre_equalize = st.checkbox("Normalize band means (pre-stretch)", value=True, help="Scales raw bands to have the same global mean before processing.")

            st.markdown("---")
            self.params.make_pca_helper = st.checkbox("Generate PCA helper composite", value=False, help="Creates an RGB image from principal components to highlight subtle variance.")
            if self.params.make_pca_helper:
                scale_mode = st.selectbox("PC scaling", ["minmax", "zscore"], index=0, help="MinMax scales each PC to 0-1. Z-score standardizes and clips at 3 sigma.")
                pc_r, pc_g, pc_b = st.selectbox("R=", ["PC1","PC2","PC3"], 0), st.selectbox("G=", ["PC1","PC2","PC3"], 1), st.selectbox("B=", ["PC1","PC2","PC3"], 2)
                show_gray, whiten_pca = st.checkbox("Show PCs as grayscale", True), st.checkbox("Whiten PCA", False)
                self.params.pca_opts = PCAOptions(scale_mode, {'R': pc_r, 'G': pc_g, 'B': pc_b}, show_gray, whiten_pca)

            self.params.make_ratio = st.checkbox("Generate ratio composite", value=False, help="Builds an RGB image from useful band ratios.")
            if self.params.make_ratio:
                formula = st.selectbox("Ratio formula", ["Default (R/G, B/R, (SWP-BP)/(SWP+BP))", "Alt1 (R/(G+B), G/(R+B), B/(R+G))", "Alt2 ((R-G)/(R+G), (G-B)/(G+B), (B-R)/(B+R))"])
                rp_low, rp_high = st.slider("Ratio percentiles", 0.0, 100.0, (2.0, 98.0), 0.1)
                r_gamma = st.slider("Ratio gamma", 0.1, 3.0, 1.0, 0.01)
                self.params.ratio_opts = RatioOptions(formula, rp_low, rp_high, r_gamma)


    # ---------------------------- Main Processing Pipeline ---------------------------
    def _process_image(self) -> np.ndarray:
        """Runs the full image processing pipeline based on user parameters."""
        # Step 1: Map uploaded files to bands and apply corrections
        self.source_bands = {k: self.raw_bands[v] for k, v in self.params.channel_map.items()}
        self._apply_dark_flat_correction()

        # Step 2: Optional pre-stretch normalization
        if self.params.pre_equalize:
            means = {k: float(v.mean()) for k, v in self.source_bands.items()}
            target_mean = np.mean(list(means.values()))
            self.source_bands = {k: v.astype(np.float32) * _safe_divide(target_mean, means[k]) for k,v in self.source_bands.items()}

        # Step 3: Per-band stretch & gamma
        stretched = {c: self._stretch_percentile_gamma(self.source_bands[c], *self.params.stretch[c]) for c in ['R', 'G', 'B']}

        # Step 4: Color balance
        R, G, B = stretched['R'], stretched['G'], stretched['B']
        if self.white_patch_enabled and self.white_patch_coords:
            x1, y1, x2, y2 = self.white_patch_coords
            ref_means = {'R': R[y1:y2,x1:x2].mean(), 'G': G[y1:y2,x1:x2].mean(), 'B': B[y1:y2,x1:x2].mean()}
            target = np.mean(list(ref_means.values()))
            gains = [_safe_divide(target, m) for m in ref_means.values()]
        elif self.params.auto_balance:
            means = [R.mean(), G.mean(), B.mean()]
            target = np.mean([m for m in means if m > EPSILON])
            gains = [_safe_divide(target, m) for m in means]
        else:
            gains = self.params.gains or [1.0, 1.0, 1.0]

        R, G, B = [np.clip(ch.astype(np.float32) * g, 0, 255).astype(np.uint8) for ch, g in zip([R, G, B], gains)]

        # Store post-gain channels for ROI stats and build the composite
        self.processed_channels = {'R': R, 'G': G, 'B': B}
        rgb = np.dstack([R, G, B])

        # Step 5: Generate helper composites (PCA, Ratios)
        if self.params.make_pca_helper: self._generate_pca_composite()
        if self.params.make_ratio: self._generate_ratio_composite()

        # Step 6: Optional final tweaks
        if self.params.use_decor: rgb = self._decorrelation_stretch(rgb)
        if self.params.saturation != 1.0: rgb = self._boost_saturation(rgb, self.params.saturation)
            
        return rgb

    def _generate_pca_composite(self):
        """Calculates PCA components and creates the PCA RGB helper image."""
        stack = np.dstack([self.source_bands[c] for c in ['R', 'G', 'B']]).astype(np.float32)
        h, w, _ = stack.shape
        X = stack.reshape(-1, 3)
        pca = PCA(n_components=3, whiten=self.params.pca_opts.whiten)
        Y = pca.fit_transform(X - X.mean(axis=0))
        self.pca_explained = pca.explained_variance_ratio_

        pcs = [Y[:, i].reshape(h, w) for i in range(3)]
        scaled_pcs = []
        for comp in pcs:
            if self.params.pca_opts.scale_mode == 'zscore':
                mean, std = comp.mean(), comp.std() + EPSILON
                comp_norm = (np.clip((comp - mean) / (3 * std), -1, 1) + 1) / 2.0
            else: # minmax
                comp_norm = (comp - comp.min()) / (comp.max() - comp.min() + EPSILON)
            scaled_pcs.append((comp_norm * 255).astype(np.uint8))
        self.pca_components = scaled_pcs

        idx_map = {'PC1': 0, 'PC2': 1, 'PC3': 2}
        self.pca_rgb = np.dstack([scaled_pcs[idx_map[self.params.pca_opts.map[c]]] for c in ['R', 'G', 'B']])

    def _generate_ratio_composite(self):
        """Builds a 3-channel ratio composite based on the selected formula."""
        formula = self.params.ratio_opts.formula
        R, G, B = [self.processed_channels[c].astype(np.float32) for c in ['R', 'G', 'B']]
        raw_bp, raw_swp = self.raw_bands[self.params.channel_map['G']], self.raw_bands[self.params.channel_map['B']]

        if formula.startswith("Default"):
            channels = [_safe_divide(R, G), _safe_divide(B, R), _safe_divide(raw_swp - raw_bp, raw_swp + raw_bp)]
        elif formula.startswith("Alt1"):
            channels = [_safe_divide(R, G + B), _safe_divide(G, R + B), _safe_divide(B, R + G)]
        else: # Alt2
            channels = [_safe_divide(R - G, R + G), _safe_divide(G - B, G + B), _safe_divide(B - R, B + R)]

        ro, norm_ch = self.params.ratio_opts, []
        for arr in channels:
            lo, hi = np.percentile(arr, (ro.low, ro.high))
            arr_clip = np.clip(_safe_divide(arr - lo, hi - lo), 0, 1)
            if ro.gamma != 1.0: arr_clip = np.power(arr_clip, 1.0 / ro.gamma)
            norm_ch.append((arr_clip * 255).astype(np.uint8))
        self.ratio_rgb = np.dstack(norm_ch)

    # ---------------------------- UI for Analysis and Interaction --------------------------

    def _display_roi_analysis(self, final_rgb: np.ndarray) -> np.ndarray:
        """Handles the UI and logic for ROI selection and analysis."""
        st.subheader("Region of Interest (ROI) Analysis")
        if not st.checkbox("Enable ROI controls", value=False): return final_rgb

        h, w = final_rgb.shape[:2]
        x1, x2 = st.slider("X range", 0, w-1, (int(w*0.25), int(w*0.75)))
        y1, y2 = st.slider("Y range", 0, h-1, (int(h*0.25), int(h*0.75)))
        
        roi_R, roi_G, roi_B = [self.processed_channels[c][y1:y2, x1:x2] for c in ['R', 'G', 'B']]
        raw_roi_bands = {k: v[y1:y2, x1:x2] for k, v in self.source_bands.items()}

        st.write("Post-Stretch ROI Channel Stats")
        df = pd.DataFrame([{"mean":a.mean(), "std":a.std()} for a in [roi_R, roi_G, roi_B]], index=["R", "G", "B"])
        st.dataframe(df.style.format("{:.2f}"))

        metrics = self._calculate_roi_metrics(roi_R.mean(), roi_G.mean(), roi_B.mean(), **raw_roi_bands)
        st.write("ROI Metrics")
        st.dataframe(pd.DataFrame.from_dict(metrics, orient="index", columns=["Value"]).style.format("{:.4f}"))
        st.write(f"**Heuristic pigment match:** { _suggest_pigment(metrics)}")

        self._handle_roi_saving(metrics, (x1, y1, x2, y2), roi_R, roi_G, roi_B, **raw_roi_bands)
        return self._draw_roi_boxes(final_rgb.copy(), (x1, y1, x2, y2))

    def _white_patch_picker(self, img: np.ndarray) -> None:
        """Interactive white-patch picker using streamlit-image-coordinates if available."""
        st.markdown("### White patch picker (click on the image)")
        if not HAS_IMG_COORDS:
            st.info("Install `streamlit-image-coordinates` to enable this: `pip install streamlit-image-coordinates`.")
            return

        h, w = img.shape[:2]
        half = st.slider("Patch half-size (pixels)", 1, min(50, h // 4, w // 4), 20)
        coords = img_coords(Image.fromarray(img), key="wp_click_picker")
        if coords:
            x, y = int(coords["x"]), int(coords["y"])
            x1, y1, x2, y2 = max(0, x-half), max(0, y-half), min(w, x+half), min(h, y+half)
            self.white_patch_enabled = True
            self.white_patch_coords = (x1, y1, x2, y2)
            st.success(f"White patch set to: x=[{x1},{x2}), y=[{y1},{y2}). Re-run to apply.")

    def _calculate_roi_metrics(self, rm, gm, bm, R, G, B) -> Dict[str, float]:
        """Calculates a set of useful metrics for a given ROI's mean values and raw bands."""
        metrics = {}
        def safe_metric(name, func):
            try: metrics[name] = float(func())
            except Exception: metrics[name] = np.nan
        safe_metric("R/G", lambda: _safe_divide(rm, gm))
        safe_metric("B/R", lambda: _safe_divide(bm, rm))
        safe_metric("(SWP-BP)/(SWP+BP)", lambda: np.mean(_safe_divide(B - G, B + G)))
        safe_metric("(LWP-BP)/(LWP+BP)", lambda: np.mean(_safe_divide(R - G, R + G)))
        return metrics
    
    def _handle_roi_saving(self, metrics, coords, r_roi, g_roi, b_roi, R, G, B):
        """Manages the UI for saving and clearing ROIs."""
        c1, c2, c3 = st.columns([2, 1, 1])
        roi_label = c1.text_input("ROI label", f"ROI-{len(st.session_state['rois'])+1}")
        if c2.button("Save ROI"):
            new_roi = {"label": roi_label, "x1": coords[0], "y1": coords[1], "x2": coords[2], "y2": coords[3],
                       "R_mean": r_roi.mean(), "G_mean": g_roi.mean(), "B_mean": b_roi.mean(),
                       "LWP_raw_mean": R.mean(), "BP_raw_mean": G.mean(), "SWP_raw_mean": B.mean()}
            st.session_state["rois"].append({**new_roi, **metrics})
            st.toast(f"Saved {roi_label}")

        if st.session_state["rois"] and c3.button("Clear ROIs"):
            st.session_state["rois"] = []
            st.toast("Cleared all saved ROIs")

        if st.session_state["rois"]:
            st.write("Saved ROI Data")
            st.dataframe(pd.DataFrame(st.session_state["rois"]).set_index("label"))
    
    def _draw_roi_boxes(self, image: np.ndarray, current_roi: Tuple[int, int, int, int]) -> np.ndarray:
        """Draws saved ROIs (green) and the current active ROI (red) on the image."""
        if not HAS_OPENCV: return image # Fallback: don't draw if opencv is missing
        thickness = max(1, image.shape[0] // 500)
        for r in st.session_state["rois"]:
            cv2.rectangle(image, (r["x1"], r["y1"]), (r["x2"], r["y2"]), (0, 255, 0), thickness)
        cv2.rectangle(image, (current_roi[0], current_roi[1]), (current_roi[2], current_roi[3]), (255, 0, 0), thickness)
        return image

    def _pc1_thresholding_ui(self, base_rgb: np.ndarray):
        """UI for creating masks from PC1 and analyzing the resulting classes."""
        st.subheader("PC1 Mask Builder")
        if self.pca_components is None:
            st.info("Enable 'Generate PCA helper composite' first.")
            return

        pc1 = self.pca_components[0]
        method = st.radio("Thresholding method", ["Quantiles (3 classes)", "Otsu (2 classes)"], horizontal=True)
        
        if method.startswith("Quantiles"):
            q_lo, q_hi = st.slider("Quantile cuts", 0.0, 1.0, (0.33, 0.66), 0.01)
            thr_lo, thr_hi = np.quantile(pc1, q_lo), np.quantile(pc1, q_hi)
            classes = np.full(pc1.shape, 1, dtype=np.uint8)
            classes[pc1 <= thr_lo], classes[pc1 >= thr_hi] = 0, 2
        else: # Otsu
            if not HAS_OPENCV:
                st.warning("Otsu thresholding requires OpenCV. Please install `opencv-python`.")
                return
            _, binary = cv2.threshold(pc1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            classes = (binary > 0).astype(np.uint8)

        palette = np.array([[255,0,0], [0,255,0], [0,0,255]], dtype=np.uint8)
        overlay = base_rgb.copy()
        for c in np.unique(classes):
            overlay[classes == c] = (overlay[classes == c] * 0.4 + palette[c] * 0.6).astype(np.uint8)
        st.image(overlay, caption="PC1 class overlay", use_container_width=True)
        
        results = []
        for c in np.unique(classes):
            mask = classes == c
            r_m, g_m, b_m = [self.processed_channels[ch][mask].mean() for ch in ['R','G','B']]
            raw_bands = {k: v[mask] for k, v in self.source_bands.items()}
            metrics = self._calculate_roi_metrics(r_m, g_m, b_m, **raw_bands)
            results.append({"class": c, "pixels": mask.sum(), **metrics})
        df = pd.DataFrame(results).set_index("class")
        st.dataframe(df.style.format("{:.4f}"))

    def _display_pca_tabs(self, display_rgb: np.ndarray) -> None:
        """Interactive PCA scatter plot for pixel selection and analysis."""
        st.write("### Interactive PCA Scatterplot Explorer")
        if self.pca_components is None:
            st.warning("PCA components not available. Enable 'Generate PCA helper composite' first.")
            return

        df_pca = pd.DataFrame({f'PC{i+1}': pc.flatten() for i, pc in enumerate(self.pca_components)})
        max_pts = st.slider('Max points to plot', 1000, len(df_pca), 50000, 1000)
        df_sample = df_pca.sample(n=max_pts, random_state=1)
        
        c1, c2, c3 = st.columns(3)
        dims = ['PC1', 'PC2', 'PC3']
        x_ax, y_ax, c_ax = c1.selectbox('X',dims,0), c2.selectbox('Y',dims,1), c3.selectbox('Color',dims,2)
        
        fig = px.scatter(df_sample, x=x_ax, y=y_ax, color=c_ax, opacity=0.6)
        fig.update_layout(dragmode='lasso')
        sel = st.plotly_chart(fig, use_container_width=True, on_select="rerun")
        
        if not sel.selection or not sel.selection['points']: return

        selected_indices = [p['pointIndex'] for p in sel.selection['points']]
        original_indices = df_sample.index[selected_indices]
        
        h, w = display_rgb.shape[:2]
        rows, cols = np.unravel_index(original_indices, (h, w))

        tab_mean, tab_overlay = st.tabs(["Mean Signature", "Overlay"])
        with tab_mean:
            means = [self.source_bands[c][rows, cols].mean() for c in ['R', 'G', 'B']]
            spec_fig = px.bar(x=['LWP', 'BP', 'SWP'], y=means, title="Mean Raw Signature of Selection")
            st.plotly_chart(spec_fig, use_container_width=True)
        with tab_overlay:
            mask = np.zeros((h, w), dtype=bool)
            mask[rows, cols] = True
            overlay = np.where(mask[..., None], (display_rgb*0.5 + [127,0,0]).astype(np.uint8), display_rgb)
            st.image(overlay, use_container_width=True)

    def _run_batch_mode(self):
        """Process triplets in alphabetical order and package results as a zip file."""
        st.subheader("Batch mode results")
        names = sorted(self.raw_bands.keys())
        if len(names) % 3 != 0: st.error("Batch mode expects a multiple of 3 files."); return
        
        zbuf = io.BytesIO()
        with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as zf:
            for i in range(0, len(names), 3):
                t = names[i:i+3]
                lwp, bp, swp = t[0], t[1], t[2] # Simple assumption
                self.params.channel_map = {'R': lwp, 'G': bp, 'B': swp}
                final_rgb = self._process_image()
                
                buf = io.BytesIO()
                Image.fromarray(final_rgb).save(buf, format="PNG")
                zf.writestr(f"composite_{i//3 + 1}.png", buf.getvalue())
                
                st.image(final_rgb, caption=f"Composite for: {', '.join(t)}", use_container_width=True)
        
        st.download_button("Download All Composites (ZIP)", zbuf.getvalue(), "batch_composites.zip", "application/zip")

    # ----------------------------- Main Execution -----------------------------
    def run(self):
        st.set_page_config(page_title="IR False-Colour Composer", layout="wide")
        st.title("Infrared False-Colour Composer & Analyzer")
        st.markdown("An interactive tool for creating and analyzing infrared false-colour composites, inspired by the methodologies for multispectral pigment analysis in art conservation.")
        
        self._setup_sidebar()
        if self.params.batch_mode:
            self._run_batch_mode()
            return

        final_rgb = self._process_image()
        display_rgb = self._display_roi_analysis(final_rgb)
        
        with st.expander("White patch: click picker (optional)", expanded=False):
            self._white_patch_picker(final_rgb)

        st.subheader("Results")
        tabs = ["False-colour", "R", "G", "B"]
        if self.ratio_rgb is not None: tabs.append("Ratio Composite")
        if self.pca_rgb is not None:
            tabs.append("PCA Composite")
            if self.params.pca_opts.show_gray: tabs.extend(["PC1", "PC2", "PC3"])
            tabs.append("PC1 Mask Builder")
            tabs.append("PCA Scatter Analysis")
        
        tab_widgets = st.tabs(tabs)
        tab_widgets[0].image(display_rgb, caption="Final Composite", use_container_width=True)
        tab_widgets[1].image(self.processed_channels['R'], caption="Stretched R channel", use_container_width=True)
        tab_widgets[2].image(self.processed_channels['G'], caption="Stretched G channel", use_container_width=True)
        tab_widgets[3].image(self.processed_channels['B'], caption="Stretched B channel", use_container_width=True)
        
        idx = 4
        if self.ratio_rgb is not None: tab_widgets[idx].image(self.ratio_rgb, "Ratio composite"); idx+=1
        if self.pca_rgb is not None:
            tab_widgets[idx].image(self.pca_rgb, "PCA helper composite"); idx+=1
            if self.params.pca_opts.show_gray:
                for i in range(3): tab_widgets[idx+i].image(self.pca_components[i], f"PC{i+1}");
                idx+=3
            with tab_widgets[idx]: self._pc1_thresholding_ui(final_rgb); idx+=1
            with tab_widgets[idx]: self._display_pca_tabs(display_rgb); idx+=1

        st.sidebar.header("4. Download Results")
        buf = io.BytesIO(); Image.fromarray(display_rgb).save(buf, "PNG")
        st.sidebar.download_button("Download Composite PNG", buf.getvalue(), "composite.png", "image/png")
        if st.session_state["rois"]:
            csv = pd.DataFrame(st.session_state["rois"]).set_index("label").to_csv().encode("utf-8")
            st.sidebar.download_button("Download ROI Data (CSV)", csv, "roi_data.csv", "text/csv")
        report = _generate_markdown_report(asdict(self.params), st.session_state["rois"])
        st.sidebar.download_button("Download Report (MD)", report.encode('utf-8'), "report.md", "text/markdown")


if __name__ == "__main__":
    app = FalseColourApp()
    app.run()

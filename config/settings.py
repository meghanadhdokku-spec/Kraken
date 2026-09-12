"""
Central configuration for Kraken.
All path, AOI, and model constants live here — import this everywhere else.
"""

import subprocess
import sys
from pathlib import Path
import os

# ── auto-install missing dependencies ────────────────────────────────────────
# Maps import name → pip install name (only differs where they don't match).
_REQUIRED = {
    "dotenv":       "python-dotenv",
    "rasterio":     "rasterio",
    "numpy":        "numpy",
    "scipy":        "scipy",
    "cv2":          "opencv-python",
    "matplotlib":   "matplotlib",
    "shapely":      "shapely",
    "geopandas":    "geopandas",
    "geodatasets":  "geodatasets",
    "pyproj":       "pyproj",
    "requests":     "requests",
    "tqdm":         "tqdm",
    "yaml":         "pyyaml",
}

def _ensure_deps() -> None:
    missing = []
    for import_name, pip_name in _REQUIRED.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pip_name)

    if missing:
        print(f"[SETUP] Installing {len(missing)} missing package(s): {', '.join(missing)}")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet"] + missing
        )
        print("[SETUP] Done — continuing.")

_ensure_deps()

from dotenv import load_dotenv
load_dotenv()

# ── Project root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

# ── Data directories ──────────────────────────────────────────────────────────
DATA_DIR        = ROOT / "data"
RAW_DIR         = DATA_DIR / "raw"
PROCESSED_DIR   = DATA_DIR / "processed"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
AIS_DIR         = DATA_DIR / "ais"

# ── Output / model dirs ───────────────────────────────────────────────────────
MODELS_DIR  = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"

# ── Copernicus credentials (from .env) ────────────────────────────────────────
COPERNICUS_USER     = os.getenv("COPERNICUS_USER", "")
COPERNICUS_PASSWORD = os.getenv("COPERNICUS_PASSWORD", "")
SENTINEL_API_URL    = os.getenv(
    "SENTINEL_API_URL",
    "https://catalogue.dataspace.copernicus.eu/odata/v1",  # Copernicus Dataspace
)

# ── Area of Interest: Gulf of Guinea ─────────────────────────────────────────
# Rough bounding box (lon_min, lat_min, lon_max, lat_max)
GOG_BBOX = (-5.0, -5.0, 10.0, 5.0)

# WKT polygon derived from GOG_BBOX (sentinelsat accepts WKT)
GOG_WKT = (
    "POLYGON(("
    "-5.0 -5.0, "
    "10.0 -5.0, "
    "10.0  5.0, "
    "-5.0  5.0, "
    "-5.0 -5.0"
    "))"
)

# ── Sentinel-1 acquisition parameters ────────────────────────────────────────
S1_PLATFORM      = "Sentinel-1"
S1_PRODUCT_TYPE  = "GRD"
S1_SENSOR_MODE   = "IW"          # Interferometric Wide swath (~250 km, 10 m)
S1_POLARISATION  = "VV VH"
S1_DEFAULT_START = "NOW-30DAYS"
S1_DEFAULT_END   = "NOW"

# ── Preprocessing ─────────────────────────────────────────────────────────────
# Output GeoTIFF resolution in metres after reprojection
TARGET_RESOLUTION_M = 10

# ── CFAR detector ─────────────────────────────────────────────────────────────
CFAR_GUARD_CELLS  = 2
CFAR_BACKGROUND_CELLS = 8
CFAR_FALSE_ALARM_RATE = 1e-6

# ── YOLOv8 ───────────────────────────────────────────────────────────────────
YOLO_MODEL_SIZE   = "n"          # nano — swap for s/m/l/x
YOLO_IMG_SIZE     = 800          # matches native xView3 chip size
YOLO_EPOCHS       = 50
YOLO_BATCH        = 16
YOLO_CONF_THRESH  = 0.25
YOLO_IOU_THRESH   = 0.45
YOLO_PRETRAINED   = f"yolov8{YOLO_MODEL_SIZE}.pt"

# ── xView3 dataset ────────────────────────────────────────────────────────────
XVIEW3_DIR        = ANNOTATIONS_DIR / "xview3"
XVIEW3_CHIP_SIZE  = 800          # native chip size (px); chips are square

# Class mapping — matches src/train/xview3_prep.py CLASS_MAP
XVIEW3_CLASS_NAMES = ["non_vessel", "vessel", "fishing_vessel"]

# xView3 confidence tiers to include in training labels
# "HIGH" only = cleanest labels; add "MEDIUM" for more data
XVIEW3_MIN_CONFIDENCE = "LOW"    # include all: HIGH, MEDIUM, LOW

# ── AIS dark vessel matching ──────────────────────────────────────────────────
GFW_API_TOKEN      = os.getenv("GFW_API_TOKEN", "")   # Global Fishing Watch API token
AIS_MATCH_RADIUS_KM = 1.0                             # km — detection ↔ AIS match radius

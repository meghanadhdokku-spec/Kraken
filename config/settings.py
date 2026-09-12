"""
Central configuration for Kraken.
All path, AOI, and model constants live here — import this everywhere else.
"""

from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# ── Project root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

# ── Data directories ──────────────────────────────────────────────────────────
DATA_DIR        = ROOT / "data"
RAW_DIR         = DATA_DIR / "raw"
PROCESSED_DIR   = DATA_DIR / "processed"
ANNOTATIONS_DIR = DATA_DIR / "annotations"

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
YOLO_IMG_SIZE     = 640
YOLO_EPOCHS       = 50
YOLO_BATCH        = 16
YOLO_CONF_THRESH  = 0.25
YOLO_IOU_THRESH   = 0.45
YOLO_PRETRAINED   = f"yolov8{YOLO_MODEL_SIZE}.pt"

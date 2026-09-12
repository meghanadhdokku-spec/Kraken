"""
Generate a synthetic Sentinel-1 processed GeoTIFF for pipeline testing.

The output is a 2-band (VV, VH) float32 GeoTIFF in EPSG:4326 covering a
small patch of the Gulf of Guinea.  Ten bright point-targets are planted at
known lon/lat positions to simulate vessel detections.

Usage:
    python scripts/make_test_scene.py [--out PATH] [--size N] [--seed S]

Output path defaults to  data/processed/synthetic_scene_processed.tif
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.settings import PROCESSED_DIR


# Gulf of Guinea patch — ocean, no land, good for testing land-mask logic
_DEFAULT_BBOX = (1.0, 1.0, 3.0, 3.0)   # west, south, east, north (degrees)

# Ten synthetic vessel positions inside the default bbox
VESSEL_POSITIONS = [
    (1.20, 1.80), (1.55, 2.10), (1.80, 1.35),
    (2.05, 2.60), (2.30, 1.90), (2.50, 2.40),
    (1.40, 2.85), (1.70, 1.55), (2.70, 1.70),
    (2.85, 2.85),
]


def make_synthetic_scene(
    out_path: Path,
    size: int        = 512,
    bbox: tuple      = _DEFAULT_BBOX,
    n_vessels: int   = 10,
    clutter_mean: float = 1.0,
    vessel_snr: float   = 50.0,
    seed: int        = 42,
) -> Path:
    """
    Write a synthetic 2-band SAR σ⁰ (dB) GeoTIFF.

    Parameters
    ----------
    out_path    : destination path for the GeoTIFF
    size        : pixel size (size × size)
    bbox        : (west, south, east, north) in degrees
    n_vessels   : number of bright targets to plant (max 10)
    clutter_mean: mean linear power of the Rayleigh-distributed background
    vessel_snr  : planted target power = clutter_mean × vessel_snr
    seed        : RNG seed for reproducibility
    """
    rng  = np.random.default_rng(seed)
    west, south, east, north = bbox

    # Background: Rayleigh clutter (exponential in power, chi-2 in amplitude)
    background = rng.exponential(scale=clutter_mean, size=(2, size, size)).astype(np.float64)

    # Plant vessels as bright 3×3 blobs at known positions
    transform = from_bounds(west, south, east, north, size, size)
    inv = ~transform
    positions_used = []
    for lon, lat in VESSEL_POSITIONS[:n_vessels]:
        col, row = inv * (lon, lat)
        col, row = int(col), int(row)
        if 2 <= row < size - 2 and 2 <= col < size - 2:
            for b in range(2):
                background[b, row - 1:row + 2, col - 1:col + 2] = clutter_mean * vessel_snr
            positions_used.append((lon, lat))

    # Convert linear power → σ⁰ dB  (same formula used in preprocessing)
    with np.errstate(divide="ignore", invalid="ignore"):
        scene_db = (10.0 * np.log10(np.where(background > 0, background, np.nan))).astype(np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver":    "GTiff",
        "dtype":     "float32",
        "count":     2,
        "width":     size,
        "height":    size,
        "crs":       CRS.from_epsg(4326),
        "transform": transform,
        "nodata":    float("nan"),
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(scene_db)

    print(f"[OK]  Synthetic scene written → {out_path}")
    print(f"      Size:     {size}×{size} px")
    print(f"      BBox:     W={west} S={south} E={east} N={north}")
    print(f"      Vessels planted at:")
    for lon, lat in positions_used:
        print(f"        lon={lon:.2f}  lat={lat:.2f}")

    return out_path


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out",  type=Path, default=PROCESSED_DIR / "synthetic_scene_processed.tif")
    p.add_argument("--size", type=int,  default=512)
    p.add_argument("--seed", type=int,  default=42)
    p.add_argument("--vessels", type=int, default=10, metavar="N",
                   help="Number of vessel targets to plant (1-10)")
    args = p.parse_args()
    make_synthetic_scene(args.out, size=args.size, n_vessels=args.vessels, seed=args.seed)


if __name__ == "__main__":
    main()

"""
Cell-Averaging CFAR (CA-CFAR) vessel detector on Sentinel-1 SAR imagery.

Window layout (half-widths in pixels):
  ┌─────────────────────────────────┐
  │        Background ring          │  ← half-width = guard + background
  │   ┌─────────────────────────┐   │
  │   │       Guard ring        │   │  ← half-width = guard
  │   │   ┌─────────────────┐   │   │
  │   │   │  Cell Under     │   │   │
  │   │   │  Test (CUT)     │   │   │
  │   │   └─────────────────┘   │   │
  │   └─────────────────────────┘   │
  └─────────────────────────────────┘

Threshold:  T = α · mean(background)
            α = N · (Pfa^(-1/N) − 1)   [Rayleigh clutter, linear power]

A pixel is a detection when its linear power exceeds T.

Detections are post-processed with:
  • minimum / maximum area filter  (removes noise and land blobs)
  • distance-based NMS             (merges nearby scatter clusters)
  • optional water mask            (suppresses land false alarms)

Usage:
    python -m src.detect.cfar_detect \
        --scene data/processed/S1A_..._processed.tif
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import rasterio
import rasterio.transform as rtransform
from scipy.ndimage import uniform_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import (
    OUTPUTS_DIR,
    CFAR_GUARD_CELLS,
    CFAR_BACKGROUND_CELLS,
    CFAR_FALSE_ALARM_RATE,
)
from src.utils.geo_utils import detections_to_geojson, save_geojson


# ── threshold maths ────────────────────────────────────────────────────────────

def _alpha(n_background: int, pfa: float) -> float:
    """
    CA-CFAR scaling factor for Rayleigh-distributed clutter (linear power).

    Derived from Pfa = (1 + α/N)^{-N}  →  α = N·(Pfa^{-1/N} − 1).
    """
    return n_background * (pfa ** (-1.0 / n_background) - 1.0)


# ── CFAR kernel ────────────────────────────────────────────────────────────────

def cfar_detect(
    image: np.ndarray,
    guard: int = CFAR_GUARD_CELLS,
    background: int = CFAR_BACKGROUND_CELLS,
    pfa: float = CFAR_FALSE_ALARM_RATE,
) -> np.ndarray:
    """
    2-D CA-CFAR on a single-band image.

    Parameters
    ----------
    image      : (H, W) float32, linear power scale (not dB).
                 NaN / nodata should be set to 0 beforehand.
    guard      : Guard-cell half-width in pixels.
    background : Background-cell half-width (added outside guard zone).
    pfa        : Target probability of false alarm.

    Returns
    -------
    Binary uint8 mask: 1 = detection, 0 = background.
    """
    total_half = guard + background
    total_side = 2 * total_half + 1
    guard_side = 2 * guard + 1

    n_bg  = total_side ** 2 - guard_side ** 2
    alpha = _alpha(n_bg, pfa)

    # Efficient box-sum via uniform_filter (mean × size² = sum)
    total_sum = uniform_filter(image, size=total_side, mode="reflect") * total_side ** 2
    guard_sum = uniform_filter(image, size=guard_side,  mode="reflect") * guard_side ** 2
    bg_mean   = (total_sum - guard_sum) / n_bg

    # Correct CA-CFAR threshold: T = α · bg_mean
    threshold  = alpha * bg_mean
    detections = (image > threshold).astype(np.uint8)
    return detections


# ── post-processing ────────────────────────────────────────────────────────────

def apply_water_mask(
    image_db: np.ndarray,
    mask: np.ndarray,
    vv_db_threshold: float = -10.0,
) -> np.ndarray:
    """
    Suppress detections on pixels whose VV σ⁰ is above a land-threshold.

    Open ocean typically returns −15 to −20 dB VV; land is ≫ −10 dB.
    This is a coarse heuristic — pair with a proper land mask in production.
    """
    land = image_db > vv_db_threshold
    return np.where(land, 0, mask).astype(np.uint8)


def connected_components(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Thin wrapper around cv2 connectedComponentsWithStats."""
    import cv2
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    return labels, stats, centroids


def _iou(a: dict, b: dict) -> float:
    """Intersection-over-Union of two (x,y,w,h) boxes."""
    ax1, ay1 = a["x"], a["y"]
    ax2, ay2 = ax1 + a["w"], ay1 + a["h"]
    bx1, by1 = b["x"], b["y"]
    bx2, by2 = bx1 + b["w"], by1 + b["h"]

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union


def nms_detections(boxes: list[dict], iou_threshold: float = 0.3) -> list[dict]:
    """
    Greedy IoU-based NMS.  Keeps the largest-area box when two overlap.
    Also merges boxes whose centroids are within 2× the largest box diagonal.
    """
    if not boxes:
        return []

    sorted_boxes = sorted(boxes, key=lambda b: b["area_px"], reverse=True)
    kept = []
    suppressed = set()

    for i, box in enumerate(sorted_boxes):
        if i in suppressed:
            continue
        kept.append(box)
        for j in range(i + 1, len(sorted_boxes)):
            if j not in suppressed and _iou(box, sorted_boxes[j]) > iou_threshold:
                suppressed.add(j)

    return kept


def filter_detections(
    mask: np.ndarray,
    min_area_px: int = 4,
    max_area_px: int = 5000,
) -> list[dict]:
    """
    Convert binary mask → filtered bounding box list.

    min_area_px  : drops noise specks
    max_area_px  : drops land/coastline blobs (vessels are small targets)
    """
    import cv2
    labels, stats, centroids = connected_components(mask)

    boxes = []
    for lbl in range(1, len(stats)):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area < min_area_px or area > max_area_px:
            continue
        boxes.append({
            "x":       int(stats[lbl, cv2.CC_STAT_LEFT]),
            "y":       int(stats[lbl, cv2.CC_STAT_TOP]),
            "w":       int(stats[lbl, cv2.CC_STAT_WIDTH]),
            "h":       int(stats[lbl, cv2.CC_STAT_HEIGHT]),
            "cx":      float(centroids[lbl, 0]),
            "cy":      float(centroids[lbl, 1]),
            "area_px": area,
        })
    return boxes


def add_geo_coords(boxes: list[dict], transform) -> list[dict]:
    """Add lon/lat centroid to each box dict."""
    for b in boxes:
        lon, lat = rtransform.xy(transform, b["cy"], b["cx"])
        b["lon"] = float(lon)
        b["lat"] = float(lat)
    return boxes


# ── I/O ────────────────────────────────────────────────────────────────────────

def save_mask(
    mask: np.ndarray,
    reference_path: Path,
    out_path: Path,
) -> None:
    with rasterio.open(reference_path) as src:
        profile = src.profile.copy()
    profile.update(count=1, dtype="uint8", nodata=0)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mask, 1)
    print(f"[OK]   Mask      → {out_path}")


def save_detections_csv(boxes: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["lon", "lat", "cx_px", "cy_px", "x", "y", "w", "h", "area_px"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for b in boxes:
            writer.writerow({
                "lon": f"{b['lon']:.6f}",
                "lat": f"{b['lat']:.6f}",
                "cx_px": f"{b['cx']:.1f}",
                "cy_px": f"{b['cy']:.1f}",
                "x": b["x"], "y": b["y"],
                "w": b["w"], "h": b["h"],
                "area_px": b["area_px"],
            })
    print(f"[OK]   CSV       → {out_path}")


# ── pipeline ───────────────────────────────────────────────────────────────────

def detect_scene(
    scene_path: Path,
    band_index: int = 1,
    guard: int = CFAR_GUARD_CELLS,
    background: int = CFAR_BACKGROUND_CELLS,
    pfa: float = CFAR_FALSE_ALARM_RATE,
    min_area_px: int = 4,
    max_area_px: int = 5000,
    use_water_mask: bool = True,
    nms_iou: float = 0.3,
    out_dir: Path = OUTPUTS_DIR,
) -> list[dict]:
    """
    Full CFAR detection pipeline for one preprocessed GeoTIFF.

    Returns a list of detection dicts with pixel + geo coordinates.
    """
    print(f"\n[CFAR] {scene_path.name}  (band {band_index})")

    with rasterio.open(scene_path) as src:
        data_db   = src.read(band_index).astype(np.float32)
        transform = src.transform

    # Convert dB → linear power; fill NaN nodata with 0
    linear = np.where(np.isnan(data_db), 0.0, 10.0 ** (data_db / 10.0))

    raw_mask = cfar_detect(linear, guard, background, pfa)
    print(f"  Raw detections (pixels): {int(raw_mask.sum())}")

    if use_water_mask:
        raw_mask = apply_water_mask(data_db, raw_mask)
        print(f"  After water mask:        {int(raw_mask.sum())}")

    boxes = filter_detections(raw_mask, min_area_px, max_area_px)
    print(f"  Connected components:     {len(boxes)}")

    boxes = nms_detections(boxes, nms_iou)
    print(f"  After NMS:               {len(boxes)}")

    boxes = add_geo_coords(boxes, transform)

    # Save outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = scene_path.stem
    save_mask(raw_mask, scene_path, out_dir / f"{stem}_cfar_mask.tif")
    save_detections_csv(boxes, out_dir / f"{stem}_cfar_detections.csv")

    geojson = detections_to_geojson(boxes, stem, detector="cfar")
    save_geojson(geojson, out_dir / f"{stem}_cfar_detections.geojson")

    print(f"  Final vessel candidates: {len(boxes)}")
    return boxes


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="CA-CFAR vessel detector on SAR σ⁰ GeoTIFF.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--scene",        type=Path,  required=True)
    p.add_argument("--band",         type=int,   default=1,
                   help="Band index (1=VV, 2=VH)")
    p.add_argument("--guard",        type=int,   default=CFAR_GUARD_CELLS)
    p.add_argument("--background",   type=int,   default=CFAR_BACKGROUND_CELLS)
    p.add_argument("--pfa",          type=float, default=CFAR_FALSE_ALARM_RATE)
    p.add_argument("--min-area",     type=int,   default=4)
    p.add_argument("--max-area",     type=int,   default=5000)
    p.add_argument("--no-water-mask", action="store_true")
    p.add_argument("--nms-iou",      type=float, default=0.3)
    p.add_argument("--out-dir",      type=Path,  default=OUTPUTS_DIR)
    return p.parse_args(argv)


def main(argv=None):
    args  = parse_args(argv)
    boxes = detect_scene(
        scene_path     = args.scene,
        band_index     = args.band,
        guard          = args.guard,
        background     = args.background,
        pfa            = args.pfa,
        min_area_px    = args.min_area,
        max_area_px    = args.max_area,
        use_water_mask = not args.no_water_mask,
        nms_iou        = args.nms_iou,
        out_dir        = args.out_dir,
    )
    if boxes:
        print(f"\n{'LON':>10}  {'LAT':>9}  {'AREA_PX':>8}")
        print("-" * 32)
        for b in boxes:
            print(f"  {b['lon']:8.4f}  {b['lat']:8.4f}  {b['area_px']:>8}")


if __name__ == "__main__":
    main()

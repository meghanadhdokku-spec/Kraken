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

Dual-band mode (default):
  CFAR is run on both VV (band 1) and VH (band 2) independently.
  The two binary masks are combined with logical OR before post-processing,
  which catches targets detectable in either polarisation.

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

def apply_land_mask(
    mask: np.ndarray,
    transform,
    crs,
    land_shp_path: "Path | None" = None,
) -> np.ndarray:
    """
    Zero CFAR detections that fall on land using Natural Earth polygons.

    Rasterizes land polygons onto the scene grid so that compact bright
    targets (vessels) are preserved regardless of their backscatter level,
    while extended land and coastal clutter is suppressed.
    """
    from src.utils.geo_utils import build_land_mask
    H, W = mask.shape
    land = build_land_mask(transform, H, W, crs, land_shp_path)
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
    linear_image: "np.ndarray | None" = None,
) -> list[dict]:
    """
    Convert binary mask → filtered bounding box list.

    min_area_px   : drops noise specks
    max_area_px   : drops land/coastline blobs (vessels are small targets)
    linear_image  : optional (H, W) linear-power array used to compute a
                    per-detection confidence score.  When provided, each box
                    gets a ``conf`` key equal to the mean linear power of the
                    component's pixels divided by the global mean power of all
                    detected pixels (proxy for the CFAR threshold ratio).
                    Values > 1.0 indicate a target brighter than average.
                    When not provided, ``conf`` is set to None.
    """
    import cv2
    labels, stats, centroids = connected_components(mask)

    # Global background reference: mean power over all detected pixels
    if linear_image is not None:
        global_mean_power = float(np.nanmean(linear_image[mask > 0])) if mask.any() else 1.0
        if global_mean_power == 0.0:
            global_mean_power = 1.0
    else:
        global_mean_power = None

    boxes = []
    for lbl in range(1, len(stats)):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if area < min_area_px or area > max_area_px:
            continue

        box: dict = {
            "x":       int(stats[lbl, cv2.CC_STAT_LEFT]),
            "y":       int(stats[lbl, cv2.CC_STAT_TOP]),
            "w":       int(stats[lbl, cv2.CC_STAT_WIDTH]),
            "h":       int(stats[lbl, cv2.CC_STAT_HEIGHT]),
            "cx":      float(centroids[lbl, 0]),
            "cy":      float(centroids[lbl, 1]),
            "area_px": area,
        }

        if linear_image is not None:
            component_pixels = linear_image[labels == lbl]
            mean_component   = float(np.nanmean(component_pixels))
            box["conf"]      = mean_component / global_mean_power
        else:
            box["conf"] = None

        boxes.append(box)
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
    fieldnames = ["lon", "lat", "cx_px", "cy_px", "x", "y", "w", "h", "area_px", "conf"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for b in boxes:
            conf_val = b.get("conf")
            writer.writerow({
                "lon": f"{b['lon']:.6f}",
                "lat": f"{b['lat']:.6f}",
                "cx_px": f"{b['cx']:.1f}",
                "cy_px": f"{b['cy']:.1f}",
                "x": b["x"], "y": b["y"],
                "w": b["w"], "h": b["h"],
                "area_px": b["area_px"],
                "conf": f"{conf_val:.4f}" if conf_val is not None else "",
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
    use_land_mask: bool = True,
    land_mask_path: "Path | None" = None,
    nms_iou: float = 0.3,
    out_dir: Path = OUTPUTS_DIR,
    dual_band: bool = True,
) -> list[dict]:
    """
    Full CFAR detection pipeline for one preprocessed GeoTIFF.

    Parameters
    ----------
    dual_band : bool, default True
        When True, run CA-CFAR on both VV (band 1) and VH (band 2) and
        combine the binary masks with logical OR before post-processing.
        This improves recall for targets that are only detectable in one
        polarisation.  When False, only band ``band_index`` is processed
        (original single-band behaviour).

    Returns a list of detection dicts with pixel + geo coordinates.
    """
    if dual_band:
        print(f"\n[CFAR] {scene_path.name}  (dual-band VV+VH)")
    else:
        print(f"\n[CFAR] {scene_path.name}  (band {band_index})")

    with rasterio.open(scene_path) as src:
        if dual_band:
            data_db_vv = src.read(1).astype(np.float32)
            data_db_vh = src.read(2).astype(np.float32)
        else:
            data_db_vv = src.read(band_index).astype(np.float32)
            data_db_vh = None
        transform = src.transform
        crs       = src.crs

    # Convert dB → linear power; fill NaN nodata with 0
    linear = np.where(np.isnan(data_db_vv), 0.0, 10.0 ** (data_db_vv / 10.0))

    if dual_band:
        linear_vh = np.where(np.isnan(data_db_vh), 0.0, 10.0 ** (data_db_vh / 10.0))
        mask_vv   = cfar_detect(linear,    guard, background, pfa)
        mask_vh   = cfar_detect(linear_vh, guard, background, pfa)
        raw_mask  = np.logical_or(mask_vv, mask_vh).astype(np.uint8)
        print(f"  VV detections (pixels):  {int(mask_vv.sum())}")
        print(f"  VH detections (pixels):  {int(mask_vh.sum())}")
        print(f"  Combined OR mask:         {int(raw_mask.sum())}")
    else:
        raw_mask = cfar_detect(linear, guard, background, pfa)
        print(f"  Raw detections (pixels): {int(raw_mask.sum())}")

    if use_land_mask:
        raw_mask = apply_land_mask(raw_mask, transform, crs, land_mask_path)
        print(f"  After land mask:         {int(raw_mask.sum())}")

    boxes = filter_detections(raw_mask, min_area_px, max_area_px,
                              linear_image=linear)
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
                   help="Band index used when --no-dual-band is set (1=VV, 2=VH)")
    p.add_argument("--guard",        type=int,   default=CFAR_GUARD_CELLS)
    p.add_argument("--background",   type=int,   default=CFAR_BACKGROUND_CELLS)
    p.add_argument("--pfa",          type=float, default=CFAR_FALSE_ALARM_RATE)
    p.add_argument("--min-area",     type=int,   default=4)
    p.add_argument("--max-area",     type=int,   default=5000)
    p.add_argument("--no-land-mask",   action="store_true",
                   help="Skip shapefile land mask (keep all detections)")
    p.add_argument("--land-mask-path", type=Path, default=None,
                   help="Path to a local land shapefile (default: Natural Earth built-in)")
    p.add_argument("--nms-iou",      type=float, default=0.3)
    p.add_argument("--out-dir",      type=Path,  default=OUTPUTS_DIR)
    p.add_argument("--no-dual-band", action="store_true",
                   help="Run CFAR on VV only instead of fusing VV+VH (dual-band is on by default)")
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
        use_land_mask  = not args.no_land_mask,
        land_mask_path = args.land_mask_path,
        nms_iou        = args.nms_iou,
        out_dir        = args.out_dir,
        dual_band      = not args.no_dual_band,
    )
    if boxes:
        print(f"\n{'LON':>10}  {'LAT':>9}  {'AREA_PX':>8}  {'CONF':>6}")
        print("-" * 42)
        for b in boxes:
            conf_str = f"{b['conf']:6.3f}" if b.get("conf") is not None else "   N/A"
            print(f"  {b['lon']:8.4f}  {b['lat']:8.4f}  {b['area_px']:>8}  {conf_str}")


if __name__ == "__main__":
    main()

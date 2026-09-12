"""
Cell-Averaging CFAR (CA-CFAR) vessel detector on SAR σ⁰ imagery.

CA-CFAR slides a window over the image:
  ┌─────────────────────────┐
  │   Background cells      │
  │   ┌───────────────┐     │
  │   │  Guard cells  │     │
  │   │  ┌─────────┐  │     │
  │   │  │   CUT   │  │     │  CUT = Cell Under Test
  │   │  └─────────┘  │     │
  │   └───────────────┘     │
  └─────────────────────────┘

A pixel is flagged when its value exceeds:
    threshold = mean(background) + α
where α is chosen to meet the target false alarm rate.

Usage:
    python -m src.detect.cfar_detect --scene data/processed/scene.tif
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import uniform_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import (
    PROCESSED_DIR,
    OUTPUTS_DIR,
    CFAR_GUARD_CELLS,
    CFAR_BACKGROUND_CELLS,
    CFAR_FALSE_ALARM_RATE,
)


# ── CFAR core ──────────────────────────────────────────────────────────────────

def _alpha_from_pfa(n_background: int, pfa: float) -> float:
    """
    Closed-form α for CA-CFAR under Rayleigh clutter (linear scale).
    α = n * (pfa^(-1/n) - 1)
    """
    return n_background * (pfa ** (-1.0 / n_background) - 1.0)


def cfar_detect(
    image: np.ndarray,
    guard: int = CFAR_GUARD_CELLS,
    background: int = CFAR_BACKGROUND_CELLS,
    pfa: float = CFAR_FALSE_ALARM_RATE,
) -> np.ndarray:
    """
    Apply 2-D CA-CFAR to a single-band image (rows, cols).

    Parameters
    ----------
    image      : 2-D float32 array in linear (not dB) scale.
    guard      : Guard cell half-width in pixels.
    background : Background cell half-width in pixels (excludes guard zone).
    pfa        : Target probability of false alarm.

    Returns
    -------
    Binary mask (uint8): 1 = detection, 0 = background.
    """
    total_half  = guard + background
    total_side  = 2 * total_half + 1
    guard_side  = 2 * guard + 1

    # number of background cells
    n_bg = total_side ** 2 - guard_side ** 2
    alpha = _alpha_from_pfa(n_bg, pfa)

    # sum over total window and guard window via uniform_filter (fast)
    total_sum = uniform_filter(image, size=total_side, mode="reflect") * total_side ** 2
    guard_sum = uniform_filter(image, size=guard_side,  mode="reflect") * guard_side ** 2

    bg_mean = (total_sum - guard_sum) / n_bg

    threshold = bg_mean * (1.0 + alpha)
    detections = (image > threshold).astype(np.uint8)
    return detections


def detections_to_bboxes(
    mask: np.ndarray,
    min_area_px: int = 4,
) -> list[dict]:
    """
    Convert a binary detection mask to bounding boxes using connected components.

    Returns list of dicts with keys: x, y, w, h, cx, cy (pixel coords).
    """
    import cv2

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    boxes = []
    for label in range(1, num_labels):  # skip background (label 0)
        area = stats[label, cv2.CC_STAT_AREA]
        if area < min_area_px:
            continue
        x  = int(stats[label, cv2.CC_STAT_LEFT])
        y  = int(stats[label, cv2.CC_STAT_TOP])
        w  = int(stats[label, cv2.CC_STAT_WIDTH])
        h  = int(stats[label, cv2.CC_STAT_HEIGHT])
        cx = float(centroids[label, 0])
        cy = float(centroids[label, 1])
        boxes.append({"x": x, "y": y, "w": w, "h": h, "cx": cx, "cy": cy, "area_px": area})

    return boxes


def pixel_to_geo(
    boxes: list[dict],
    transform,
) -> list[dict]:
    """Add lon/lat centroid to each bounding box using the rasterio transform."""
    import rasterio.transform as rt

    for box in boxes:
        lon, lat = rt.xy(transform, box["cy"], box["cx"])
        box["lon"] = lon
        box["lat"] = lat
    return boxes


# ── pipeline ───────────────────────────────────────────────────────────────────

def detect_scene(
    scene_path: Path,
    band_index: int = 1,      # 1 = VV, 2 = VH
    guard: int = CFAR_GUARD_CELLS,
    background: int = CFAR_BACKGROUND_CELLS,
    pfa: float = CFAR_FALSE_ALARM_RATE,
    min_area_px: int = 4,
    out_dir: Path = OUTPUTS_DIR,
) -> list[dict]:
    print(f"[INFO] Running CA-CFAR on {scene_path.name} (band {band_index}) …")

    with rasterio.open(scene_path) as src:
        data = src.read(band_index).astype(np.float32)
        transform = src.transform

    # dB → linear for CFAR statistics
    linear = 10.0 ** (data / 10.0)
    linear = np.nan_to_num(linear, nan=0.0)

    mask = cfar_detect(linear, guard, background, pfa)
    boxes = detections_to_bboxes(mask, min_area_px)
    boxes = pixel_to_geo(boxes, transform)

    print(f"[INFO] {len(boxes)} detection(s) found.")

    # save detection mask
    out_dir.mkdir(parents=True, exist_ok=True)
    mask_path = out_dir / f"{scene_path.stem}_cfar_mask.tif"
    with rasterio.open(scene_path) as src:
        profile = src.profile.copy()
    profile.update(count=1, dtype="uint8", nodata=0)
    with rasterio.open(mask_path, "w", **profile) as dst:
        dst.write(mask, 1)
    print(f"[OK]   Mask → {mask_path}")

    return boxes


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="CA-CFAR vessel detector.")
    parser.add_argument("--scene",      type=Path, required=True)
    parser.add_argument("--band",       type=int,  default=1, help="Band index (1=VV, 2=VH)")
    parser.add_argument("--guard",      type=int,  default=CFAR_GUARD_CELLS)
    parser.add_argument("--background", type=int,  default=CFAR_BACKGROUND_CELLS)
    parser.add_argument("--pfa",        type=float, default=CFAR_FALSE_ALARM_RATE)
    parser.add_argument("--min-area",   type=int,  default=4)
    parser.add_argument("--out-dir",    type=Path, default=OUTPUTS_DIR)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    boxes = detect_scene(
        args.scene,
        band_index=args.band,
        guard=args.guard,
        background=args.background,
        pfa=args.pfa,
        min_area_px=args.min_area,
        out_dir=args.out_dir,
    )
    if boxes:
        print("\nDetections (lon, lat, area_px):")
        for b in boxes:
            print(f"  ({b['lon']:.4f}, {b['lat']:.4f})  area={b['area_px']} px")


if __name__ == "__main__":
    main()

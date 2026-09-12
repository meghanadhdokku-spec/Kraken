"""
YOLOv8 inference on tiled SAR imagery.

The processed GeoTIFF is tiled into YOLO_IMG_SIZE × YOLO_IMG_SIZE chips,
run through the model, and detections are stitched back to image coordinates.

Usage:
    python -m src.detect.yolo_detect \
        --scene data/processed/scene.tif \
        --weights models/best.pt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import (
    OUTPUTS_DIR,
    YOLO_IMG_SIZE,
    YOLO_CONF_THRESH,
    YOLO_IOU_THRESH,
    YOLO_PRETRAINED,
)


# ── tiling helpers ─────────────────────────────────────────────────────────────

def tile_image(
    data: np.ndarray,
    tile_size: int = YOLO_IMG_SIZE,
    overlap: int = 64,
) -> list[dict]:
    """
    Yield tiles from a (C, H, W) float32 array normalised to [0, 255] uint8.

    Each tile dict has: chip (H,W,3 uint8), row_off, col_off.
    """
    _, H, W = data.shape
    stride = tile_size - overlap

    # Normalise each band to 0-255 independently
    def _norm(band: np.ndarray) -> np.ndarray:
        lo, hi = np.nanpercentile(band, 2), np.nanpercentile(band, 98)
        clipped = np.clip(band, lo, hi)
        if hi == lo:
            return np.zeros_like(clipped, dtype=np.uint8)
        return ((clipped - lo) / (hi - lo) * 255).astype(np.uint8)

    bands_u8 = [_norm(data[b]) for b in range(data.shape[0])]
    # Build 3-channel: VV, VH, VV (RGB-like)
    if len(bands_u8) >= 2:
        rgb = np.stack([bands_u8[0], bands_u8[1], bands_u8[0]], axis=-1)
    else:
        rgb = np.stack([bands_u8[0]] * 3, axis=-1)

    tiles = []
    for r in range(0, H, stride):
        for c in range(0, W, stride):
            r_end = min(r + tile_size, H)
            c_end = min(c + tile_size, W)
            chip  = rgb[r:r_end, c:c_end]
            if chip.shape[0] < 16 or chip.shape[1] < 16:
                continue
            tiles.append({"chip": chip, "row_off": r, "col_off": c})

    return tiles


def stitch_detections(
    tile_results: list[dict],
    transform,
) -> list[dict]:
    """
    Merge per-tile detections back to global pixel coords, then to lon/lat.
    """
    import rasterio.transform as rt

    all_boxes = []
    for item in tile_results:
        row_off = item["row_off"]
        col_off = item["col_off"]
        for det in item["detections"]:
            x1 = det["x1"] + col_off
            y1 = det["y1"] + row_off
            x2 = det["x2"] + col_off
            y2 = det["y2"] + row_off
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            lon, lat = rt.xy(transform, cy, cx)
            all_boxes.append({
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "conf": det["conf"],
                "cls":  det["cls"],
                "lon":  lon,
                "lat":  lat,
            })
    return all_boxes


# ── inference ─────────────────────────────────────────────────────────────────

def run_yolo_on_scene(
    scene_path: Path,
    weights: Path,
    tile_size: int = YOLO_IMG_SIZE,
    conf: float = YOLO_CONF_THRESH,
    iou: float = YOLO_IOU_THRESH,
    out_dir: Path = OUTPUTS_DIR,
) -> list[dict]:
    model = YOLO(str(weights))
    print(f"[INFO] Loaded weights: {weights}")

    with rasterio.open(scene_path) as src:
        data      = src.read().astype(np.float32)
        transform = src.transform

    tiles = tile_image(data, tile_size)
    print(f"[INFO] {len(tiles)} tile(s) to process …")

    tile_results = []
    for tile in tiles:
        chip = tile["chip"]
        results = model.predict(chip, conf=conf, iou=iou, verbose=False)
        dets = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                dets.append({
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "conf": float(box.conf[0]),
                    "cls":  int(box.cls[0]),
                })
        tile_results.append({"row_off": tile["row_off"], "col_off": tile["col_off"], "detections": dets})

    boxes = stitch_detections(tile_results, transform)
    print(f"[INFO] {len(boxes)} detection(s) after stitching.")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_txt = out_dir / f"{scene_path.stem}_yolo_detections.txt"
    with open(out_txt, "w") as f:
        f.write("lon,lat,conf,cls,x1,y1,x2,y2\n")
        for b in boxes:
            f.write(f"{b['lon']:.6f},{b['lat']:.6f},{b['conf']:.4f},{b['cls']},"
                    f"{b['x1']:.1f},{b['y1']:.1f},{b['x2']:.1f},{b['y2']:.1f}\n")
    print(f"[OK]   Detections → {out_txt}")
    return boxes


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="YOLOv8 inference on SAR scene.")
    parser.add_argument("--scene",    type=Path, required=True)
    parser.add_argument("--weights",  type=Path, default=Path(YOLO_PRETRAINED),
                        help="Path to .pt weights file")
    parser.add_argument("--tile-size", type=int, default=YOLO_IMG_SIZE)
    parser.add_argument("--conf",     type=float, default=YOLO_CONF_THRESH)
    parser.add_argument("--iou",      type=float, default=YOLO_IOU_THRESH)
    parser.add_argument("--out-dir",  type=Path,  default=OUTPUTS_DIR)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_yolo_on_scene(
        args.scene, args.weights, args.tile_size,
        args.conf, args.iou, args.out_dir,
    )


if __name__ == "__main__":
    main()

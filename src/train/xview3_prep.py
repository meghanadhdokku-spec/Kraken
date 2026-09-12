"""
xView3-SAR → YOLO dataset preparation.

Expected input layout (from the xView3 competition download):
  <xview3_dir>/
    train/
      chips/    ← 2-band GeoTIFF chips, VV=band1, VH=band2
      labels.csv
    val/
      chips/
      labels.csv
    test/
      chips/    ← no labels (held-out)

Chip filenames:  {scene_id}_{row_min:04d}_{col_min:04d}.tif
  where row_min / col_min are the chip's top-left corner in scene pixel coords
  and the chip is XVIEW3_CHIP_SIZE × XVIEW3_CHIP_SIZE pixels.

Labels CSV columns used:
  scene_id, detect_scene_row, detect_scene_column,
  is_vessel, is_fishing, vessel_length_m, confidence

Output (YOLO format):
  <xview3_dir>/
    train/images/   ← normalised 3-channel uint8 PNG
    train/labels/   ← per-chip YOLO .txt
    val/images/
    val/labels/
    dataset.yaml

Class map:
  0 = non_vessel    (is_vessel == False)
  1 = vessel        (is_vessel == True,  is_fishing != True)
  2 = fishing_vessel(is_vessel == True,  is_fishing == True)

Usage:
    python -m src.train.xview3_prep --xview3-dir data/annotations/xview3
"""

import argparse
import re
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import (
    XVIEW3_DIR,
    XVIEW3_CHIP_SIZE,
    XVIEW3_CLASS_NAMES,
    XVIEW3_MIN_CONFIDENCE,
    YOLO_IMG_SIZE,
)

# ── constants ──────────────────────────────────────────────────────────────────

CLASS_MAP: dict[tuple[bool | None, bool | None], int] = {
    # (is_vessel, is_fishing) → yolo class id
    (False, None): 0,   # non-vessel (flotsam, ambiguous)
    (True,  False): 1,  # confirmed vessel, not fishing
    (True,  True):  2,  # confirmed fishing vessel  ← primary IUU target
    (True,  None):  1,  # vessel, fishing status unknown
}

CONFIDENCE_ORDER = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}

# Vessel bounding-box aspect ratio: width ≈ length / 3 for typical ships
VESSEL_ASPECT = 3.0

# Minimum bounding-box side in pixels (2 px = ~20 m at 10 m/px)
MIN_BOX_PX = 2.0


# ── data types ─────────────────────────────────────────────────────────────────

class Detection(NamedTuple):
    scene_id:      str
    scene_row:     float      # pixel row in full scene
    scene_col:     float      # pixel col in full scene
    is_vessel:     bool | None
    is_fishing:    bool | None
    length_m:      float      # vessel length in metres; NaN if unknown
    confidence:    str        # HIGH / MEDIUM / LOW


class ChipInfo(NamedTuple):
    path:      Path
    scene_id:  str
    row_min:   int            # top-left row in scene pixel coords
    col_min:   int            # top-left col in scene pixel coords
    chip_size: int            # assumed square


# ── CSV parsing ────────────────────────────────────────────────────────────────

def _parse_bool(val: str) -> bool | None:
    """'1', '1.0', 'True' → True | '0', '0.0' → False | '' / 'nan' → None"""
    v = val.strip().lower()
    if v in ("", "nan", "none", "null"):
        return None
    try:
        return bool(int(float(v)))
    except ValueError:
        return {"true": True, "false": False}.get(v)


def _parse_float(val: str) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return float("nan")


def _confidence_ok(conf: str, min_conf: str) -> bool:
    """Return True if conf is at least as good as min_conf."""
    return CONFIDENCE_ORDER.get(conf.upper(), 99) <= CONFIDENCE_ORDER.get(min_conf.upper(), 99)


def load_labels_csv(csv_path: Path, min_confidence: str = XVIEW3_MIN_CONFIDENCE) -> list[Detection]:
    """
    Parse an xView3 ground-truth CSV and return Detection records.

    Rows without valid scene coordinates are silently dropped.
    Rows below min_confidence threshold are excluded.
    """
    import csv

    detections: list[Detection] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # normalise key names (the CSV header can have leading spaces)
            row = {k.strip(): v.strip() for k, v in row.items()}

            try:
                scene_row = float(row["detect_scene_row"])
                scene_col = float(row["detect_scene_column"])
            except (KeyError, ValueError):
                continue

            confidence = row.get("confidence", "LOW").upper()
            if not _confidence_ok(confidence, min_confidence):
                continue

            detections.append(Detection(
                scene_id   = row.get("scene_id", ""),
                scene_row  = scene_row,
                scene_col  = scene_col,
                is_vessel  = _parse_bool(row.get("is_vessel", "")),
                is_fishing = _parse_bool(row.get("is_fishing", "")),
                length_m   = _parse_float(row.get("vessel_length_m", "")),
                confidence = confidence,
            ))
    return detections


# ── chip discovery ─────────────────────────────────────────────────────────────

# Regex for chip filenames: {scene_id}_{row_min:04d}_{col_min:04d}.tif
# scene_id may itself contain underscores so we match the last two numeric parts
_CHIP_RE = re.compile(r"^(.+?)_(\d{4})_(\d{4})\.tif$", re.IGNORECASE)


def discover_chips(chips_dir: Path, chip_size: int = XVIEW3_CHIP_SIZE) -> list[ChipInfo]:
    """
    Scan chips_dir for .tif files and parse scene_id + position from filenames.

    Skips files whose names don't match the expected pattern.
    """
    infos: list[ChipInfo] = []
    for tif in sorted(chips_dir.glob("*.tif")):
        m = _CHIP_RE.match(tif.name)
        if not m:
            continue
        scene_id, row_min, col_min = m.group(1), int(m.group(2)), int(m.group(3))
        infos.append(ChipInfo(tif, scene_id, row_min, col_min, chip_size))
    return infos


# ── scene→chip coordinate mapping ─────────────────────────────────────────────

def build_scene_index(
    chips: list[ChipInfo],
) -> dict[str, list[ChipInfo]]:
    """Group ChipInfo objects by scene_id for fast lookup."""
    idx: dict[str, list[ChipInfo]] = {}
    for chip in chips:
        idx.setdefault(chip.scene_id, []).append(chip)
    return idx


def detection_to_chip(det: Detection, chip: ChipInfo) -> tuple[float, float, float, float] | None:
    """
    Map a scene-level detection to chip-relative YOLO (cx, cy, w, h) ∈ [0,1].

    Returns None if the detection falls outside this chip.
    """
    chip_size = chip.chip_size
    local_row = det.scene_row - chip.row_min
    local_col = det.scene_col - chip.col_min

    if not (0 <= local_row < chip_size and 0 <= local_col < chip_size):
        return None

    # Bounding-box size from vessel length
    length_m = det.length_m if not (det.length_m != det.length_m) else 20.0  # NaN → 20 m
    length_px = max(length_m / 10.0, MIN_BOX_PX)   # 10 m/px for S1 GRD
    width_px  = max(length_px / VESSEL_ASPECT, MIN_BOX_PX)

    cx = local_col / chip_size
    cy = local_row / chip_size
    w  = min(length_px / chip_size, 1.0)
    h  = min(width_px  / chip_size, 1.0)

    return cx, cy, w, h


def assign_class(det: Detection) -> int:
    """Map (is_vessel, is_fishing) → YOLO class id via CLASS_MAP."""
    key = (det.is_vessel, det.is_fishing)
    if key in CLASS_MAP:
        return CLASS_MAP[key]
    # Fallback: treat unknown is_fishing as None
    return CLASS_MAP.get((det.is_vessel, None), 1)


# ── image conversion ───────────────────────────────────────────────────────────

def _norm_band(band: np.ndarray, lo_pct: int = 2, hi_pct: int = 98) -> np.ndarray:
    """Percentile-clip + linear rescale → uint8 [0, 255]."""
    valid = band[np.isfinite(band)]
    if valid.size == 0:
        return np.zeros_like(band, dtype=np.uint8)
    lo = float(np.percentile(valid, lo_pct))
    hi = float(np.percentile(valid, hi_pct))
    if hi == lo:
        return np.zeros_like(band, dtype=np.uint8)
    clipped = np.clip(band, lo, hi)
    return ((clipped - lo) / (hi - lo) * 255).astype(np.uint8)


def chip_tif_to_png(chip_path: Path, out_path: Path) -> None:
    """
    Read a 2-band SAR GeoTIFF (VV, VH) and write a 3-channel uint8 PNG.

    Channel layout: [VV, VH, VV]  —  gives YOLO a pseudo-RGB image.
    Percentile normalisation is computed per-chip so each chip uses
    its full dynamic range, avoiding scene-wide brightness drift.
    """
    import rasterio
    from PIL import Image

    with rasterio.open(chip_path) as src:
        vv = src.read(1).astype(np.float32)
        vh = src.read(2).astype(np.float32) if src.count >= 2 else vv.copy()

    r = _norm_band(vv)
    g = _norm_band(vh)
    b = _norm_band(vv)

    rgb = np.stack([r, g, b], axis=-1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(out_path)


# ── YOLO label writing ─────────────────────────────────────────────────────────

def write_yolo_labels(
    chip: ChipInfo,
    detections: list[Detection],
    out_path: Path,
) -> int:
    """
    Write a YOLO .txt label file for one chip.

    Returns the number of valid detections written.
    """
    lines = []
    for det in detections:
        box = detection_to_chip(det, chip)
        if box is None:
            continue
        cls = assign_class(det)
        cx, cy, w, h = box
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""))
    return len(lines)


# ── dataset YAML ──────────────────────────────────────────────────────────────

def write_dataset_yaml(
    xview3_dir: Path,
    class_names: list[str] = XVIEW3_CLASS_NAMES,
) -> Path:
    cfg = {
        "path":  str(xview3_dir.resolve()),
        "train": "train/images",
        "val":   "val/images",
        "nc":    len(class_names),
        "names": class_names,
    }
    out = xview3_dir / "dataset.yaml"
    with open(out, "w") as f:
        yaml.dump(cfg, f, sort_keys=False)
    print(f"[OK]   dataset.yaml → {out}")
    return out


# ── full preparation pipeline ──────────────────────────────────────────────────

def prepare_split(
    split: str,
    xview3_dir: Path,
    chip_size: int = XVIEW3_CHIP_SIZE,
    min_confidence: str = XVIEW3_MIN_CONFIDENCE,
    convert_images: bool = True,
) -> dict:
    """
    Prepare one split (train / val) of the xView3 dataset.

    Returns summary counts.
    """
    chips_dir  = xview3_dir / split / "chips"
    labels_csv = xview3_dir / split / "labels.csv"
    images_dir = xview3_dir / split / "images"
    labels_dir = xview3_dir / split / "labels"

    if not chips_dir.is_dir():
        print(f"[SKIP] {split}/chips/ not found — skipping {split}")
        return {}

    chips = discover_chips(chips_dir, chip_size)
    print(f"[INFO] {split}: discovered {len(chips)} chips")

    detections: list[Detection] = []
    if labels_csv.exists():
        detections = load_labels_csv(labels_csv, min_confidence)
        print(f"[INFO] {split}: loaded {len(detections)} detection labels")

    scene_idx = build_scene_index(chips)
    det_by_scene: dict[str, list[Detection]] = {}
    for det in detections:
        det_by_scene.setdefault(det.scene_id, []).append(det)

    total_chips   = 0
    total_labels  = 0
    empty_chips   = 0

    for chip in tqdm(chips, desc=f"  {split}", unit="chip"):
        # ── label file ──
        chip_dets = det_by_scene.get(chip.scene_id, [])
        label_out = labels_dir / f"{chip.path.stem}.txt"
        n = write_yolo_labels(chip, chip_dets, label_out)
        total_labels += n
        if n == 0:
            empty_chips += 1

        # ── image file ──
        if convert_images:
            png_out = images_dir / f"{chip.path.stem}.png"
            if not png_out.exists():
                chip_tif_to_png(chip.path, png_out)

        total_chips += 1

    summary = {
        "chips":       total_chips,
        "labels":      total_labels,
        "empty_chips": empty_chips,
    }
    print(
        f"[DONE] {split}: {total_chips} chips, "
        f"{total_labels} labels, {empty_chips} background chips"
    )
    return summary


def prepare_dataset(
    xview3_dir: Path = XVIEW3_DIR,
    splits: list[str] | None = None,
    chip_size: int = XVIEW3_CHIP_SIZE,
    min_confidence: str = XVIEW3_MIN_CONFIDENCE,
    convert_images: bool = True,
) -> Path:
    """
    Prepare all splits and write dataset.yaml.  Returns path to dataset.yaml.
    """
    if splits is None:
        splits = ["train", "val"]

    print(f"\n[xView3 PREP] root={xview3_dir}")
    for split in splits:
        prepare_split(split, xview3_dir, chip_size, min_confidence, convert_images)

    yaml_path = write_dataset_yaml(xview3_dir)
    return yaml_path


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Prepare xView3-SAR dataset for YOLOv8.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--xview3-dir",    type=Path, default=XVIEW3_DIR)
    p.add_argument("--splits",        nargs="+", default=["train", "val"])
    p.add_argument("--chip-size",     type=int,  default=XVIEW3_CHIP_SIZE)
    p.add_argument("--min-confidence", default=XVIEW3_MIN_CONFIDENCE,
                   choices=["HIGH", "MEDIUM", "LOW"])
    p.add_argument("--no-images",     action="store_true",
                   help="Skip image conversion (labels only)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    prepare_dataset(
        xview3_dir    = args.xview3_dir,
        splits        = args.splits,
        chip_size     = args.chip_size,
        min_confidence= args.min_confidence,
        convert_images= not args.no_images,
    )


if __name__ == "__main__":
    main()

"""
Fine-tune YOLOv8 on xView3-SAR ship detection dataset.

xView3 ships labels as CSV; this script converts them to YOLO format
(one .txt per chip) and trains the model.

Usage:
    python -m src.train.train_yolo \
        --data-dir data/annotations/xview3 \
        --out-dir  models/yolo_run1
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import (
    ANNOTATIONS_DIR,
    MODELS_DIR,
    YOLO_PRETRAINED,
    YOLO_IMG_SIZE,
    YOLO_EPOCHS,
    YOLO_BATCH,
    YOLO_MODEL_SIZE,
)


# ── xView3 → YOLO label conversion ────────────────────────────────────────────

def convert_xview3_labels(
    csv_path: Path,
    chips_dir: Path,
    out_labels_dir: Path,
    img_size: int = YOLO_IMG_SIZE,
) -> None:
    """
    Convert xView3 CSV labels to per-chip YOLO .txt files.

    xView3 CSV columns (subset used here):
      detect_scene_row, detect_scene_column, vessel_length_m, chip_id

    YOLO format: <class> <cx_norm> <cy_norm> <w_norm> <h_norm>
    We use class 0 = vessel (single class for now).
    """
    import csv

    out_labels_dir.mkdir(parents=True, exist_ok=True)
    chip_boxes: dict[str, list] = {}

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            chip_id = row.get("chip_id") or row.get("scene_id", "unknown")
            try:
                col = float(row["detect_scene_column"])
                r   = float(row["detect_scene_row"])
                length_m = float(row.get("vessel_length_m", 20) or 20)
            except (ValueError, KeyError):
                continue

            # Approximate bounding box: length → pixels at ~10 m/px
            px_size = max(length_m / 10.0, 2.0)
            cx_norm = col / img_size
            cy_norm = r   / img_size
            w_norm  = px_size / img_size
            h_norm  = px_size / img_size

            # Clamp to [0, 1]
            cx_norm = min(max(cx_norm, 0.0), 1.0)
            cy_norm = min(max(cy_norm, 0.0), 1.0)
            w_norm  = min(max(w_norm,  1e-4), 1.0)
            h_norm  = min(max(h_norm,  1e-4), 1.0)

            chip_boxes.setdefault(chip_id, []).append(
                f"0 {cx_norm:.6f} {cy_norm:.6f} {w_norm:.6f} {h_norm:.6f}"
            )

    for chip_id, lines in chip_boxes.items():
        label_file = out_labels_dir / f"{chip_id}.txt"
        label_file.write_text("\n".join(lines) + "\n")

    print(f"[INFO] Converted {sum(len(v) for v in chip_boxes.values())} labels "
          f"across {len(chip_boxes)} chips → {out_labels_dir}")


def write_dataset_yaml(
    dataset_root: Path,
    train_img_dir: Path,
    val_img_dir: Path,
    yaml_path: Path,
    class_names: list[str] | None = None,
) -> None:
    """Write the dataset.yaml that Ultralytics YOLO expects."""
    if class_names is None:
        class_names = ["vessel"]

    cfg = {
        "path":  str(dataset_root.resolve()),
        "train": str(train_img_dir.resolve()),
        "val":   str(val_img_dir.resolve()),
        "nc":    len(class_names),
        "names": class_names,
    }
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w") as f:
        yaml.dump(cfg, f, sort_keys=False)
    print(f"[INFO] Dataset YAML → {yaml_path}")


# ── training ───────────────────────────────────────────────────────────────────

def train(
    data_yaml: Path,
    out_dir: Path = MODELS_DIR / "yolo_run",
    model_size: str = YOLO_MODEL_SIZE,
    epochs: int = YOLO_EPOCHS,
    img_size: int = YOLO_IMG_SIZE,
    batch: int = YOLO_BATCH,
    pretrained: str = YOLO_PRETRAINED,
) -> Path:
    model = YOLO(pretrained)
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=img_size,
        batch=batch,
        project=str(out_dir.parent),
        name=out_dir.name,
        exist_ok=True,
        device="auto",
    )
    best_weights = out_dir / "weights" / "best.pt"
    print(f"[DONE] Best weights → {best_weights}")
    return best_weights


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train YOLOv8 on xView3-SAR.")
    parser.add_argument("--data-dir",  type=Path, default=ANNOTATIONS_DIR / "xview3",
                        help="xView3 dataset root (expects train/ val/ subdirs + labels CSV)")
    parser.add_argument("--out-dir",   type=Path, default=MODELS_DIR / "yolo_run")
    parser.add_argument("--epochs",    type=int,  default=YOLO_EPOCHS)
    parser.add_argument("--batch",     type=int,  default=YOLO_BATCH)
    parser.add_argument("--img-size",  type=int,  default=YOLO_IMG_SIZE)
    parser.add_argument("--model-size", default=YOLO_MODEL_SIZE,
                        choices=["n", "s", "m", "l", "x"])
    parser.add_argument("--convert-labels", action="store_true",
                        help="Convert xView3 CSV labels to YOLO format first")
    parser.add_argument("--labels-csv", type=Path,
                        help="xView3 labels CSV (required with --convert-labels)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    data_dir = args.data_dir

    train_img_dir = data_dir / "train" / "images"
    val_img_dir   = data_dir / "val"   / "images"

    if args.convert_labels:
        if not args.labels_csv:
            sys.exit("[ERROR] --labels-csv required with --convert-labels")
        convert_xview3_labels(
            args.labels_csv,
            train_img_dir,
            data_dir / "train" / "labels",
            args.img_size,
        )

    yaml_path = data_dir / "dataset.yaml"
    write_dataset_yaml(data_dir, train_img_dir, val_img_dir, yaml_path)

    pretrained = f"yolov8{args.model_size}.pt"
    train(yaml_path, args.out_dir, args.model_size, args.epochs, args.img_size, args.batch, pretrained)


if __name__ == "__main__":
    main()

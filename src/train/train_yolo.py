"""
YOLOv8 fine-tuning on xView3-SAR.

Workflow:
  1. (Optional) Run xview3_prep.prepare_dataset() to build the YOLO directory.
  2. Train from a COCO-pretrained YOLOv8 checkpoint with SAR-tuned augmentation.
  3. After training, auto-validate and print mAP / class-level metrics.

SAR augmentation philosophy:
  - No HSV shifts   (SAR amplitude has no hue/saturation)
  - Flips OK        (SAR imagery has no canonical up direction)
  - 90° rotations   (vessels appear at all angles)
  - Modest scale jitter (vessels are small; large scale changes lose context)
  - No perspective warp (SAR already has range/azimuth distortion)
  - Mosaic ON       (boosts small-object detection; 4 chips per mosaic)

Usage:
    # Full run from raw xView3:
    python -m src.train.train_yolo \
        --xview3-dir data/annotations/xview3 \
        --prep        # run xview3_prep first

    # Train only (dataset already prepared):
    python -m src.train.train_yolo --data data/annotations/xview3/dataset.yaml

    # Resume interrupted run:
    python -m src.train.train_yolo \
        --data data/annotations/xview3/dataset.yaml \
        --resume models/yolo_vessel/weights/last.pt
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import (
    XVIEW3_DIR,
    MODELS_DIR,
    YOLO_PRETRAINED,
    YOLO_IMG_SIZE,
    YOLO_EPOCHS,
    YOLO_BATCH,
    YOLO_MODEL_SIZE,
    YOLO_CONF_THRESH,
    YOLO_IOU_THRESH,
    XVIEW3_CLASS_NAMES,
    XVIEW3_MIN_CONFIDENCE,
)


# ── augmentation config ────────────────────────────────────────────────────────

SAR_AUGMENT = {
    # Colour / radiometric — disabled for SAR
    "hsv_h":        0.0,
    "hsv_s":        0.0,
    "hsv_v":        0.3,    # mild brightness shift still helps
    # Geometric — all fine for SAR
    "degrees":      90.0,   # full 90° rotation steps
    "fliplr":       0.5,
    "flipud":       0.5,
    "translate":    0.1,
    "scale":        0.3,    # keep small: vessels are already tiny targets
    "shear":        0.0,
    "perspective":  0.0,
    # Mosaic / mixup
    "mosaic":       1.0,
    "mixup":        0.0,    # mixup confuses small-object labels
    "copy_paste":   0.0,
    # Other
    "erasing":      0.2,    # random erasing helps occlusion robustness
}

# Extra training kwargs forwarded to model.train()
TRAIN_KWARGS = {
    "optimizer":    "AdamW",
    "lr0":          1e-3,
    "lrf":          0.01,       # final LR = lr0 × lrf
    "momentum":     0.937,
    "weight_decay": 5e-4,
    "warmup_epochs": 3,
    "warmup_momentum": 0.8,
    "box":          7.5,        # box regression loss weight
    "cls":          0.5,
    "dfl":          1.5,
    "patience":     20,         # early stopping patience
    "save_period":  10,         # checkpoint every N epochs
    "val":          True,
    "plots":        True,
    "verbose":      True,
}


# ── training ───────────────────────────────────────────────────────────────────

def train(
    data_yaml:   Path,
    out_dir:     Path,
    model_size:  str  = YOLO_MODEL_SIZE,
    epochs:      int  = YOLO_EPOCHS,
    img_size:    int  = YOLO_IMG_SIZE,
    batch:       int  = YOLO_BATCH,
    resume_from: Path | None = None,
    device:      str  = "auto",
) -> Path:
    """
    Fine-tune YOLOv8 on xView3-SAR.

    If resume_from is given, training continues from that checkpoint.
    Returns path to the best weights file.
    """
    from ultralytics import YOLO

    if resume_from:
        print(f"[INFO] Resuming from {resume_from}")
        model = YOLO(str(resume_from))
    else:
        pretrained = f"yolov8{model_size}.pt"
        print(f"[INFO] Starting from {pretrained}")
        model = YOLO(pretrained)

    results = model.train(
        data    = str(data_yaml),
        epochs  = epochs,
        imgsz   = img_size,
        batch   = batch,
        project = str(out_dir.parent),
        name    = out_dir.name,
        exist_ok= True,
        device  = device,
        resume  = resume_from is not None,
        **SAR_AUGMENT,
        **TRAIN_KWARGS,
    )

    best = out_dir / "weights" / "best.pt"
    print(f"\n[DONE] Best weights → {best}")
    return best


# ── validation ─────────────────────────────────────────────────────────────────

def validate(
    weights:  Path,
    data_yaml: Path,
    img_size:  int   = YOLO_IMG_SIZE,
    conf:      float = YOLO_CONF_THRESH,
    iou:       float = YOLO_IOU_THRESH,
    split:     str   = "val",
) -> dict:
    """
    Run validation and return the metrics dict.

    Prints per-class mAP50 and mAP50-95.
    """
    from ultralytics import YOLO

    model   = YOLO(str(weights))
    metrics = model.val(
        data   = str(data_yaml),
        imgsz  = img_size,
        conf   = conf,
        iou    = iou,
        split  = split,
        verbose= True,
    )

    print("\n── Validation results ──────────────────────────────")
    # metrics.box holds per-class results in Ultralytics ≥ 8.0
    names = XVIEW3_CLASS_NAMES
    for i, name in enumerate(names):
        try:
            p  = float(metrics.box.p[i])
            r  = float(metrics.box.r[i])
            m50 = float(metrics.box.ap50[i])
            m5095 = float(metrics.box.ap[i])
            print(f"  {name:<20}  P={p:.3f}  R={r:.3f}  "
                  f"mAP50={m50:.3f}  mAP50-95={m5095:.3f}")
        except (IndexError, AttributeError):
            pass

    try:
        print(f"\n  ALL               "
              f"mAP50={metrics.box.map50:.3f}  "
              f"mAP50-95={metrics.box.map:.3f}")
    except AttributeError:
        pass

    return metrics


# ── ONNX export ────────────────────────────────────────────────────────────────

def export_onnx(weights: Path, img_size: int = YOLO_IMG_SIZE) -> Path:
    """Export best.pt → best.onnx for deployment."""
    from ultralytics import YOLO

    model    = YOLO(str(weights))
    out_path = model.export(format="onnx", imgsz=img_size, simplify=True)
    print(f"[OK]   ONNX model → {out_path}")
    return Path(out_path)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Fine-tune YOLOv8 on xView3-SAR.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data",       type=Path, default=None,
                   help="Path to dataset.yaml (skips --xview3-dir if given)")
    p.add_argument("--xview3-dir", type=Path, default=XVIEW3_DIR)
    p.add_argument("--out-dir",    type=Path, default=MODELS_DIR / "yolo_vessel")
    p.add_argument("--model-size", default=YOLO_MODEL_SIZE,
                   choices=["n", "s", "m", "l", "x"])
    p.add_argument("--epochs",     type=int,  default=YOLO_EPOCHS)
    p.add_argument("--batch",      type=int,  default=YOLO_BATCH)
    p.add_argument("--img-size",   type=int,  default=YOLO_IMG_SIZE)
    p.add_argument("--device",     default="auto",
                   help="cuda:0 / cpu / mps / auto")
    p.add_argument("--resume",     type=Path, default=None,
                   help="Resume from last.pt checkpoint")
    p.add_argument("--prep",       action="store_true",
                   help="Run xview3_prep before training")
    p.add_argument("--min-confidence", default=XVIEW3_MIN_CONFIDENCE,
                   choices=["HIGH", "MEDIUM", "LOW"])
    p.add_argument("--validate-only", action="store_true",
                   help="Skip training; just validate --resume weights")
    p.add_argument("--export-onnx",   action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # ── 1. Optional dataset prep ───────────────────────────────────────────────
    if args.prep:
        from src.train.xview3_prep import prepare_dataset
        data_yaml = prepare_dataset(
            xview3_dir    = args.xview3_dir,
            min_confidence= args.min_confidence,
        )
    elif args.data:
        data_yaml = args.data
    else:
        data_yaml = args.xview3_dir / "dataset.yaml"

    if not data_yaml.exists():
        sys.exit(
            f"[ERROR] dataset.yaml not found at {data_yaml}.\n"
            "  Run with --prep to generate it, or pass --data explicitly."
        )

    # ── 2. Validate only ──────────────────────────────────────────────────────
    if args.validate_only:
        if not args.resume:
            sys.exit("[ERROR] --validate-only requires --resume <weights.pt>")
        validate(args.resume, data_yaml, args.img_size)
        return

    # ── 3. Train ───────────────────────────────────────────────────────────────
    best_weights = train(
        data_yaml   = data_yaml,
        out_dir     = args.out_dir,
        model_size  = args.model_size,
        epochs      = args.epochs,
        img_size    = args.img_size,
        batch       = args.batch,
        resume_from = args.resume,
        device      = args.device,
    )

    # ── 4. Auto-validate ──────────────────────────────────────────────────────
    validate(best_weights, data_yaml, args.img_size)

    # ── 5. Optional ONNX export ───────────────────────────────────────────────
    if args.export_onnx:
        export_onnx(best_weights, args.img_size)


if __name__ == "__main__":
    main()

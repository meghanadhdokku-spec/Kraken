"""
Post-training evaluation for the Kraken vessel detector.

Computes:
  • Per-class precision, recall, F1, mAP50, mAP50-95
  • Precision-Recall curve (saved as PNG)
  • Confusion matrix
  • Maritime F-score at multiple confidence thresholds (IUU-relevant metric)

Usage:
    python -m src.train.evaluate \
        --weights models/yolo_vessel/weights/best.pt \
        --data    data/annotations/xview3/dataset.yaml \
        --split   val
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import (
    OUTPUTS_DIR,
    YOLO_IMG_SIZE,
    YOLO_CONF_THRESH,
    YOLO_IOU_THRESH,
    XVIEW3_CLASS_NAMES,
)


# ── metrics helpers ────────────────────────────────────────────────────────────

def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p  = tp / (tp + fp + 1e-9)
    r  = tp / (tp + fn + 1e-9)
    f1 = 2 * p * r / (p + r + 1e-9)
    return p, r, f1


def sweep_thresholds(
    preds: list[dict],
    gts:   list[dict],
    iou_thresh: float = 0.5,
    conf_steps: int   = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sweep confidence threshold and return (confs, precisions, recalls).

    preds: list of {'conf': float, 'cls': int, 'x1':, 'y1':, 'x2':, 'y2':}
    gts:   list of {'cls': int, 'x1':, 'y1':, 'x2':, 'y2':}
    """
    thresholds = np.linspace(0.0, 1.0, conf_steps)
    precisions = np.zeros(conf_steps)
    recalls    = np.zeros(conf_steps)

    for i, thresh in enumerate(thresholds):
        filtered = [p for p in preds if p["conf"] >= thresh]
        tp, fp, fn = _match(filtered, gts, iou_thresh)
        p, r, _ = precision_recall_f1(tp, fp, fn)
        precisions[i] = p
        recalls[i]    = r

    return thresholds, precisions, recalls


def _iou_box(a: dict, b: dict) -> float:
    ix1 = max(a["x1"], b["x1"]); iy1 = max(a["y1"], b["y1"])
    ix2 = min(a["x2"], b["x2"]); iy2 = min(a["y2"], b["y2"])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
    area_b = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
    return inter / (area_a + area_b - inter)


def _match(preds: list[dict], gts: list[dict], iou_thresh: float) -> tuple[int, int, int]:
    """Greedy TP/FP/FN matching by IoU."""
    matched_gt = set()
    tp = 0
    for pred in preds:
        best_iou = 0.0
        best_j   = -1
        for j, gt in enumerate(gts):
            if j in matched_gt:
                continue
            if pred.get("cls") != gt.get("cls"):
                continue
            iou = _iou_box(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_j   = j
        if best_iou >= iou_thresh:
            tp += 1
            matched_gt.add(best_j)
    fp = len(preds) - tp
    fn = len(gts)   - len(matched_gt)
    return tp, max(fp, 0), max(fn, 0)


# ── Ultralytics validation wrapper ────────────────────────────────────────────

def run_validation(
    weights:    Path,
    data_yaml:  Path,
    img_size:   int   = YOLO_IMG_SIZE,
    conf:       float = YOLO_CONF_THRESH,
    iou:        float = YOLO_IOU_THRESH,
    split:      str   = "val",
    out_dir:    Path  = OUTPUTS_DIR / "eval",
) -> dict:
    from ultralytics import YOLO

    out_dir.mkdir(parents=True, exist_ok=True)
    model   = YOLO(str(weights))
    metrics = model.val(
        data    = str(data_yaml),
        imgsz   = img_size,
        conf    = conf,
        iou     = iou,
        split   = split,
        project = str(out_dir),
        name    = "yolo_eval",
        exist_ok= True,
        plots   = True,
        verbose = True,
    )
    return metrics


def print_metrics_table(metrics, class_names: list[str] = XVIEW3_CLASS_NAMES) -> None:
    header = f"{'CLASS':<20}  {'P':>6}  {'R':>6}  {'F1':>6}  {'mAP50':>7}  {'mAP50-95':>9}"
    print("\n" + header)
    print("-" * len(header))
    for i, name in enumerate(class_names):
        try:
            p    = float(metrics.box.p[i])
            r    = float(metrics.box.r[i])
            f1   = 2 * p * r / (p + r + 1e-9)
            m50  = float(metrics.box.ap50[i])
            m5095 = float(metrics.box.ap[i])
            print(f"{name:<20}  {p:6.3f}  {r:6.3f}  {f1:6.3f}  {m50:7.3f}  {m5095:9.3f}")
        except (IndexError, AttributeError):
            print(f"{name:<20}  {'n/a':>6}")
    try:
        m50   = metrics.box.map50
        m5095 = metrics.box.map
        print(f"\n{'ALL':<20}  {'':>6}  {'':>6}  {'':>6}  {m50:7.3f}  {m5095:9.3f}")
    except AttributeError:
        pass


def plot_pr_curve(
    thresholds: np.ndarray,
    precisions: np.ndarray,
    recalls:    np.ndarray,
    out_path:   Path,
    class_name: str = "all",
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # PR curve
    ax = axes[0]
    ax.plot(recalls, precisions, lw=2, color="steelblue")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"PR Curve — {class_name}")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # F1 vs threshold
    ax = axes[1]
    f1 = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    ax.plot(thresholds, f1, lw=2, color="darkorange", label="F1")
    best_i = int(np.argmax(f1))
    ax.axvline(thresholds[best_i], color="gray", ls="--", alpha=0.6,
               label=f"best conf={thresholds[best_i]:.2f} F1={f1[best_i]:.3f}")
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("F1")
    ax.set_title("F1 vs Confidence")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK]   PR curve → {out_path}")


def fishing_fscore(
    metrics,
    fishing_class_idx: int = 2,
    beta: float = 0.5,
) -> float | None:
    """
    F-beta score for the fishing_vessel class (IUU primary target).

    beta < 1 weights precision more heavily — for IUU alerts we prefer
    fewer false alarms over missing real vessels (precision-biased).
    """
    try:
        p = float(metrics.box.p[fishing_class_idx])
        r = float(metrics.box.r[fishing_class_idx])
        fb = (1 + beta**2) * p * r / (beta**2 * p + r + 1e-9)
        print(f"\n[IUU]  fishing_vessel  F{beta}={fb:.4f}  "
              f"P={p:.4f}  R={r:.4f}")
        return fb
    except (IndexError, AttributeError):
        return None


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Evaluate trained YOLOv8 vessel detector.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--weights",    type=Path, required=True)
    p.add_argument("--data",       type=Path, required=True,
                   help="dataset.yaml")
    p.add_argument("--split",      default="val",
                   choices=["train", "val", "test"])
    p.add_argument("--img-size",   type=int,  default=YOLO_IMG_SIZE)
    p.add_argument("--conf",       type=float, default=YOLO_CONF_THRESH)
    p.add_argument("--iou",        type=float, default=YOLO_IOU_THRESH)
    p.add_argument("--out-dir",    type=Path,  default=OUTPUTS_DIR / "eval")
    p.add_argument("--beta",       type=float, default=0.5,
                   help="Beta for F-beta IUU score (< 1 = precision-biased)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    metrics = run_validation(
        args.weights, args.data,
        args.img_size, args.conf, args.iou,
        args.split, args.out_dir,
    )
    print_metrics_table(metrics)
    fishing_fscore(metrics, beta=args.beta)


if __name__ == "__main__":
    main()

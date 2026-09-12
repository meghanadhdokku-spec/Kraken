"""Visualisation helpers: render SAR chips and overlay bounding boxes."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def sar_to_rgb(data: np.ndarray, percentile: int = 98) -> np.ndarray:
    """
    Convert a (C, H, W) or (H, W) SAR array to uint8 RGB for display.
    Uses [VV, VH, VV] channel assignment.
    """
    def _norm(band: np.ndarray) -> np.ndarray:
        lo = np.nanpercentile(band, 100 - percentile)
        hi = np.nanpercentile(band, percentile)
        if hi == lo:
            return np.zeros_like(band, dtype=np.uint8)
        clipped = np.clip(band, lo, hi)
        return ((clipped - lo) / (hi - lo) * 255).astype(np.uint8)

    if data.ndim == 2:
        gray = _norm(data)
        return np.stack([gray, gray, gray], axis=-1)

    vv = _norm(data[0])
    vh = _norm(data[1]) if data.shape[0] > 1 else vv
    return np.stack([vv, vh, vv], axis=-1)


def plot_detections(
    data: np.ndarray,
    boxes: list[dict],
    title: str = "Detections",
    out_path: Path | None = None,
    box_color: str = "red",
) -> None:
    """
    Overlay bounding boxes on a SAR chip and display / save.

    boxes: list of dicts with x, y, w, h  (pixel coords).
    """
    rgb = sar_to_rgb(data)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(rgb)
    ax.set_title(title)
    ax.axis("off")

    for b in boxes:
        # support both CFAR (x,y,w,h) and YOLO (x1,y1,x2,y2)
        if "w" in b:
            rect = mpatches.Rectangle(
                (b["x"], b["y"]), b["w"], b["h"],
                linewidth=1, edgecolor=box_color, facecolor="none",
            )
        else:
            x1, y1 = b["x1"], b["y1"]
            w = b["x2"] - x1
            h = b["y2"] - y1
            rect = mpatches.Rectangle(
                (x1, y1), w, h,
                linewidth=1, edgecolor=box_color, facecolor="none",
            )
        ax.add_patch(rect)

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"[OK]   Figure → {out_path}")
    else:
        plt.show()
    plt.close(fig)

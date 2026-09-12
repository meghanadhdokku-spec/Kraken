"""Visualisation helpers: render SAR chips and overlay bounding boxes."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ── colour coding per detector ─────────────────────────────────────────────────
DETECTOR_COLORS = {"cfar": "red", "yolo": "cyan", "both": "yellow"}
CLASS_COLORS    = {0: "gray", 1: "lime", 2: "orange"}
CLASS_NAMES     = {0: "non_vessel", 1: "vessel", 2: "fishing_vessel"}


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


# ── scene overview ─────────────────────────────────────────────────────────────

def render_scene_overview(
    scene_path: Path,
    cfar_boxes: list[dict] | None = None,
    yolo_boxes: list[dict] | None = None,
    out_path: Path | None = None,
    downsample: int = 4,
) -> None:
    """
    Render a full-scene SAR overview with CFAR (red) and YOLO (cyan) boxes.

    Downsamples the scene by `downsample` factor for display.
    Saves to out_path if provided, otherwise shows interactively.
    """
    import rasterio

    with rasterio.open(scene_path) as src:
        out_shape = (src.count, src.height // downsample, src.width // downsample)
        data = src.read(out_shape=out_shape, resampling=rasterio.enums.Resampling.average)

    rgb = sar_to_rgb(data)
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.imshow(rgb)
    ax.set_title(f"Scene overview — {scene_path.stem}", fontsize=11)
    ax.axis("off")

    legend_handles = []

    def _add_boxes(boxes, color, label, fmt):
        if not boxes:
            return
        for b in boxes:
            if fmt == "xywh":
                x, y, w, h = b["x"] / downsample, b["y"] / downsample, b["w"] / downsample, b["h"] / downsample
            else:
                x = b["x1"] / downsample; y = b["y1"] / downsample
                w = (b["x2"] - b["x1"]) / downsample; h = (b["y2"] - b["y1"]) / downsample
            ax.add_patch(mpatches.Rectangle((x, y), w, h,
                         linewidth=0.8, edgecolor=color, facecolor="none"))
        legend_handles.append(mpatches.Patch(edgecolor=color, facecolor="none", label=label))

    if cfar_boxes:
        _add_boxes(cfar_boxes, "red",  f"CFAR ({len(cfar_boxes)})", "xywh")
    if yolo_boxes:
        _add_boxes(yolo_boxes, "cyan", f"YOLO ({len(yolo_boxes)})", "xyxy")

    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right", fontsize=9,
                  framealpha=0.7, facecolor="#111")

    plt.tight_layout()
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"[OK]   Overview → {out_path}")
    else:
        plt.show()
    plt.close(fig)


# ── detection chip grid ────────────────────────────────────────────────────────

def render_detection_grid(
    scene_path: Path,
    boxes: list[dict],
    out_path: Path | None = None,
    crop_px: int = 64,
    max_chips: int = 64,
    cols: int = 8,
) -> None:
    """
    Draw a grid of 64×64 pixel crops centred on each detection.

    Colour-codes box border by class (gray=non_vessel, lime=vessel, orange=fishing).
    """
    import rasterio

    with rasterio.open(scene_path) as src:
        data = src.read().astype(np.float32)

    boxes = boxes[:max_chips]
    rows  = (len(boxes) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = np.array(axes).reshape(-1)  # flatten for easy indexing

    half = crop_px // 2
    _, H, W = data.shape

    for i, b in enumerate(boxes):
        # centroid in pixel coords
        if "cx" in b:
            cx, cy = int(b["cx"]), int(b["cy"])
        elif "x1" in b:
            cx = int((b["x1"] + b["x2"]) / 2)
            cy = int((b["y1"] + b["y2"]) / 2)
        else:
            cx = int(b["x"] + b["w"] / 2)
            cy = int(b["y"] + b["h"] / 2)

        r0 = max(0, cy - half); r1 = min(H, cy + half)
        c0 = max(0, cx - half); c1 = min(W, cx + half)
        chip = data[:, r0:r1, c0:c1]
        rgb  = sar_to_rgb(chip)

        cls   = b.get("cls", 1)
        color = CLASS_COLORS.get(cls, "white")
        conf  = b.get("conf")
        label = CLASS_NAMES.get(cls, "vessel")
        if conf is not None:
            label = f"{label}\n{conf:.2f}"

        ax = axes[i]
        ax.imshow(rgb)
        ax.set_title(label, fontsize=6, color=color)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)
            spine.set_visible(True)

    for j in range(len(boxes), len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"Detection crops — {scene_path.stem}", fontsize=10)
    plt.tight_layout()

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"[OK]   Grid    → {out_path}")
    else:
        plt.show()
    plt.close(fig)

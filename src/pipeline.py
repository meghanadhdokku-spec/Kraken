"""
End-to-end Kraken pipeline: download → preprocess → detect → visualise.

Usage:
    # Full run (requires Copernicus credentials in .env):
    python -m src.pipeline --limit 2

    # Skip download (process already-downloaded scenes):
    python -m src.pipeline --skip-download --scene data/raw/S1A_....zip

    # Skip both download and preprocess (detect on existing GeoTIFF):
    python -m src.pipeline --skip-download --skip-preprocess \
        --scene data/processed/S1A_..._processed.tif

    # YOLO only (requires trained weights):
    python -m src.pipeline --skip-download --skip-preprocess \
        --scene data/processed/scene.tif \
        --detector yolo --weights models/yolo_vessel/weights/best.pt

    # Both detectors + visualisation:
    python -m src.pipeline --skip-download --skip-preprocess \
        --scene data/processed/scene.tif \
        --detector both --weights models/yolo_vessel/weights/best.pt \
        --viz
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import (
    RAW_DIR,
    PROCESSED_DIR,
    OUTPUTS_DIR,
    GOG_BBOX,
    GOG_WKT,
    COPERNICUS_USER,
    COPERNICUS_PASSWORD,
    CFAR_GUARD_CELLS,
    CFAR_BACKGROUND_CELLS,
    CFAR_FALSE_ALARM_RATE,
    YOLO_CONF_THRESH,
    YOLO_IOU_THRESH,
    YOLO_IMG_SIZE,
)
from src.download.sentinel_download import (
    get_access_token,
    query_scenes,
    download_scenes,
    _default_dates,
    _bbox_to_wkt,
)
from src.preprocess.sar_preprocess import preprocess_scene
from src.detect.cfar_detect import detect_scene as cfar_detect_scene
from src.utils.geo_utils import merge_geojson, detections_to_geojson, save_geojson


def run_pipeline(
    limit: int            = 3,
    start: str | None     = None,
    end: str | None       = None,
    bbox: tuple | None    = None,
    skip_download: bool   = False,
    skip_preprocess: bool = False,
    scene: Path | None    = None,
    out_dir: Path         = OUTPUTS_DIR,
    speckle_kernel: int   = 7,
    cfar_guard: int       = CFAR_GUARD_CELLS,
    cfar_bg: int          = CFAR_BACKGROUND_CELLS,
    pfa: float            = CFAR_FALSE_ALARM_RATE,
    detector: str         = "cfar",
    weights: Path | None  = None,
    yolo_conf: float      = YOLO_CONF_THRESH,
    yolo_iou: float       = YOLO_IOU_THRESH,
    yolo_tile: int        = YOLO_IMG_SIZE,
    viz: bool             = False,
) -> list[dict]:
    """
    Orchestrate the full pipeline and return all detections across scenes.
    detector: 'cfar' | 'yolo' | 'both'
    """
    all_detections = []

    # ── 1. Download ────────────────────────────────────────────────────────────
    if skip_download:
        if scene is None:
            raw_scenes = sorted(RAW_DIR.glob("*.zip")) + sorted(RAW_DIR.glob("*.SAFE"))
        else:
            raw_scenes = [scene] if scene.suffix.lower() in (".zip", "") else []
    else:
        if not COPERNICUS_USER or not COPERNICUS_PASSWORD:
            sys.exit(
                "[ERROR] No Copernicus credentials.\n"
                "  Set COPERNICUS_USER / COPERNICUS_PASSWORD in .env "
                "or use --skip-download."
            )
        default_start, default_end = _default_dates()
        date_start = start or default_start
        date_end   = end   or default_end
        footprint  = _bbox_to_wkt(*bbox) if bbox else GOG_WKT
        token      = get_access_token(COPERNICUS_USER, COPERNICUS_PASSWORD)
        products   = query_scenes(footprint, date_start, date_end, limit)
        raw_scenes = download_scenes(products, RAW_DIR, token)

    if not raw_scenes and not skip_preprocess:
        print("[WARN] No scenes to process.")
        return []

    # ── 2. Preprocess ──────────────────────────────────────────────────────────
    if skip_preprocess:
        if scene is not None:
            processed_scenes = [scene]
        else:
            processed_scenes = sorted(PROCESSED_DIR.glob("*_processed.tif"))
    else:
        processed_scenes = []
        for raw in raw_scenes:
            try:
                tif = preprocess_scene(raw, PROCESSED_DIR, GOG_BBOX, speckle_kernel)
                processed_scenes.append(tif)
            except Exception as exc:
                print(f"[WARN] Preprocess failed for {raw.name}: {exc}")

    if not processed_scenes:
        print("[WARN] No processed scenes available.")
        return []

    # ── 3. Detect ──────────────────────────────────────────────────────────────
    for tif in processed_scenes:
        stem = tif.stem
        cfar_boxes: list[dict] = []
        yolo_boxes: list[dict] = []

        if detector in ("cfar", "both"):
            try:
                cfar_boxes = cfar_detect_scene(
                    tif,
                    guard=cfar_guard,
                    background=cfar_bg,
                    pfa=pfa,
                    out_dir=out_dir,
                )
                all_detections.extend(cfar_boxes)
            except Exception as exc:
                print(f"[WARN] CFAR failed for {tif.name}: {exc}")

        if detector in ("yolo", "both"):
            if weights is None:
                print("[WARN] --weights required for YOLO detector; skipping.")
            else:
                try:
                    from src.detect.yolo_detect import run_yolo_on_scene
                    yolo_boxes = run_yolo_on_scene(
                        tif, weights,
                        tile_size=yolo_tile,
                        conf=yolo_conf,
                        iou=yolo_iou,
                        out_dir=out_dir,
                    )
                    all_detections.extend(yolo_boxes)
                except Exception as exc:
                    print(f"[WARN] YOLO failed for {tif.name}: {exc}")

        # ── combined GeoJSON when both detectors ran ───────────────────────────
        if detector == "both" and (cfar_boxes or yolo_boxes):
            cfar_fc = detections_to_geojson(cfar_boxes, stem, "cfar")
            yolo_fc = detections_to_geojson(yolo_boxes, stem, "yolo")
            combined = merge_geojson(cfar_fc, yolo_fc)
            save_geojson(combined, out_dir / f"{stem}_detections.geojson")

        # ── visualisation ──────────────────────────────────────────────────────
        if viz:
            try:
                from src.utils.viz import render_scene_overview, render_detection_grid
                render_scene_overview(
                    tif,
                    cfar_boxes=cfar_boxes or None,
                    yolo_boxes=yolo_boxes or None,
                    out_path=out_dir / f"{stem}_overview.png",
                )
                all_boxes = cfar_boxes + yolo_boxes
                if all_boxes:
                    render_detection_grid(
                        tif,
                        all_boxes,
                        out_path=out_dir / f"{stem}_chips.png",
                    )
            except Exception as exc:
                print(f"[WARN] Visualisation failed for {tif.name}: {exc}")

    print(f"\n[PIPELINE DONE] Total vessel candidates: {len(all_detections)}")
    return all_detections


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="End-to-end Kraken vessel detection pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--limit",            type=int,   default=3)
    p.add_argument("--start",            type=str,   default=None)
    p.add_argument("--end",              type=str,   default=None)
    p.add_argument("--bbox", nargs=4,    type=float, metavar=("W", "S", "E", "N"))
    p.add_argument("--skip-download",    action="store_true")
    p.add_argument("--skip-preprocess",  action="store_true")
    p.add_argument("--scene",            type=Path,  default=None,
                   help="Single scene to process (SAFE zip or processed .tif)")
    p.add_argument("--out-dir",          type=Path,  default=OUTPUTS_DIR)
    p.add_argument("--kernel",           type=int,   default=7)
    p.add_argument("--guard",            type=int,   default=CFAR_GUARD_CELLS)
    p.add_argument("--background",       type=int,   default=CFAR_BACKGROUND_CELLS)
    p.add_argument("--pfa",              type=float, default=CFAR_FALSE_ALARM_RATE)
    p.add_argument("--detector",         default="cfar",
                   choices=["cfar", "yolo", "both"],
                   help="Which detector(s) to run")
    p.add_argument("--weights",          type=Path,  default=None,
                   help="YOLO weights .pt file (required for yolo/both)")
    p.add_argument("--yolo-conf",        type=float, default=YOLO_CONF_THRESH)
    p.add_argument("--yolo-iou",         type=float, default=YOLO_IOU_THRESH)
    p.add_argument("--yolo-tile",        type=int,   default=YOLO_IMG_SIZE)
    p.add_argument("--viz",              action="store_true",
                   help="Save scene overview and detection chip grid PNGs")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    run_pipeline(
        limit           = args.limit,
        start           = args.start,
        end             = args.end,
        bbox            = tuple(args.bbox) if args.bbox else None,
        skip_download   = args.skip_download,
        skip_preprocess = args.skip_preprocess,
        scene           = args.scene,
        out_dir         = args.out_dir,
        speckle_kernel  = args.kernel,
        cfar_guard      = args.guard,
        cfar_bg         = args.background,
        pfa             = args.pfa,
        detector        = args.detector,
        weights         = args.weights,
        yolo_conf       = args.yolo_conf,
        yolo_iou        = args.yolo_iou,
        yolo_tile       = args.yolo_tile,
        viz             = args.viz,
    )


if __name__ == "__main__":
    main()

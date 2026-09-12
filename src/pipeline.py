"""
End-to-end Kraken pipeline: download → preprocess → CFAR detect.

Usage:
    # Full run (requires Copernicus credentials in .env):
    python -m src.pipeline --limit 2

    # Skip download (process already-downloaded scenes):
    python -m src.pipeline --skip-download --scene data/raw/S1A_....zip

    # Skip both download and preprocess (detect on existing GeoTIFF):
    python -m src.pipeline --skip-download --skip-preprocess \
        --scene data/processed/S1A_..._processed.tif
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
    CFAR_GUARD_CELLS,
    CFAR_BACKGROUND_CELLS,
    CFAR_FALSE_ALARM_RATE,
)
from src.download.sentinel_download import (
    build_api,
    query_scenes,
    download_scenes,
    _default_dates,
    _bbox_to_wkt,
    COPERNICUS_USER,
    COPERNICUS_PASSWORD,
    SENTINEL_API_URL,
)
from src.preprocess.sar_preprocess import preprocess_scene
from src.detect.cfar_detect import detect_scene


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
) -> list[dict]:
    """
    Orchestrate the full pipeline and return all detections across scenes.
    """
    all_detections = []

    # ── 1. Download ────────────────────────────────────────────────────────────
    if skip_download:
        if scene is None:
            raw_scenes = sorted(RAW_DIR.glob("*.zip")) + sorted(RAW_DIR.glob("*.SAFE"))
        else:
            raw_scenes = [scene] if scene.suffix.lower() in (".zip", "") else []
            if not raw_scenes:
                # treat as already-preprocessed
                raw_scenes = []
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
        footprint  = _bbox_to_wkt(*bbox) if bbox else (
            f"POLYGON(("
            f"{GOG_BBOX[0]} {GOG_BBOX[1]}, "
            f"{GOG_BBOX[2]} {GOG_BBOX[1]}, "
            f"{GOG_BBOX[2]} {GOG_BBOX[3]}, "
            f"{GOG_BBOX[0]} {GOG_BBOX[3]}, "
            f"{GOG_BBOX[0]} {GOG_BBOX[1]}"
            f"))"
        )
        api      = build_api(COPERNICUS_USER, COPERNICUS_PASSWORD, SENTINEL_API_URL)
        products = query_scenes(api, footprint, date_start, date_end, limit)
        raw_scenes = download_scenes(api, products, RAW_DIR)

    if not raw_scenes and not skip_preprocess:
        print("[WARN] No scenes to process.")
        return []

    # ── 2. Preprocess ──────────────────────────────────────────────────────────
    if skip_preprocess:
        # scene is assumed to already be a processed GeoTIFF
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
        try:
            boxes = detect_scene(
                tif,
                guard=cfar_guard,
                background=cfar_bg,
                pfa=pfa,
                out_dir=out_dir,
            )
            all_detections.extend(boxes)
        except Exception as exc:
            print(f"[WARN] CFAR failed for {tif.name}: {exc}")

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
    )


if __name__ == "__main__":
    main()

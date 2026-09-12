"""
End-to-end smoke test for the Kraken pipeline — no credentials required.

Steps:
  1. Generate a synthetic SAR scene (10 planted vessel targets)
  2. Run CFAR detection on both VV and VH bands
  3. Run AIS matching against a synthetic AIS CSV (5 vessels at known positions)
  4. Print a summary table and save outputs to data/outputs/smoke_test/

Usage:
    python scripts/smoke_test.py [--no-land-mask] [--pfa FLOAT]

Expected result:
  10 vessel candidates detected, ~5 matched to AIS, ~5 flagged dark.
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.make_test_scene import make_synthetic_scene, VESSEL_POSITIONS
from src.detect.cfar_detect import detect_scene
from src.match.ais_match import run_ais_matching

_OUT_DIR   = Path("data/outputs/smoke_test")
_SCENE_PATH = _OUT_DIR / "synthetic_scene_processed.tif"
_AIS_CSV    = _OUT_DIR / "synthetic_ais.csv"


def _write_ais_csv(path: Path, scene_time: str = "2023-04-12T06:00:00") -> None:
    """Write a synthetic AIS CSV — first 5 vessel positions, others absent."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["lat", "lon", "mmsi", "vessel_name", "flag", "timestamp"]
        )
        writer.writeheader()
        for i, (lon, lat) in enumerate(VESSEL_POSITIONS[:5]):
            writer.writerow({
                "lat": lat, "lon": lon,
                "mmsi":        f"63600{i:04d}",
                "vessel_name": f"MV Synthetic-{i+1}",
                "flag":        "GH",
                "timestamp":   scene_time,
            })
    print(f"[OK]  AIS CSV written → {path}  ({5} vessels)")


def _print_summary(detections: list[dict]) -> None:
    n      = len(detections)
    dark   = sum(1 for d in detections if d.get("dark_vessel"))
    matched = n - dark
    print()
    print("=" * 58)
    print(f"  SMOKE TEST SUMMARY")
    print("=" * 58)
    print(f"  Detections:        {n}")
    print(f"  AIS-matched:       {matched}")
    print(f"  Dark (no AIS):     {dark}")
    print("=" * 58)
    print()
    if n >= 8:
        print("  PASS — pipeline detected expected vessel targets")
    else:
        print("  WARN — fewer detections than expected (check PFA setting)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-land-mask", action="store_true",
                    help="Skip land mask (faster, less accurate)")
    ap.add_argument("--pfa", type=float, default=1e-5,
                    help="CFAR false-alarm rate (default: 1e-5)")
    args = ap.parse_args()

    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Generate synthetic scene ────────────────────────────────────────────
    print("\n[STEP 1] Generating synthetic SAR scene...")
    make_synthetic_scene(_SCENE_PATH, size=512)

    # ── 2. CFAR detection ──────────────────────────────────────────────────────
    print("\n[STEP 2] Running CFAR detector...")
    detections = detect_scene(
        _SCENE_PATH,
        pfa=args.pfa,
        use_land_mask=not args.no_land_mask,
        out_dir=_OUT_DIR,
        dual_band=True,
    )

    if not detections:
        print("[FAIL] No detections produced — check scene generation or PFA.")
        sys.exit(1)

    # ── 3. AIS matching ────────────────────────────────────────────────────────
    print("\n[STEP 3] Running AIS matching...")
    _write_ais_csv(_AIS_CSV)
    detections = run_ais_matching(
        detections,
        scene_path=Path("S1A_IW_GRDH_1SDV_20230412T060000_20230412T060030_fake"),
        radius_km=2.0,
        ais_csv_path=_AIS_CSV,
    )

    # ── 4. Summary ─────────────────────────────────────────────────────────────
    _print_summary(detections)
    print(f"  Outputs saved to:  {_OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()

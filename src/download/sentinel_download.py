"""
Sentinel-1 GRD download script — Gulf of Guinea vessel detection.

Usage:
    python -m src.download.sentinel_download [OPTIONS]

Options:
    --start DATE        Start date (YYYY-MM-DD).  Default: 30 days ago.
    --end   DATE        End date   (YYYY-MM-DD).  Default: today.
    --bbox  W S E N     Custom bounding box.      Default: Gulf of Guinea.
    --limit N           Max scenes to download.   Default: 5.
    --query-only        Print product list; skip download.
    --out-dir PATH      Override download directory.
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

from sentinelsat import SentinelAPI, geojson_to_wkt, read_geojson
from sentinelsat.exceptions import UnauthorizedError, ServerError

# allow `python -m src.download.sentinel_download` from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import (
    COPERNICUS_USER,
    COPERNICUS_PASSWORD,
    SENTINEL_API_URL,
    GOG_WKT,
    RAW_DIR,
    S1_PRODUCT_TYPE,
    S1_SENSOR_MODE,
    S1_POLARISATION,
)


# ── helpers ────────────────────────────────────────────────────────────────────

def _bbox_to_wkt(west: float, south: float, east: float, north: float) -> str:
    return (
        f"POLYGON(("
        f"{west} {south}, "
        f"{east} {south}, "
        f"{east} {north}, "
        f"{west} {north}, "
        f"{west} {south}"
        f"))"
    )


def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def _default_dates() -> tuple[str, str]:
    end   = datetime.utcnow()
    start = end - timedelta(days=30)
    return start.strftime("%Y%m%d"), end.strftime("%Y%m%d")


# ── core ───────────────────────────────────────────────────────────────────────

def build_api(user: str, password: str, url: str) -> SentinelAPI:
    """Connect to Copernicus Dataspace and return an authenticated API handle."""
    try:
        api = SentinelAPI(user, password, url)
        return api
    except UnauthorizedError:
        sys.exit(
            "[ERROR] Copernicus credentials rejected. "
            "Check COPERNICUS_USER / COPERNICUS_PASSWORD in your .env file."
        )


def query_scenes(
    api: SentinelAPI,
    footprint_wkt: str,
    date_start: str,
    date_end: str,
    limit: int,
) -> dict:
    """
    Query Copernicus for Sentinel-1 GRD/IW products that intersect the footprint.

    Returns an OrderedDict of {uuid: product_info}.
    """
    print(f"[INFO] Querying scenes from {date_start} to {date_end} …")

    products = api.query(
        footprint_wkt,
        date=(date_start, date_end),
        platformname="Sentinel-1",
        producttype=S1_PRODUCT_TYPE,
        sensoroperationalmode=S1_SENSOR_MODE,
        polarisationmode=S1_POLARISATION,
        limit=limit,
    )

    print(f"[INFO] Found {len(products)} scene(s).")
    return products


def print_scene_table(api: SentinelAPI, products: dict) -> None:
    """Pretty-print a table of found scenes."""
    if not products:
        print("[INFO] No scenes to display.")
        return

    gdf = api.to_geodataframe(products)
    cols = ["title", "beginposition", "size", "uuid"]
    available_cols = [c for c in cols if c in gdf.columns]
    print("\n" + gdf[available_cols].to_string(index=False) + "\n")


def download_scenes(
    api: SentinelAPI,
    products: dict,
    out_dir: Path,
) -> list[Path]:
    """
    Download all products to *out_dir*.

    Already-downloaded files are skipped automatically (sentinelsat checksums).
    Returns list of downloaded file paths.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if not products:
        print("[INFO] Nothing to download.")
        return []

    print(f"[INFO] Downloading {len(products)} scene(s) → {out_dir}")

    try:
        results = api.download_all(products, directory_path=out_dir)
    except ServerError as exc:
        sys.exit(f"[ERROR] Copernicus server error: {exc}")

    downloaded = []
    for uuid, info in results.downloaded.items():
        path = Path(info["path"])
        print(f"[OK]   {path.name}")
        downloaded.append(path)

    for uuid in results.failed:
        print(f"[WARN] Failed to download {uuid}")

    for uuid in results.skipped:
        print(f"[SKIP] Already present: {uuid}")

    return downloaded


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Sentinel-1 GRD scenes for vessel detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    default_start, default_end = _default_dates()

    parser.add_argument("--start", default=default_start, help="Start date YYYYMMDD or YYYY-MM-DD")
    parser.add_argument("--end",   default=default_end,   help="End date   YYYYMMDD or YYYY-MM-DD")
    parser.add_argument(
        "--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"),
        help="Custom AOI bounding box (overrides Gulf of Guinea default)",
    )
    parser.add_argument("--limit",      type=int,  default=5,     help="Max scenes to fetch")
    parser.add_argument("--query-only", action="store_true",      help="List scenes without downloading")
    parser.add_argument("--out-dir",    type=Path, default=RAW_DIR, help="Download destination directory")
    parser.add_argument("--user",       default=COPERNICUS_USER,     help="Copernicus username")
    parser.add_argument("--password",   default=COPERNICUS_PASSWORD, help="Copernicus password")
    parser.add_argument("--api-url",    default=SENTINEL_API_URL,    help="Copernicus API endpoint")

    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    if not args.user or not args.password:
        sys.exit(
            "[ERROR] No Copernicus credentials found.\n"
            "  Set COPERNICUS_USER and COPERNICUS_PASSWORD in a .env file,\n"
            "  or pass --user / --password on the command line."
        )

    # normalise date strings to YYYYMMDD (sentinelsat wants that format)
    def _norm(d: str) -> str:
        if "-" in d:
            return datetime.strptime(d, "%Y-%m-%d").strftime("%Y%m%d")
        return d

    date_start = _norm(args.start)
    date_end   = _norm(args.end)

    footprint = _bbox_to_wkt(*args.bbox) if args.bbox else GOG_WKT

    api = build_api(args.user, args.password, args.api_url)

    products = query_scenes(api, footprint, date_start, date_end, args.limit)
    print_scene_table(api, products)

    if args.query_only:
        print("[INFO] --query-only set; skipping download.")
        return

    downloaded = download_scenes(api, products, args.out_dir)
    print(f"\n[DONE] {len(downloaded)} file(s) in {args.out_dir}")


if __name__ == "__main__":
    main()

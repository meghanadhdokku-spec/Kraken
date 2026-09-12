"""
Sentinel-1 GRD download script — Gulf of Guinea vessel detection.

Uses Copernicus Dataspace OData API with OAuth2 (replaces legacy SciHub/sentinelsat).

Usage:
    python -m src.download.sentinel_download [OPTIONS]

Options:
    --start DATE        Start date (YYYY-MM-DD or YYYYMMDD). Default: 30 days ago.
    --end   DATE        End date.                            Default: today.
    --bbox  W S E N     Custom bounding box.                 Default: Gulf of Guinea.
    --limit N           Max scenes to list/download.         Default: 5.
    --query-only        Print product list; skip download.
    --out-dir PATH      Override download directory.
    --user EMAIL        Copernicus username (or set COPERNICUS_USER in .env).
    --password PASS     Copernicus password (or set COPERNICUS_PASSWORD in .env).
"""

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import (
    COPERNICUS_USER,
    COPERNICUS_PASSWORD,
    GOG_WKT,
    RAW_DIR,
    S1_PRODUCT_TYPE,
    S1_SENSOR_MODE,
)

# ── Copernicus Dataspace endpoints ─────────────────────────────────────────────
_TOKEN_URL  = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
_SEARCH_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
_ZIPPER_URL = "https://zipper.dataspace.copernicus.eu/odata/v1/Products({uuid})/$value"


# ── helpers ────────────────────────────────────────────────────────────────────

def _bbox_to_wkt(west: float, south: float, east: float, north: float) -> str:
    return (
        f"POLYGON(("
        f"{west} {south},{east} {south},"
        f"{east} {north},{west} {north},"
        f"{west} {south}"
        f"))"
    )


def _parse_date(s: str) -> datetime:
    fmt = "%Y-%m-%d" if "-" in s else "%Y%m%d"
    return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)


def _default_dates() -> tuple[str, str]:
    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=30)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


# ── auth ───────────────────────────────────────────────────────────────────────

def get_access_token(user: str, password: str) -> str:
    """Obtain OAuth2 access token from Copernicus Identity Service."""
    resp = requests.post(_TOKEN_URL, data={
        "grant_type":    "password",
        "client_id":     "cdse-public",
        "username":      user,
        "password":      password,
    }, timeout=30)
    if resp.status_code == 401:
        sys.exit("[ERROR] Copernicus credentials rejected. Check user/password.")
    if not resp.ok:
        sys.exit(f"[ERROR] Token request failed {resp.status_code}: {resp.text[:200]}")
    return resp.json()["access_token"]


# ── search ─────────────────────────────────────────────────────────────────────

def query_scenes(
    footprint_wkt: str,
    date_start: str,
    date_end: str,
    limit: int,
) -> list[dict]:
    """
    Query Copernicus Dataspace OData for Sentinel-1 GRD/IW scenes.

    Returns list of product dicts with id, name, size, date.
    """
    start_dt = _parse_date(date_start).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    end_dt   = _parse_date(date_end).strftime("%Y-%m-%dT%H:%M:%S.000Z")

    filter_parts = [
        "Collection/Name eq 'SENTINEL-1'",
        f"Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType'"
        f" and att/OData.CSC.StringAttribute/Value eq '{S1_PRODUCT_TYPE}')",
        f"Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'operationalMode'"
        f" and att/OData.CSC.StringAttribute/Value eq '{S1_SENSOR_MODE}')",
        f"ContentDate/Start gt {start_dt}",
        f"ContentDate/Start lt {end_dt}",
        f"OData.CSC.Intersects(area=geography'SRID=4326;{footprint_wkt}')",
    ]

    params = {
        "$filter": " and ".join(filter_parts),
        "$orderby": "ContentDate/Start desc",
        "$top":     limit,
        "$expand":  "Attributes",
    }

    print(f"[INFO] Querying scenes {date_start} → {date_end} …")
    resp = requests.get(_SEARCH_URL, params=params, timeout=60)
    if not resp.ok:
        sys.exit(f"[ERROR] Search failed {resp.status_code}: {resp.text[:300]}")

    products = resp.json().get("value", [])
    print(f"[INFO] Found {len(products)} scene(s).")
    return products


# ── display ────────────────────────────────────────────────────────────────────

def print_scene_table(products: list[dict]) -> None:
    if not products:
        print("[INFO] No scenes found.")
        return
    header = f"{'Title':<70}  {'Date':<22}  {'Size':>10}  ID"
    print("\n" + header)
    print("-" * len(header))
    for p in products:
        title = p.get("Name", "?")[:68]
        date  = p.get("ContentDate", {}).get("Start", "?")[:19]
        size  = p.get("ContentLength", 0)
        size_mb = f"{size / 1e6:.0f} MB" if size else "?"
        uid   = p.get("Id", "?")
        print(f"{title:<70}  {date:<22}  {size_mb:>10}  {uid}")
    print()


# ── download ───────────────────────────────────────────────────────────────────

def download_scenes(
    products: list[dict],
    out_dir: Path,
    token: str,
) -> list[Path]:
    """Download products to out_dir using the Copernicus zipper service."""
    out_dir.mkdir(parents=True, exist_ok=True)
    downloaded = []

    headers = {"Authorization": f"Bearer {token}"}

    for p in products:
        uid  = p["Id"]
        name = p.get("Name", uid)
        dest = out_dir / f"{name}.zip"

        if dest.exists():
            print(f"[SKIP] Already present: {dest.name}")
            downloaded.append(dest)
            continue

        url = _ZIPPER_URL.format(uuid=uid)
        print(f"[INFO] Downloading {name} …")
        try:
            with requests.get(url, headers=headers, stream=True, timeout=600) as r:
                if r.status_code == 401:
                    print(f"[WARN] Auth expired for {uid} — skipping.")
                    continue
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                wrote = 0
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
                        wrote += len(chunk)
                        if total:
                            pct = wrote / total * 100
                            print(f"\r  {pct:5.1f}%", end="", flush=True)
            print(f"\r[OK]   {dest.name}")
            downloaded.append(dest)
        except Exception as exc:
            print(f"[WARN] Failed {name}: {exc}")

    return downloaded


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Sentinel-1 GRD scenes (Copernicus Dataspace OAuth2).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    default_start, default_end = _default_dates()

    parser.add_argument("--start",      default=default_start,      help="Start date YYYYMMDD or YYYY-MM-DD")
    parser.add_argument("--end",        default=default_end,        help="End date")
    parser.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"),
                        help="Custom AOI bounding box")
    parser.add_argument("--limit",      type=int,  default=5,       help="Max scenes")
    parser.add_argument("--query-only", action="store_true",        help="List only, no download")
    parser.add_argument("--out-dir",    type=Path, default=RAW_DIR, help="Download directory")
    parser.add_argument("--user",       default=COPERNICUS_USER,     help="Copernicus email")
    parser.add_argument("--password",   default=COPERNICUS_PASSWORD, help="Copernicus password")

    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    if not args.user or not args.password:
        sys.exit(
            "[ERROR] No Copernicus credentials found.\n"
            "  Set COPERNICUS_USER and COPERNICUS_PASSWORD in a .env file,\n"
            "  or pass --user / --password on the command line."
        )

    footprint = _bbox_to_wkt(*args.bbox) if args.bbox else GOG_WKT

    products = query_scenes(footprint, args.start, args.end, args.limit)
    print_scene_table(products)

    if args.query_only:
        print("[INFO] --query-only; skipping download.")
        return

    token = get_access_token(args.user, args.password)
    downloaded = download_scenes(products, args.out_dir, token)
    print(f"\n[DONE] {len(downloaded)} file(s) in {args.out_dir}")


if __name__ == "__main__":
    main()

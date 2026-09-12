"""
AIS dark vessel matching for CFAR detections.

For each SAR detection, query Global Fishing Watch AIS API for vessels
within a search radius at the scene acquisition time. Detections with no
AIS match within the radius are flagged as 'dark' — a key IUU indicator.

Requires GFW_API_TOKEN in .env (free at https://globalfishingwatch.org/our-apis/)
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
from datetime import datetime, timezone, timedelta
from math import asin, cos, radians, sin, sqrt
from pathlib import Path

log = logging.getLogger(__name__)


# ── Haversine distance ────────────────────────────────────────────────────────

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two (lat, lon) points."""
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * R * asin(sqrt(max(0.0, min(1.0, a))))


# ── AIS fetch ─────────────────────────────────────────────────────────────────

def fetch_ais_vessels(
    bbox: tuple[float, float, float, float],  # west, south, east, north
    start_time: str,                           # ISO 8601
    end_time: str,                             # ISO 8601
    api_token: str | None = None,
    ais_csv_path: Path | None = None,
) -> list[dict]:
    """
    Return AIS vessel positions within *bbox* and the given time window.

    Parameters
    ----------
    bbox          : (west, south, east, north) in decimal degrees
    start_time    : ISO 8601 string, e.g. "2023-04-12T06:00:00Z"
    end_time      : ISO 8601 string
    api_token     : Global Fishing Watch API bearer token.  If None and
                    ``ais_csv_path`` is also None the function logs a warning
                    and returns an empty list.
    ais_csv_path  : Path to a local AIS CSV file (columns: lat, lon, mmsi,
                    vessel_name, flag, timestamp).  When provided, the API is
                    skipped and the CSV is filtered instead.

    Returns
    -------
    List of dicts with keys: lat, lon, vessel_id, vessel_name, flag.
    """
    if ais_csv_path is not None:
        return _fetch_from_csv(bbox, start_time, end_time, ais_csv_path)

    if not api_token:
        log.warning(
            "No GFW API token provided and no AIS CSV fallback — "
            "returning empty vessel list."
        )
        return []

    return _fetch_from_gfw(bbox, start_time, end_time, api_token)


def _fetch_from_gfw(
    bbox: tuple[float, float, float, float],
    start_time: str,
    end_time: str,
    api_token: str,
) -> list[dict]:
    """Query the Global Fishing Watch v3 API for all vessel types."""
    try:
        import requests  # type: ignore
    except ImportError:
        log.warning("'requests' package not installed — cannot query GFW API.")
        return []

    west, south, east, north = bbox
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json",
    }

    vessels: list[dict] = []

    # Query both fishing events and all-vessel presence to cover every type
    # (fishing vessels, cargo, tankers, passenger, etc.)
    _datasets = [
        ("public-global-fishing-events:latest",  "events"),
        ("public-global-presence:latest",         "presence"),
    ]

    for dataset, kind in _datasets:
        url = "https://gateway.api.globalfishingwatch.org/v3/events"
        params = {
            "datasets[0]": dataset,
            "start-date": start_time,
            "end-date": end_time,
            "bbox": f"{west},{south},{east},{north}",
            "limit": 500,
            "offset": 0,
        }
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=15)
            if resp.status_code == 422:
                # Dataset not supported on this endpoint — skip silently
                log.debug("GFW dataset %s not available on events endpoint.", dataset)
                continue
            resp.raise_for_status()
            data = resp.json()
            entries = data if isinstance(data, list) else data.get("entries", [])

            seen = {v["vessel_id"] for v in vessels}
            for entry in entries:
                pos = entry.get("position", {})
                lat = pos.get("lat") or entry.get("lat")
                lon = pos.get("lon") or entry.get("lon")
                if lat is None or lon is None:
                    continue
                vessel_info = entry.get("vessel", {})
                vid = vessel_info.get("id") or entry.get("vessel_id", "")
                if vid and vid in seen:
                    continue  # deduplicate across dataset queries
                seen.add(vid)
                vessels.append({
                    "lat": float(lat),
                    "lon": float(lon),
                    "vessel_id": vid,
                    "vessel_name": vessel_info.get("name") or entry.get("vessel_name", ""),
                    "flag": vessel_info.get("flag") or entry.get("flag", ""),
                    "timestamp": entry.get("start", entry.get("timestamp", "")),
                })
            log.info("GFW %s dataset returned %d entries.", kind, len(entries))

        except Exception as exc:
            log.warning("GFW %s query failed (%s) — skipping.", kind, exc)

    log.info("GFW API total unique vessels: %d", len(vessels))
    return vessels


def _fetch_from_csv(
    bbox: tuple[float, float, float, float],
    start_time: str,
    end_time: str,
    csv_path: Path,
) -> list[dict]:
    """Load AIS positions from a local CSV and filter by bbox + time window."""
    west, south, east, north = bbox

    t_start = _parse_iso(start_time)
    t_end = _parse_iso(end_time)

    vessels: list[dict] = []
    try:
        with open(csv_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    lat = float(row["lat"])
                    lon = float(row["lon"])
                except (KeyError, ValueError):
                    continue

                # Spatial filter
                if not (south <= lat <= north and west <= lon <= east):
                    continue

                # Temporal filter (best-effort; skip if timestamp absent)
                ts_str = row.get("timestamp", "")
                if ts_str:
                    try:
                        ts = _parse_iso(ts_str)
                        if ts < t_start or ts > t_end:
                            continue
                    except ValueError:
                        pass  # unparseable timestamp — include anyway

                vessels.append({
                    "lat": lat,
                    "lon": lon,
                    "vessel_id": row.get("mmsi", row.get("vessel_id", "")),
                    "vessel_name": row.get("vessel_name", ""),
                    "flag": row.get("flag", ""),
                    "timestamp": ts_str,
                })

    except FileNotFoundError:
        log.warning("AIS CSV not found: %s — returning empty vessel list.", csv_path)
    except Exception as exc:
        log.warning("Error reading AIS CSV (%s) — returning empty vessel list.", exc)

    log.info("Local AIS CSV supplied %d vessels after bbox/time filter.", len(vessels))
    return vessels


def _parse_iso(ts: str) -> datetime:
    """Parse an ISO 8601 timestamp to a timezone-aware datetime (UTC)."""
    ts = ts.rstrip("Z")
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(ts, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"Cannot parse timestamp: {ts!r}")


# ── Detection ↔ AIS matching ──────────────────────────────────────────────────

def match_detections_to_ais(
    detections: list[dict],
    ais_vessels: list[dict],
    radius_km: float = 1.0,
) -> list[dict]:
    """
    Annotate each detection with the nearest AIS vessel (if within *radius_km*).

    Modifies each detection dict in-place and returns the list.

    Added keys
    ----------
    ais_match         : bool — True if an AIS vessel was found within radius
    dark_vessel       : bool — True when ais_match is False
    matched_vessel_id : str | None
    matched_vessel_name : str | None
    matched_flag      : str | None
    ais_distance_km   : float | None — distance to nearest matched vessel
    """
    for det in detections:
        det_lat = float(det.get("lat", det.get("latitude", 0.0)))
        det_lon = float(det.get("lon", det.get("longitude", 0.0)))

        best_dist = float("inf")
        best_vessel: dict | None = None

        for vessel in ais_vessels:
            try:
                v_lat = float(vessel["lat"])
                v_lon = float(vessel["lon"])
            except (KeyError, ValueError, TypeError):
                continue

            dist = _haversine_km(det_lat, det_lon, v_lat, v_lon)
            if dist < best_dist:
                best_dist = dist
                best_vessel = vessel

        if best_vessel is not None and best_dist <= radius_km:
            det["ais_match"] = True
            det["dark_vessel"] = False
            det["matched_vessel_id"] = best_vessel.get("vessel_id", "")
            det["matched_vessel_name"] = best_vessel.get("vessel_name", "")
            det["matched_flag"] = best_vessel.get("flag", "")
            det["ais_distance_km"] = round(best_dist, 4)
        else:
            det["ais_match"] = False
            det["dark_vessel"] = True
            det["matched_vessel_id"] = None
            det["matched_vessel_name"] = None
            det["matched_flag"] = None
            det["ais_distance_km"] = None

    return detections


# ── Scene timestamp extraction ────────────────────────────────────────────────

# Sentinel-1 filename pattern:
# S1A_IW_GRDH_1SDV_20230412T060000_20230412T060030_...
_S1_TS_RE = re.compile(
    r"S1[A-Z]_\w+_\w+_\w+_"
    r"(?P<start>\d{8}T\d{6})_"
    r"(?P<stop>\d{8}T\d{6})_"
)


def _parse_scene_times(scene_path: Path) -> tuple[str, str]:
    """
    Extract acquisition start/stop times from a Sentinel-1 filename.

    Returns (start_iso, end_iso) as ISO 8601 strings with UTC suffix.
    Raises ValueError if the pattern is not found.
    """
    name = scene_path.stem
    m = _S1_TS_RE.search(name)
    if not m:
        raise ValueError(
            f"Cannot extract acquisition time from filename: {scene_path.name!r}. "
            "Expected Sentinel-1 naming convention "
            "(e.g. S1A_IW_GRDH_1SDV_20230412T060000_20230412T060030_...)."
        )
    start_raw = m.group("start")   # e.g. 20230412T060000
    stop_raw = m.group("stop")

    # Extend window ±30 min to account for vessel movement during pass
    start_dt = datetime.strptime(start_raw, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
    stop_dt = datetime.strptime(stop_raw, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
    start_dt -= timedelta(minutes=30)
    stop_dt += timedelta(minutes=30)

    to_iso = lambda dt: dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return to_iso(start_dt), to_iso(stop_dt)


# ── Top-level orchestration ───────────────────────────────────────────────────

def run_ais_matching(
    detections: list[dict],
    scene_path: Path,
    radius_km: float = 1.0,
    api_token: str | None = None,
    ais_csv_path: Path | None = None,
) -> list[dict]:
    """
    Full AIS matching pipeline for a single SAR scene.

    1. Extracts acquisition time from *scene_path* filename.
    2. Builds a bounding box from detection coordinates (+0.1° padding).
    3. Fetches AIS vessels via GFW API or local CSV.
    4. Matches each detection to the nearest AIS vessel.
    5. Prints a summary and returns the annotated detection list.
    """
    if not detections:
        log.info("No detections provided — skipping AIS matching.")
        return detections

    # --- Scene time window -------------------------------------------------------
    try:
        start_time, end_time = _parse_scene_times(scene_path)
        log.info("Scene time window: %s → %s", start_time, end_time)
    except ValueError as exc:
        log.warning("%s — using ±1 h window around now.", exc)
        now = datetime.now(timezone.utc)
        start_time = (now - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        end_time = (now + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")

    # --- Bounding box from detections -------------------------------------------
    pad = 0.1
    lats = [float(d.get("lat", d.get("latitude", 0.0))) for d in detections]
    lons = [float(d.get("lon", d.get("longitude", 0.0))) for d in detections]
    bbox = (
        min(lons) - pad,
        min(lats) - pad,
        max(lons) + pad,
        max(lats) + pad,
    )
    log.info("Detection bbox (W,S,E,N): %.4f, %.4f, %.4f, %.4f", *bbox)

    # --- Fetch AIS data ----------------------------------------------------------
    ais_vessels = fetch_ais_vessels(
        bbox=bbox,
        start_time=start_time,
        end_time=end_time,
        api_token=api_token,
        ais_csv_path=ais_csv_path,
    )

    # --- Match -------------------------------------------------------------------
    annotated = match_detections_to_ais(detections, ais_vessels, radius_km=radius_km)

    # --- Summary -----------------------------------------------------------------
    n_total = len(annotated)
    n_matched = sum(1 for d in annotated if d.get("ais_match"))
    n_dark = n_total - n_matched
    print(
        f"AIS matching complete: {n_total} detections | "
        f"{n_matched} AIS matches | {n_dark} dark vessels "
        f"(radius={radius_km} km)"
    )

    return annotated


# ── CLI ───────────────────────────────────────────────────────────────────────

def _load_detections_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _save_detections_csv(detections: list[dict], path: Path) -> None:
    if not detections:
        log.warning("No detections to save.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(detections[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(detections)
    print(f"Output saved to: {path}")


if __name__ == "__main__":
    import os

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Match CFAR SAR detections against AIS vessel broadcasts."
    )
    parser.add_argument(
        "--detections",
        required=True,
        type=Path,
        help="Path to CFAR detections CSV (must have lat/lon columns).",
    )
    parser.add_argument(
        "--scene",
        required=True,
        type=Path,
        help="Sentinel-1 scene path (used for timestamp extraction from filename).",
    )
    parser.add_argument(
        "--radius-km",
        type=float,
        default=1.0,
        help="AIS match search radius in km (default: 1.0).",
    )
    parser.add_argument(
        "--api-token",
        type=str,
        default=None,
        help="Global Fishing Watch API token (overrides GFW_API_TOKEN env var).",
    )
    parser.add_argument(
        "--ais-csv",
        type=Path,
        default=None,
        help="Local AIS CSV fallback (columns: lat, lon, mmsi, vessel_name, flag, timestamp).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Output CSV path. Defaults to replacing '_cfar_detections.csv' "
            "with '_ais_detections.csv' in the input filename."
        ),
    )
    args = parser.parse_args()

    # Resolve API token
    token = args.api_token or os.getenv("GFW_API_TOKEN", "") or None

    # Resolve output path
    out_path: Path
    if args.out:
        out_path = args.out
    else:
        stem = args.detections.stem.replace("_cfar_detections", "_ais_detections")
        if "_cfar_detections" not in args.detections.stem:
            stem = args.detections.stem + "_ais_detections"
        out_path = args.detections.parent / (stem + ".csv")

    # Load, match, save
    dets = _load_detections_csv(args.detections)
    dets = run_ais_matching(
        detections=dets,
        scene_path=args.scene,
        radius_km=args.radius_km,
        api_token=token,
        ais_csv_path=args.ais_csv,
    )
    _save_detections_csv(dets, out_path)

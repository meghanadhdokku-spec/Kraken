"""Rasterio / geospatial helpers shared across modules."""

import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import xy as _xy


def read_band(tif_path: Path, band: int = 1) -> tuple[np.ndarray, dict]:
    with rasterio.open(tif_path) as src:
        data = src.read(band).astype(np.float32)
        meta = src.profile.copy()
    return data, meta


def pixel_to_lonlat(
    rows: np.ndarray,
    cols: np.ndarray,
    transform,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert arrays of (row, col) pixel coords to (lon, lat) using a rasterio transform."""
    lons, lats = zip(*[_xy(transform, r, c) for r, c in zip(rows, cols)])
    return np.array(lons), np.array(lats)


def save_geotiff(
    data: np.ndarray,
    meta: dict,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = meta.copy()
    if data.ndim == 2:
        meta.update(count=1)
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(data, 1)
    else:
        meta.update(count=data.shape[0])
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(data)


# ── GeoJSON ────────────────────────────────────────────────────────────────────

def detections_to_geojson(
    detections: list[dict],
    scene_name: str,
    detector: str,
) -> dict:
    """
    Convert a list of detection dicts to a GeoJSON FeatureCollection.

    Accepts both CFAR format (x, y, w, h) and YOLO format (x1, y1, x2, y2).
    Each detection must already have lon/lat fields.
    """
    CLASS_NAMES = {0: "non_vessel", 1: "vessel", 2: "fishing_vessel"}
    features = []
    for det in detections:
        lon = det.get("lon")
        lat = det.get("lat")
        if lon is None or lat is None:
            continue

        if "x1" in det:
            bbox_px = [det["x1"], det["y1"], det["x2"], det["y2"]]
        else:
            bbox_px = [det["x"], det["y"],
                       det["x"] + det["w"], det["y"] + det["h"]]

        cls = det.get("cls", 0)
        props: dict = {
            "detector":  detector,
            "scene":     scene_name,
            "cls":       cls,
            "cls_name":  CLASS_NAMES.get(cls, "vessel"),
            "bbox_px":   [round(v, 1) for v in bbox_px],
        }
        if det.get("conf") is not None:
            props["conf"] = round(float(det["conf"]), 4)
        if det.get("area_px") is not None:
            props["area_px"] = int(det["area_px"])

        features.append({
            "type": "Feature",
            "geometry": {
                "type":        "Point",
                "coordinates": [round(lon, 6), round(lat, 6)],
            },
            "properties": props,
        })

    return {"type": "FeatureCollection", "features": features}


def save_geojson(geojson: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(geojson, f, indent=2)
    print(f"[OK]   GeoJSON   → {out_path}")


def merge_geojson(*collections: dict) -> dict:
    """Merge multiple FeatureCollections into one."""
    features = []
    for fc in collections:
        features.extend(fc.get("features", []))
    return {"type": "FeatureCollection", "features": features}

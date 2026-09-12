"""Rasterio / geospatial helpers shared across modules."""

import json
import warnings
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

    Optional detection fields propagated to GeoJSON properties when present:
      conf         : detection confidence score
      area_px      : component area in pixels
      length_m     : estimated vessel length in metres
      width_m      : estimated vessel width in metres
      vessel_class : coarse vessel type label
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
        if det.get("length_m") is not None:
            props["length_m"] = det["length_m"]
        if det.get("width_m") is not None:
            props["width_m"] = det["width_m"]
        if det.get("vessel_class") is not None:
            props["vessel_class"] = det["vessel_class"]

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


# ── land mask ─────────────────────────────────────────────────────────────────

def _load_natural_earth_land():
    """
    Load Natural Earth land polygons, trying geopandas built-in then geodatasets.
    Returns a GeoDataFrame with a single dissolved land geometry in EPSG:4326.
    """
    import geopandas as gpd

    # geopandas < 1.0 ships 'naturalearth_lowres' as a bundled shapefile.
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            path = gpd.datasets.get_path("naturalearth_lowres")
        gdf = gpd.read_file(path)
        gdf = gdf[gdf.geometry.notnull()].copy()
        gdf["_land"] = 1
        return gdf.dissolve(by="_land")[["geometry"]]
    except Exception:
        pass

    # geopandas >= 1.0 removed bundled datasets; geodatasets is the companion.
    try:
        import geodatasets
        return gpd.read_file(geodatasets.get_path("naturalearth.land"))
    except Exception:
        pass

    raise RuntimeError(
        "Cannot load Natural Earth land polygons.\n"
        "Install geodatasets:  pip install geodatasets\n"
        "Or pass --land-mask-path /path/to/ne_10m_land.shp"
    )


def build_land_mask(
    transform,
    height: int,
    width: int,
    crs,
    land_shp_path: "Path | None" = None,
) -> np.ndarray:
    """
    Rasterize Natural Earth land polygons onto the scene pixel grid.

    Parameters
    ----------
    transform     : Affine transform of the scene (north-up expected).
    height, width : Scene pixel dimensions.
    crs           : Scene CRS (rasterio.crs.CRS or anything rasterio accepts).
    land_shp_path : Optional path to a local shapefile; defaults to the
                    Natural Earth bundled dataset.

    Returns
    -------
    uint8 mask — 1 = land, 0 = ocean.
    """
    import geopandas as gpd
    from rasterio.crs import CRS as RioCRS
    from rasterio.features import rasterize

    if land_shp_path is not None:
        land = gpd.read_file(str(land_shp_path))
    else:
        land = _load_natural_earth_land()

    scene_crs = crs if isinstance(crs, RioCRS) else RioCRS(crs)

    # Reproject land polygons to scene CRS when needed
    if land.crs is None or land.crs.to_epsg() != scene_crs.to_epsg():
        land = land.to_crs(scene_crs)

    shapes = [
        (geom, 1)
        for geom in land.geometry
        if geom is not None and not geom.is_empty
    ]

    if not shapes:
        return np.zeros((height, width), dtype=np.uint8)

    return rasterize(
        shapes,
        out_shape=(height, width),
        transform=transform,
        fill=0,
        dtype=np.uint8,
        all_touched=False,
    )

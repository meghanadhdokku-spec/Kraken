"""Rasterio / geospatial helpers shared across modules."""

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

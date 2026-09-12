"""
Unit tests for src/preprocess/sar_preprocess.py.

Uses synthetic data — no satellite imagery required.
Run with:  python -m pytest tests/test_preprocess.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocess.sar_preprocess import (
    clip_to_bbox,
    dn_to_sigma0_db,
    lee_filter,
    write_geotiff,
)


# ── helpers ────────────────────────────────────────────────────────────────────

def _meta(h=64, w=64, crs="EPSG:4326", west=0.0, south=0.0, east=1.0, north=1.0):
    return {
        "driver":    "GTiff",
        "dtype":     "float32",
        "width":     w,
        "height":    h,
        "count":     2,
        "crs":       CRS.from_epsg(4326),
        "transform": from_bounds(west, south, east, north, w, h),
        "nodata":    float("nan"),
    }


def _data(shape=(2, 64, 64), fill=1.0):
    return np.full(shape, fill, dtype=np.float32)


# ── clip_to_bbox ───────────────────────────────────────────────────────────────

class TestClipToBbox:
    def test_output_smaller_than_input(self):
        data = _data()
        meta = _meta(west=0.0, south=0.0, east=1.0, north=1.0)
        clipped, _ = clip_to_bbox(data, meta, bbox=(0.2, 0.2, 0.8, 0.8))
        assert clipped.shape[-1] < data.shape[-1]
        assert clipped.shape[-2] < data.shape[-2]

    def test_non_overlapping_bbox_raises(self):
        data = _data()
        meta = _meta(west=0.0, south=0.0, east=1.0, north=1.0)
        with pytest.raises(ValueError, match="does not intersect"):
            clip_to_bbox(data, meta, bbox=(5.0, 5.0, 6.0, 6.0))

    def test_meta_updated(self):
        data = _data()
        meta = _meta(west=0.0, south=0.0, east=1.0, north=1.0)
        _, new_meta = clip_to_bbox(data, meta, bbox=(0.0, 0.0, 0.5, 0.5))
        assert new_meta["width"]  == data.shape[-1] // 2
        assert new_meta["height"] == data.shape[-2] // 2

    def test_bands_preserved(self):
        data = _data(shape=(3, 64, 64))
        meta = _meta()
        meta["count"] = 3
        clipped, _ = clip_to_bbox(data, meta, bbox=(0.1, 0.1, 0.9, 0.9))
        assert clipped.shape[0] == 3

    def test_full_scene_bbox_returns_same_size(self):
        data = _data()
        meta = _meta(west=0.0, south=0.0, east=1.0, north=1.0)
        clipped, _ = clip_to_bbox(data, meta, bbox=(0.0, 0.0, 1.0, 1.0))
        assert clipped.shape == data.shape

    def test_partial_overlap_clipped_to_scene(self):
        data = _data()
        meta = _meta(west=0.0, south=0.0, east=1.0, north=1.0)
        # bbox extends beyond scene on east/north sides
        clipped, _ = clip_to_bbox(data, meta, bbox=(-0.5, -0.5, 0.6, 0.6))
        assert clipped.shape[-1] > 0 and clipped.shape[-2] > 0


# ── write_geotiff ──────────────────────────────────────────────────────────────

class TestWriteGeotiff:
    def test_2d_file_created(self, tmp_path):
        arr  = np.random.rand(32, 32).astype(np.float32)
        meta = _meta(h=32, w=32)
        meta["count"] = 1
        out  = tmp_path / "out.tif"
        write_geotiff(arr, meta, out)
        assert out.exists()

    def test_3d_file_created(self, tmp_path):
        arr  = np.random.rand(2, 32, 32).astype(np.float32)
        meta = _meta(h=32, w=32)
        out  = tmp_path / "out.tif"
        write_geotiff(arr, meta, out)
        assert out.exists()

    def test_2d_roundtrip(self, tmp_path):
        arr  = np.arange(1024, dtype=np.float32).reshape(32, 32)
        meta = _meta(h=32, w=32)
        meta["count"] = 1
        out  = tmp_path / "rt.tif"
        write_geotiff(arr, meta, out)
        with rasterio.open(out) as src:
            assert np.allclose(src.read(1), arr, atol=1e-5)

    def test_3d_band_count(self, tmp_path):
        arr  = np.random.rand(2, 32, 32).astype(np.float32)
        meta = _meta(h=32, w=32)
        out  = tmp_path / "bands.tif"
        write_geotiff(arr, meta, out)
        with rasterio.open(out) as src:
            assert src.count == 2

    def test_parent_dirs_created(self, tmp_path):
        arr  = np.ones((2, 8, 8), dtype=np.float32)
        meta = _meta(h=8, w=8)
        out  = tmp_path / "deep" / "nested" / "out.tif"
        write_geotiff(arr, meta, out)
        assert out.exists()


# ── dn_to_sigma0_db ────────────────────────────────────────────────────────────
# (also covered in test_cfar.py — kept minimal here)

class TestDNToSigma0DB:
    def test_zero_input_is_nan(self):
        out = dn_to_sigma0_db(np.array([[0]], dtype=np.float32))
        assert np.isnan(out[0, 0])

    def test_positive_dn_finite(self):
        out = dn_to_sigma0_db(np.array([[500]], dtype=np.float32))
        assert np.isfinite(out[0, 0])

    def test_output_dtype_float32(self):
        out = dn_to_sigma0_db(np.ones((4, 4), dtype=np.uint16))
        assert out.dtype == np.float32


# ── lee_filter ─────────────────────────────────────────────────────────────────
# (more thorough tests in test_cfar.py)

class TestLeeFilterPreprocess:
    def test_no_nan_bleed_on_uniform(self):
        img = np.ones((32, 32), dtype=np.float32) * 5.0
        out = lee_filter(img, kernel_size=3)
        assert not np.any(np.isnan(out))

    def test_negative_variance_does_not_produce_nan(self):
        # Regression: FP subtraction (local_sq - local_mean²) can go negative.
        # np.maximum guard must prevent NaN in weight.
        rng = np.random.default_rng(7)
        img = rng.uniform(0.99, 1.01, size=(64, 64)).astype(np.float32)
        out = lee_filter(img, kernel_size=7)
        assert not np.any(np.isnan(out))

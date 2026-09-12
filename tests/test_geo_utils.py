"""Unit tests for src/utils/geo_utils.py"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.geo_utils import (
    detections_to_geojson,
    merge_geojson,
    pixel_to_lonlat,
    read_band,
    save_geojson,
    save_geotiff,
)


# ── fixtures ───────────────────────────────────────────────────────────────────

def _make_tif(tmp_path: Path, data: np.ndarray, bands: int = 1) -> Path:
    """Write a minimal GeoTIFF and return its path."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "test.tif"
    h, w = data.shape[-2], data.shape[-1]
    transform = from_bounds(0, 0, 1, 1, w, h)
    profile = {
        "driver": "GTiff",
        "dtype":  "float32",
        "width":  w,
        "height": h,
        "count":  bands,
        "crs":    "EPSG:4326",
        "transform": transform,
    }
    with rasterio.open(path, "w", **profile) as dst:
        if bands == 1:
            dst.write(data.astype("float32"), 1)
        else:
            dst.write(data.astype("float32"))
    return path


def _cfar_det(lon=1.0, lat=2.0, x=10, y=20, w=5, h=5, cls=1, area_px=25):
    return {"lon": lon, "lat": lat, "x": x, "y": y, "w": w, "h": h,
            "cls": cls, "area_px": area_px}


def _yolo_det(lon=3.0, lat=4.0, x1=10, y1=20, x2=15, y2=25, conf=0.9, cls=2):
    return {"lon": lon, "lat": lat, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "conf": conf, "cls": cls}


# ── detections_to_geojson ──────────────────────────────────────────────────────

class TestDetectionsToGeojson:
    def test_empty_input_returns_empty_collection(self):
        fc = detections_to_geojson([], "scene", "cfar")
        assert fc["type"] == "FeatureCollection"
        assert fc["features"] == []

    def test_missing_lon_skipped(self):
        det = {"lat": 1.0, "x": 0, "y": 0, "w": 5, "h": 5}
        fc = detections_to_geojson([det], "s", "cfar")
        assert fc["features"] == []

    def test_missing_lat_skipped(self):
        det = {"lon": 1.0, "x": 0, "y": 0, "w": 5, "h": 5}
        fc = detections_to_geojson([det], "s", "cfar")
        assert fc["features"] == []

    def test_cfar_format_bbox(self):
        det = _cfar_det(x=10, y=20, w=5, h=8)
        fc  = detections_to_geojson([det], "scene", "cfar")
        bbox = fc["features"][0]["properties"]["bbox_px"]
        assert bbox == [10.0, 20.0, 15.0, 28.0]

    def test_yolo_format_bbox(self):
        det = _yolo_det(x1=10, y1=20, x2=15, y2=25)
        fc  = detections_to_geojson([det], "scene", "yolo")
        bbox = fc["features"][0]["properties"]["bbox_px"]
        assert bbox == [10.0, 20.0, 15.0, 25.0]

    def test_geometry_is_point(self):
        fc = detections_to_geojson([_cfar_det()], "s", "cfar")
        geom = fc["features"][0]["geometry"]
        assert geom["type"] == "Point"
        assert len(geom["coordinates"]) == 2

    def test_coordinates_rounded_to_6dp(self):
        det = _cfar_det(lon=1.123456789, lat=-2.987654321)
        fc  = detections_to_geojson([det], "s", "cfar")
        lon, lat = fc["features"][0]["geometry"]["coordinates"]
        assert lon == pytest.approx(1.123457, abs=1e-6)
        assert lat == pytest.approx(-2.987654, abs=1e-6)

    def test_detector_and_scene_in_properties(self):
        fc = detections_to_geojson([_cfar_det()], "MY_SCENE", "cfar")
        props = fc["features"][0]["properties"]
        assert props["detector"] == "cfar"
        assert props["scene"] == "MY_SCENE"

    def test_class_names_mapped(self):
        cases = [(0, "non_vessel"), (1, "vessel"), (2, "fishing_vessel")]
        for cls_id, expected_name in cases:
            det = _cfar_det(cls=cls_id)
            fc  = detections_to_geojson([det], "s", "cfar")
            assert fc["features"][0]["properties"]["cls_name"] == expected_name

    def test_unknown_class_falls_back_to_vessel(self):
        det = _cfar_det(cls=99)
        fc  = detections_to_geojson([det], "s", "cfar")
        assert fc["features"][0]["properties"]["cls_name"] == "vessel"

    def test_conf_included_when_present(self):
        det = _yolo_det(conf=0.876543)
        fc  = detections_to_geojson([det], "s", "yolo")
        props = fc["features"][0]["properties"]
        assert "conf" in props
        assert props["conf"] == pytest.approx(0.8765, abs=1e-4)

    def test_conf_absent_when_not_in_det(self):
        det = _cfar_det()   # no "conf" key
        fc  = detections_to_geojson([det], "s", "cfar")
        assert "conf" not in fc["features"][0]["properties"]

    def test_area_px_included_when_present(self):
        det = _cfar_det(area_px=42)
        fc  = detections_to_geojson([det], "s", "cfar")
        assert fc["features"][0]["properties"]["area_px"] == 42

    def test_area_px_absent_when_not_in_det(self):
        det = _yolo_det()   # no "area_px" key
        fc  = detections_to_geojson([det], "s", "yolo")
        assert "area_px" not in fc["features"][0]["properties"]

    def test_multiple_detections_all_included(self):
        dets = [_cfar_det(lon=float(i), lat=0.0) for i in range(5)]
        fc   = detections_to_geojson(dets, "s", "cfar")
        assert len(fc["features"]) == 5

    def test_mixed_missing_and_valid(self):
        dets = [
            {"lat": 1.0, "x": 0, "y": 0, "w": 5, "h": 5},  # missing lon
            _cfar_det(lon=2.0, lat=3.0),
        ]
        fc = detections_to_geojson(dets, "s", "cfar")
        assert len(fc["features"]) == 1


# ── save_geojson ───────────────────────────────────────────────────────────────

class TestSaveGeojson:
    def test_file_created(self, tmp_path):
        fc  = detections_to_geojson([_cfar_det()], "s", "cfar")
        out = tmp_path / "out.geojson"
        save_geojson(fc, out)
        assert out.exists()

    def test_valid_json(self, tmp_path):
        fc  = detections_to_geojson([_cfar_det()], "s", "cfar")
        out = tmp_path / "out.geojson"
        save_geojson(fc, out)
        loaded = json.loads(out.read_text())
        assert loaded["type"] == "FeatureCollection"

    def test_parent_dirs_created(self, tmp_path):
        fc  = {"type": "FeatureCollection", "features": []}
        out = tmp_path / "deep" / "nested" / "out.geojson"
        save_geojson(fc, out)
        assert out.exists()

    def test_roundtrip(self, tmp_path):
        fc  = detections_to_geojson([_yolo_det()], "scene", "yolo")
        out = tmp_path / "rt.geojson"
        save_geojson(fc, out)
        loaded = json.loads(out.read_text())
        assert loaded["features"][0]["properties"]["detector"] == "yolo"


# ── merge_geojson ──────────────────────────────────────────────────────────────

class TestMergeGeojson:
    def test_two_collections_merged(self):
        fc1 = detections_to_geojson([_cfar_det(lon=1.0)], "s", "cfar")
        fc2 = detections_to_geojson([_yolo_det(lon=2.0)], "s", "yolo")
        merged = merge_geojson(fc1, fc2)
        assert merged["type"] == "FeatureCollection"
        assert len(merged["features"]) == 2

    def test_empty_collections(self):
        fc1 = {"type": "FeatureCollection", "features": []}
        fc2 = {"type": "FeatureCollection", "features": []}
        merged = merge_geojson(fc1, fc2)
        assert merged["features"] == []

    def test_single_collection_passthrough(self):
        fc = detections_to_geojson([_cfar_det()], "s", "cfar")
        merged = merge_geojson(fc)
        assert len(merged["features"]) == len(fc["features"])

    def test_no_collections(self):
        merged = merge_geojson()
        assert merged == {"type": "FeatureCollection", "features": []}

    def test_three_collections(self):
        fc1 = detections_to_geojson([_cfar_det(lon=1.0)], "s", "cfar")
        fc2 = detections_to_geojson([_yolo_det(lon=2.0)], "s", "yolo")
        fc3 = detections_to_geojson([_cfar_det(lon=3.0)], "s", "cfar")
        merged = merge_geojson(fc1, fc2, fc3)
        assert len(merged["features"]) == 3

    def test_detector_labels_preserved(self):
        fc1 = detections_to_geojson([_cfar_det()], "s", "cfar")
        fc2 = detections_to_geojson([_yolo_det()], "s", "yolo")
        merged = merge_geojson(fc1, fc2)
        detectors = {f["properties"]["detector"] for f in merged["features"]}
        assert detectors == {"cfar", "yolo"}


# ── pixel_to_lonlat ────────────────────────────────────────────────────────────

class TestPixelToLonlat:
    def _transform(self):
        # 10×10 image covering lon 0→1, lat 0→1
        return from_bounds(0.0, 0.0, 1.0, 1.0, 10, 10)

    def test_returns_arrays(self):
        t = self._transform()
        lons, lats = pixel_to_lonlat(np.array([0]), np.array([0]), t)
        assert isinstance(lons, np.ndarray)
        assert isinstance(lats, np.ndarray)

    def test_single_pixel_centre(self):
        t = self._transform()
        lons, lats = pixel_to_lonlat(np.array([5]), np.array([5]), t)
        assert 0.0 <= float(lons[0]) <= 1.0
        assert 0.0 <= float(lats[0]) <= 1.0

    def test_multiple_pixels_length_matches(self):
        t = self._transform()
        rows = np.array([0, 2, 4, 6])
        cols = np.array([1, 3, 5, 7])
        lons, lats = pixel_to_lonlat(rows, cols, t)
        assert len(lons) == 4
        assert len(lats) == 4

    def test_col_increases_lon(self):
        t = self._transform()
        lons, _ = pixel_to_lonlat(np.array([5, 5]), np.array([2, 8]), t)
        assert lons[0] < lons[1]

    def test_row_increases_decreases_lat(self):
        # rasterio: row 0 = top = higher lat
        t = self._transform()
        _, lats = pixel_to_lonlat(np.array([2, 8]), np.array([5, 5]), t)
        assert lats[0] > lats[1]


# ── read_band / save_geotiff ───────────────────────────────────────────────────

class TestReadBandSaveGeotiff:
    def test_read_band_returns_float32(self, tmp_path):
        arr = np.ones((8, 8), dtype=np.float32) * 3.14
        tif = _make_tif(tmp_path, arr)
        data, meta = read_band(tif, band=1)
        assert data.dtype == np.float32

    def test_read_band_values_match(self, tmp_path):
        arr = np.arange(64, dtype=np.float32).reshape(8, 8)
        tif = _make_tif(tmp_path, arr)
        data, _ = read_band(tif)
        assert np.allclose(data, arr)

    def test_read_band_meta_has_profile_keys(self, tmp_path):
        arr = np.ones((4, 4), dtype=np.float32)
        tif = _make_tif(tmp_path, arr)
        _, meta = read_band(tif)
        for key in ("driver", "dtype", "width", "height", "count"):
            assert key in meta

    def test_save_geotiff_2d_creates_file(self, tmp_path):
        arr  = np.random.rand(8, 8).astype(np.float32)
        tif  = _make_tif(tmp_path / "src", arr)
        _, meta = read_band(tif)
        out  = tmp_path / "out" / "saved.tif"
        save_geotiff(arr, meta, out)
        assert out.exists()

    def test_save_geotiff_2d_roundtrip(self, tmp_path):
        arr = np.random.rand(8, 8).astype(np.float32)
        tif = _make_tif(tmp_path / "src", arr)
        _, meta = read_band(tif)
        out = tmp_path / "saved.tif"
        save_geotiff(arr, meta, out)
        data, _ = read_band(out)
        assert np.allclose(data, arr, atol=1e-5)

    def test_save_geotiff_3d_roundtrip(self, tmp_path):
        arr = np.random.rand(2, 8, 8).astype(np.float32)
        tif = _make_tif(tmp_path / "src", arr, bands=2)
        _, meta = read_band(tif)
        meta["count"] = 2
        out = tmp_path / "saved.tif"
        save_geotiff(arr, meta, out)
        with rasterio.open(out) as src:
            assert src.count == 2

    def test_save_geotiff_creates_parent_dirs(self, tmp_path):
        arr = np.ones((4, 4), dtype=np.float32)
        tif = _make_tif(tmp_path / "src", arr)
        _, meta = read_band(tif)
        out = tmp_path / "a" / "b" / "c" / "out.tif"
        save_geotiff(arr, meta, out)
        assert out.exists()

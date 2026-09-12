"""Unit tests for src/utils/geo_utils.py"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.geo_utils import (
    detections_to_geojson,
    merge_geojson,
    save_geojson,
)


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


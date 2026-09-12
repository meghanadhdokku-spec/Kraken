"""
Unit tests for src/match/ais_match.py.

All tests use synthetic data — no network or API keys required.
Run with:  python -m pytest tests/test_ais_match.py -v
"""

import csv
import sys
from datetime import timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.match.ais_match import (
    _haversine_km,
    _parse_iso,
    _parse_scene_times,
    fetch_ais_vessels,
    match_detections_to_ais,
)


# ── haversine ──────────────────────────────────────────────────────────────────

class TestHaversine:
    def test_same_point_is_zero(self):
        assert _haversine_km(0.0, 0.0, 0.0, 0.0) == pytest.approx(0.0)

    def test_equator_one_degree_lon(self):
        # 1° longitude at equator ≈ 111.32 km
        d = _haversine_km(0.0, 0.0, 0.0, 1.0)
        assert 111.0 < d < 112.0

    def test_poles_half_circumference(self):
        # N pole to S pole ≈ 20_015 km
        d = _haversine_km(90.0, 0.0, -90.0, 0.0)
        assert 20_000 < d < 20_030

    def test_symmetry(self):
        a = _haversine_km(1.0, 2.0, 3.0, 4.0)
        b = _haversine_km(3.0, 4.0, 1.0, 2.0)
        assert a == pytest.approx(b)

    def test_positive_result(self):
        assert _haversine_km(10.0, 20.0, 11.0, 21.0) > 0


# ── _parse_iso ─────────────────────────────────────────────────────────────────

class TestParseIso:
    def test_full_datetime(self):
        dt = _parse_iso("2023-04-12T06:30:00")
        assert dt.year == 2023 and dt.month == 4 and dt.day == 12
        assert dt.hour == 6 and dt.minute == 30

    def test_z_suffix_stripped(self):
        dt = _parse_iso("2023-04-12T06:30:00Z")
        assert dt.tzinfo == timezone.utc

    def test_date_only(self):
        dt = _parse_iso("2023-04-12")
        assert dt.year == 2023 and dt.month == 4 and dt.day == 12

    def test_hhmm_only(self):
        dt = _parse_iso("2023-04-12T06:30")
        assert dt.hour == 6 and dt.minute == 30

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            _parse_iso("not-a-date")


# ── _parse_scene_times ─────────────────────────────────────────────────────────

class TestParseSceneTimes:
    _VALID = Path(
        "S1A_IW_GRDH_1SDV_20230412T060000_20230412T060030_047000_05A000_1234"
    )

    def test_returns_two_iso_strings(self):
        start, end = _parse_scene_times(self._VALID)
        assert "T" in start and "Z" in start
        assert "T" in end   and "Z" in end

    def test_window_extended_30min(self):
        start, end = _parse_scene_times(self._VALID)
        # original start 06:00:00 − 30 min = 05:30:00
        assert "T05:30:00Z" in start
        # original end 06:00:30 + 30 min = 06:30:30
        assert "T06:30:30Z" in end

    def test_invalid_filename_raises(self):
        with pytest.raises(ValueError):
            _parse_scene_times(Path("not_a_sentinel_filename.tif"))


# ── fetch_ais_vessels (CSV path) ───────────────────────────────────────────────

class TestFetchAisFromCSV:
    def _write_csv(self, tmp_path: Path, rows: list[dict]) -> Path:
        path = tmp_path / "ais.csv"
        if not rows:
            path.write_text("lat,lon,mmsi,vessel_name,flag,timestamp\n")
            return path
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        return path

    def _row(self, lat, lon, ts="2023-04-12T06:00:00"):
        return {"lat": lat, "lon": lon, "mmsi": "123456789",
                "vessel_name": "MV Test", "flag": "GH", "timestamp": ts}

    def test_inside_bbox_and_time_returned(self, tmp_path):
        csv_path = self._write_csv(tmp_path, [self._row(5.0, 2.0)])
        vessels = fetch_ais_vessels(
            bbox=(0.0, 0.0, 10.0, 10.0),
            start_time="2023-04-12T05:00:00Z",
            end_time="2023-04-12T07:00:00Z",
            ais_csv_path=csv_path,
        )
        assert len(vessels) == 1
        assert vessels[0]["vessel_name"] == "MV Test"

    def test_outside_bbox_filtered(self, tmp_path):
        csv_path = self._write_csv(tmp_path, [self._row(50.0, 50.0)])
        vessels = fetch_ais_vessels(
            bbox=(0.0, 0.0, 10.0, 10.0),
            start_time="2023-04-12T05:00:00Z",
            end_time="2023-04-12T07:00:00Z",
            ais_csv_path=csv_path,
        )
        assert vessels == []

    def test_outside_time_filtered(self, tmp_path):
        csv_path = self._write_csv(tmp_path, [self._row(5.0, 2.0, ts="2023-04-12T20:00:00")])
        vessels = fetch_ais_vessels(
            bbox=(0.0, 0.0, 10.0, 10.0),
            start_time="2023-04-12T05:00:00Z",
            end_time="2023-04-12T07:00:00Z",
            ais_csv_path=csv_path,
        )
        assert vessels == []

    def test_missing_timestamp_included(self, tmp_path):
        row = {"lat": 5.0, "lon": 2.0, "mmsi": "1", "vessel_name": "X",
               "flag": "GH", "timestamp": ""}
        csv_path = self._write_csv(tmp_path, [row])
        vessels = fetch_ais_vessels(
            bbox=(0.0, 0.0, 10.0, 10.0),
            start_time="2023-04-12T05:00:00Z",
            end_time="2023-04-12T07:00:00Z",
            ais_csv_path=csv_path,
        )
        assert len(vessels) == 1

    def test_missing_csv_returns_empty(self, tmp_path):
        vessels = fetch_ais_vessels(
            bbox=(0.0, 0.0, 10.0, 10.0),
            start_time="2023-04-12T05:00:00Z",
            end_time="2023-04-12T07:00:00Z",
            ais_csv_path=tmp_path / "nonexistent.csv",
        )
        assert vessels == []

    def test_no_token_no_csv_returns_empty(self):
        vessels = fetch_ais_vessels(
            bbox=(0.0, 0.0, 1.0, 1.0),
            start_time="2023-04-12T05:00:00Z",
            end_time="2023-04-12T07:00:00Z",
        )
        assert vessels == []


# ── match_detections_to_ais ────────────────────────────────────────────────────

class TestMatchDetectionsToAIS:
    def _det(self, lat=5.0, lon=2.0):
        return {"lat": lat, "lon": lon, "x": 10, "y": 10, "w": 5, "h": 5}

    def _vessel(self, lat=5.0, lon=2.0):
        return {"lat": lat, "lon": lon, "vessel_id": "V1",
                "vessel_name": "MV Test", "flag": "GH"}

    def test_nearby_vessel_matched(self):
        dets    = [self._det(5.0, 2.0)]
        vessels = [self._vessel(5.0, 2.0)]
        result  = match_detections_to_ais(dets, vessels, radius_km=1.0)
        assert result[0]["ais_match"] is True
        assert result[0]["dark_vessel"] is False

    def test_distant_vessel_dark(self):
        dets    = [self._det(5.0, 2.0)]
        vessels = [self._vessel(10.0, 10.0)]
        result  = match_detections_to_ais(dets, vessels, radius_km=1.0)
        assert result[0]["ais_match"] is False
        assert result[0]["dark_vessel"] is True

    def test_empty_ais_all_dark(self):
        dets   = [self._det(), self._det(6.0, 3.0)]
        result = match_detections_to_ais(dets, [], radius_km=1.0)
        assert all(d["dark_vessel"] for d in result)

    def test_empty_detections_returns_empty(self):
        result = match_detections_to_ais([], [self._vessel()], radius_km=1.0)
        assert result == []

    def test_match_fields_populated(self):
        dets    = [self._det(5.0, 2.0)]
        vessels = [self._vessel(5.0, 2.0)]
        result  = match_detections_to_ais(dets, vessels, radius_km=1.0)
        d = result[0]
        assert d["matched_vessel_id"] == "V1"
        assert d["matched_vessel_name"] == "MV Test"
        assert d["matched_flag"] == "GH"
        assert d["ais_distance_km"] == pytest.approx(0.0, abs=0.01)

    def test_dark_fields_none(self):
        dets   = [self._det(5.0, 2.0)]
        result = match_detections_to_ais(dets, [], radius_km=1.0)
        d = result[0]
        assert d["matched_vessel_id"] is None
        assert d["matched_vessel_name"] is None
        assert d["matched_flag"] is None
        assert d["ais_distance_km"] is None

    def test_nearest_vessel_selected(self):
        dets = [self._det(5.0, 2.0)]
        vessels = [
            {"lat": 5.001, "lon": 2.001, "vessel_id": "NEAR",
             "vessel_name": "Near", "flag": "GH"},
            {"lat": 5.05,  "lon": 2.05,  "vessel_id": "FAR",
             "vessel_name": "Far",  "flag": "GH"},
        ]
        result = match_detections_to_ais(dets, vessels, radius_km=10.0)
        assert result[0]["matched_vessel_id"] == "NEAR"

    def test_modifies_in_place(self):
        det    = self._det()
        result = match_detections_to_ais([det], [], radius_km=1.0)
        assert result[0] is det

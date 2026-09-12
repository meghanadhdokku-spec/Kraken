"""
Unit tests for xView3 dataset preparation.

Tests all coordinate-mapping, class-assignment, label-writing,
and image-normalisation logic without touching the filesystem
more than necessary (uses tmp_path fixtures).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.train.xview3_prep import (
    Detection,
    ChipInfo,
    CLASS_MAP,
    _parse_bool,
    _parse_float,
    _confidence_ok,
    load_labels_csv,
    discover_chips,
    build_scene_index,
    detection_to_chip,
    assign_class,
    write_yolo_labels,
    _norm_band,
)


# ── helper factory ─────────────────────────────────────────────────────────────

def make_det(
    scene_id="S1_SCENE",
    scene_row=400.0,
    scene_col=400.0,
    is_vessel=True,
    is_fishing=True,
    length_m=50.0,
    confidence="HIGH",
) -> Detection:
    return Detection(
        scene_id=scene_id,
        scene_row=scene_row,
        scene_col=scene_col,
        is_vessel=is_vessel,
        is_fishing=is_fishing,
        length_m=length_m,
        confidence=confidence,
    )


def make_chip(
    path=Path("S1_SCENE_0000_0000.tif"),
    scene_id="S1_SCENE",
    row_min=0,
    col_min=0,
    chip_size=800,
) -> ChipInfo:
    return ChipInfo(path=path, scene_id=scene_id,
                    row_min=row_min, col_min=col_min, chip_size=chip_size)


# ── _parse_bool ────────────────────────────────────────────────────────────────

class TestParseBool:
    @pytest.mark.parametrize("val,expected", [
        ("1",   True),
        ("1.0", True),
        ("0",   False),
        ("0.0", False),
        ("",    None),
        ("nan", None),
        ("true",  True),
        ("false", False),
    ])
    def test_values(self, val, expected):
        assert _parse_bool(val) == expected


# ── _confidence_ok ─────────────────────────────────────────────────────────────

class TestConfidenceOk:
    def test_high_passes_all_thresholds(self):
        assert _confidence_ok("HIGH", "HIGH")
        assert _confidence_ok("HIGH", "MEDIUM")
        assert _confidence_ok("HIGH", "LOW")

    def test_low_fails_high_threshold(self):
        assert not _confidence_ok("LOW", "HIGH")
        assert not _confidence_ok("LOW", "MEDIUM")
        assert     _confidence_ok("LOW", "LOW")

    def test_medium_mid_behaviour(self):
        assert     _confidence_ok("MEDIUM", "MEDIUM")
        assert not _confidence_ok("MEDIUM", "HIGH")


# ── load_labels_csv ────────────────────────────────────────────────────────────

class TestLoadLabelsCsv:
    def test_basic_parse(self, tmp_path):
        csv = tmp_path / "labels.csv"
        csv.write_text(
            "scene_id,detect_scene_row,detect_scene_column,"
            "is_vessel,is_fishing,vessel_length_m,confidence\n"
            "SCENE1,100,200,1,1,80.0,HIGH\n"
            "SCENE1,300,400,0,,nan,MEDIUM\n"
        )
        dets = load_labels_csv(csv, min_confidence="LOW")
        assert len(dets) == 2
        assert dets[0].is_vessel is True
        assert dets[0].is_fishing is True
        assert dets[1].is_vessel is False
        assert np.isnan(dets[1].length_m)

    def test_confidence_filter(self, tmp_path):
        csv = tmp_path / "labels.csv"
        csv.write_text(
            "scene_id,detect_scene_row,detect_scene_column,"
            "is_vessel,is_fishing,vessel_length_m,confidence\n"
            "SC,1,1,1,1,20,HIGH\n"
            "SC,2,2,1,0,20,LOW\n"
        )
        dets = load_labels_csv(csv, min_confidence="HIGH")
        assert len(dets) == 1
        assert dets[0].confidence == "HIGH"

    def test_bad_rows_skipped(self, tmp_path):
        csv = tmp_path / "labels.csv"
        csv.write_text(
            "scene_id,detect_scene_row,detect_scene_column,is_vessel,is_fishing,vessel_length_m,confidence\n"
            "SC,notanumber,200,1,1,20,HIGH\n"
            "SC,100,200,1,1,20,HIGH\n"
        )
        dets = load_labels_csv(csv, min_confidence="LOW")
        assert len(dets) == 1


# ── discover_chips ─────────────────────────────────────────────────────────────

class TestDiscoverChips:
    def test_valid_chips_found(self, tmp_path):
        (tmp_path / "SCENE_0000_0000.tif").touch()
        (tmp_path / "SCENE_0800_0000.tif").touch()
        (tmp_path / "not_a_chip.txt").touch()
        chips = discover_chips(tmp_path)
        assert len(chips) == 2

    def test_chip_fields_parsed(self, tmp_path):
        (tmp_path / "S1A_IW_0200_0400.tif").touch()
        chips = discover_chips(tmp_path)
        assert chips[0].row_min == 200
        assert chips[0].col_min == 400
        assert chips[0].scene_id == "S1A_IW"

    def test_non_tif_ignored(self, tmp_path):
        (tmp_path / "SCENE_0000_0000.png").touch()
        chips = discover_chips(tmp_path)
        assert chips == []

    def test_empty_dir(self, tmp_path):
        assert discover_chips(tmp_path) == []


# ── build_scene_index ──────────────────────────────────────────────────────────

class TestBuildSceneIndex:
    def test_groups_by_scene(self):
        chips = [
            make_chip(scene_id="A", row_min=0),
            make_chip(scene_id="A", row_min=800),
            make_chip(scene_id="B", row_min=0),
        ]
        idx = build_scene_index(chips)
        assert len(idx["A"]) == 2
        assert len(idx["B"]) == 1


# ── detection_to_chip ──────────────────────────────────────────────────────────

class TestDetectionToChip:
    """Core coordinate mapping — this is the most critical logic."""

    def test_detection_inside_chip(self):
        chip = make_chip(row_min=0, col_min=0, chip_size=800)
        det  = make_det(scene_row=400, scene_col=400, length_m=80)
        box  = detection_to_chip(det, chip)
        assert box is not None
        cx, cy, w, h = box
        assert pytest.approx(cx, abs=1e-4) == 400 / 800
        assert pytest.approx(cy, abs=1e-4) == 400 / 800
        assert 0 < w < 1
        assert 0 < h < 1

    def test_detection_outside_chip(self):
        chip = make_chip(row_min=0, col_min=0, chip_size=800)
        det  = make_det(scene_row=900, scene_col=400)
        assert detection_to_chip(det, chip) is None

    def test_detection_in_offset_chip(self):
        chip = make_chip(row_min=800, col_min=1600, chip_size=800)
        det  = make_det(scene_row=1000, scene_col=1800)
        box  = detection_to_chip(det, chip)
        assert box is not None
        cx, cy, w, h = box
        assert pytest.approx(cx, abs=1e-4) == (1800 - 1600) / 800
        assert pytest.approx(cy, abs=1e-4) == (1000 - 800)  / 800

    def test_on_top_left_boundary(self):
        chip = make_chip(row_min=0, col_min=0, chip_size=800)
        det  = make_det(scene_row=0, scene_col=0)
        assert detection_to_chip(det, chip) is not None

    def test_on_bottom_right_boundary_excluded(self):
        chip = make_chip(row_min=0, col_min=0, chip_size=800)
        det  = make_det(scene_row=800, scene_col=800)
        assert detection_to_chip(det, chip) is None

    def test_nan_length_uses_default(self):
        chip = make_chip(chip_size=800)
        det  = make_det(scene_row=400, scene_col=400, length_m=float("nan"))
        box  = detection_to_chip(det, chip)
        assert box is not None, "NaN length should fall back to 20 m"

    def test_box_clamped_to_unit(self):
        chip = make_chip(chip_size=800)
        det  = make_det(scene_row=400, scene_col=400, length_m=10000)
        cx, cy, w, h = detection_to_chip(det, chip)
        assert w <= 1.0
        assert h <= 1.0

    def test_normalisation_in_unit_range(self):
        chip = make_chip(chip_size=800)
        det  = make_det(scene_row=400, scene_col=400, length_m=50)
        cx, cy, w, h = detection_to_chip(det, chip)
        assert 0.0 <= cx <= 1.0
        assert 0.0 <= cy <= 1.0
        assert 0.0 <  w  <= 1.0
        assert 0.0 <  h  <= 1.0


# ── assign_class ───────────────────────────────────────────────────────────────

class TestAssignClass:
    @pytest.mark.parametrize("is_vessel,is_fishing,expected", [
        (False, None,  0),   # non-vessel
        (True,  False, 1),   # vessel
        (True,  True,  2),   # fishing vessel  ← IUU target
        (True,  None,  1),   # vessel, fishing unknown
    ])
    def test_class_map(self, is_vessel, is_fishing, expected):
        det = make_det(is_vessel=is_vessel, is_fishing=is_fishing)
        assert assign_class(det) == expected


# ── write_yolo_labels ──────────────────────────────────────────────────────────

class TestWriteYoloLabels:
    def test_label_file_created(self, tmp_path):
        chip = make_chip()
        det  = make_det(scene_row=400, scene_col=400)
        out  = tmp_path / "chip.txt"
        n    = write_yolo_labels(chip, [det], out)
        assert n == 1
        assert out.exists()

    def test_label_format(self, tmp_path):
        chip = make_chip(chip_size=800)
        det  = make_det(scene_row=400, scene_col=400, is_vessel=True,
                        is_fishing=True, length_m=80)
        out  = tmp_path / "chip.txt"
        write_yolo_labels(chip, [det], out)
        line = out.read_text().strip().split("\n")[0]
        parts = line.split()
        assert len(parts) == 5
        cls, cx, cy, w, h = int(parts[0]), *map(float, parts[1:])
        assert cls == 2          # fishing_vessel
        assert 0 <= cx <= 1
        assert 0 <= cy <= 1
        assert 0 <  w  <= 1
        assert 0 <  h  <= 1

    def test_out_of_chip_detections_excluded(self, tmp_path):
        chip = make_chip()
        det  = make_det(scene_row=9999, scene_col=9999)
        out  = tmp_path / "chip.txt"
        n    = write_yolo_labels(chip, [det], out)
        assert n == 0
        assert out.read_text().strip() == ""

    def test_multiple_detections(self, tmp_path):
        chip = make_chip(chip_size=800)
        dets = [
            make_det(scene_row=100, scene_col=100),
            make_det(scene_row=600, scene_col=600),
            make_det(scene_row=9999, scene_col=9999),  # outside
        ]
        out = tmp_path / "chip.txt"
        n   = write_yolo_labels(chip, dets, out)
        assert n == 2
        assert len(out.read_text().strip().split("\n")) == 2


# ── _norm_band ─────────────────────────────────────────────────────────────────

class TestNormBand:
    def test_output_dtype_uint8(self):
        band = np.random.rand(64, 64).astype(np.float32)
        out  = _norm_band(band)
        assert out.dtype == np.uint8

    def test_range_within_0_255(self):
        band = np.random.rand(64, 64).astype(np.float32) * 100 - 50
        out  = _norm_band(band)
        assert out.min() >= 0
        assert out.max() <= 255

    def test_constant_band_returns_zeros(self):
        band = np.full((32, 32), 5.0, dtype=np.float32)
        out  = _norm_band(band)
        assert np.all(out == 0)

    def test_all_nan_returns_zeros(self):
        band = np.full((16, 16), np.nan, dtype=np.float32)
        out  = _norm_band(band)
        assert np.all(out == 0)

    def test_monotone_preserved(self):
        band = np.array([[1.0, 5.0, 10.0]], dtype=np.float32)
        out  = _norm_band(band, lo_pct=0, hi_pct=100)
        assert out[0, 0] < out[0, 1] < out[0, 2]

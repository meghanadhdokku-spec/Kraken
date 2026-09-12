"""Unit tests for yolo_detect — NMS and IoU helpers."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Import only the pure helper functions; avoid the module-level `from ultralytics import YOLO`
import importlib, types, unittest.mock as mock

_ult_stub = types.ModuleType("ultralytics")
_ult_stub.YOLO = object
sys.modules.setdefault("ultralytics", _ult_stub)

from src.detect.yolo_detect import _iou_xyxy, nms_yolo


def box(x1, y1, x2, y2, conf=0.9, cls=1):
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": conf, "cls": cls}


# ── _iou_xyxy ──────────────────────────────────────────────────────────────────

class TestIouXyxy:
    def test_identical_boxes_iou_one(self):
        b = box(0, 0, 10, 10)
        assert _iou_xyxy(b, b) == pytest.approx(1.0)

    def test_no_overlap_iou_zero(self):
        a = box(0, 0, 10, 10)
        b = box(20, 20, 30, 30)
        assert _iou_xyxy(a, b) == 0.0

    def test_touching_edges_iou_zero(self):
        a = box(0, 0, 10, 10)
        b = box(10, 0, 20, 10)
        assert _iou_xyxy(a, b) == 0.0

    def test_half_overlap(self):
        a = box(0, 0, 10, 10)   # area 100
        b = box(5, 0, 15, 10)   # area 100, overlap 50
        iou = _iou_xyxy(a, b)
        assert iou == pytest.approx(50 / 150)

    def test_contained_box(self):
        outer = box(0, 0, 10, 10)   # area 100
        inner = box(2, 2,  8,  8)   # area 36, fully inside
        iou = _iou_xyxy(outer, inner)
        assert iou == pytest.approx(36 / 100)

    def test_symmetric(self):
        a = box(0, 0, 6, 8)
        b = box(3, 4, 9, 10)
        assert _iou_xyxy(a, b) == pytest.approx(_iou_xyxy(b, a))


# ── nms_yolo ───────────────────────────────────────────────────────────────────

class TestNmsYolo:
    def test_empty_input(self):
        assert nms_yolo([]) == []

    def test_single_box_kept(self):
        result = nms_yolo([box(0, 0, 10, 10)])
        assert len(result) == 1

    def test_non_overlapping_all_kept(self):
        boxes = [
            box(0,   0,  10, 10, conf=0.9),
            box(20, 20,  30, 30, conf=0.8),
            box(40, 40,  50, 50, conf=0.7),
        ]
        result = nms_yolo(boxes, iou_threshold=0.5)
        assert len(result) == 3

    def test_identical_boxes_only_highest_conf_kept(self):
        boxes = [
            box(0, 0, 10, 10, conf=0.5),
            box(0, 0, 10, 10, conf=0.9),
            box(0, 0, 10, 10, conf=0.7),
        ]
        result = nms_yolo(boxes, iou_threshold=0.5)
        assert len(result) == 1
        assert result[0]["conf"] == pytest.approx(0.9)

    def test_keeps_highest_conf_not_first(self):
        # Low-conf box listed first — NMS must still keep the high-conf one
        boxes = [
            box(0, 0, 10, 10, conf=0.3),
            box(1, 1,  9,  9, conf=0.95),
        ]
        result = nms_yolo(boxes, iou_threshold=0.3)
        assert len(result) == 1
        assert result[0]["conf"] == pytest.approx(0.95)

    def test_overlap_below_threshold_both_kept(self):
        # ~0.14 IoU — below default 0.5
        boxes = [
            box(0, 0, 10, 10, conf=0.9),
            box(7, 0, 17, 10, conf=0.8),
        ]
        result = nms_yolo(boxes, iou_threshold=0.5)
        assert len(result) == 2

    def test_overlap_above_threshold_lower_suppressed(self):
        # boxes overlap heavily (IoU > 0.5)
        boxes = [
            box(0, 0, 10, 10, conf=0.9),
            box(1, 1, 11, 11, conf=0.6),
        ]
        result = nms_yolo(boxes, iou_threshold=0.5)
        assert len(result) == 1
        assert result[0]["conf"] == pytest.approx(0.9)

    def test_two_separate_clusters(self):
        # cluster A: two overlapping boxes
        # cluster B: one isolated box
        boxes = [
            box(0,  0, 10, 10, conf=0.9),
            box(1,  1, 11, 11, conf=0.7),
            box(50, 50, 60, 60, conf=0.8),
        ]
        result = nms_yolo(boxes, iou_threshold=0.5)
        assert len(result) == 2
        confs = sorted(b["conf"] for b in result)
        assert confs == pytest.approx([0.8, 0.9])

    def test_custom_threshold_stricter(self):
        # With threshold=0.0, any overlap suppresses the lower box
        boxes = [
            box(0, 0, 10, 10, conf=0.9),
            box(9, 9, 19, 19, conf=0.8),  # tiny corner overlap
        ]
        result_strict  = nms_yolo(boxes, iou_threshold=0.0)
        result_lenient = nms_yolo(boxes, iou_threshold=0.5)
        assert len(result_strict)  == 1
        assert len(result_lenient) == 2

    def test_output_order_descending_conf(self):
        boxes = [
            box(0,  0, 10, 10, conf=0.5),
            box(20, 0, 30, 10, conf=0.9),
            box(40, 0, 50, 10, conf=0.7),
        ]
        result = nms_yolo(boxes, iou_threshold=0.5)
        confs = [b["conf"] for b in result]
        assert confs == sorted(confs, reverse=True)

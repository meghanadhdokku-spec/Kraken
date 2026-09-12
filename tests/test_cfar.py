"""
Unit tests for the CA-CFAR detector core.

All tests use synthetic data — no satellite imagery required.
Run with:  python -m pytest tests/test_cfar.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.detect.cfar_detect import (
    _alpha,
    cfar_detect,
    filter_detections,
    nms_detections,
    add_geo_coords,
)
from src.preprocess.sar_preprocess import dn_to_sigma0_db, lee_filter


# ── CFAR maths ─────────────────────────────────────────────────────────────────

class TestAlpha:
    def test_pfa_decreases_with_larger_alpha(self):
        """A larger PFA should yield a smaller α (lower threshold)."""
        a_tight  = _alpha(n_background=120, pfa=1e-8)
        a_loose  = _alpha(n_background=120, pfa=1e-4)
        assert a_tight > a_loose

    def test_alpha_positive(self):
        assert _alpha(120, 1e-6) > 0


class TestCFARDetect:
    """CA-CFAR should detect bright point targets embedded in uniform clutter."""

    @pytest.fixture
    def clutter_scene(self):
        rng = np.random.default_rng(42)
        img = rng.exponential(scale=1.0, size=(256, 256)).astype(np.float32)
        return img

    def test_detects_bright_target(self, clutter_scene):
        img = clutter_scene.copy()
        # plant a bright target 100× the background mean
        img[128, 128] = float(img.mean()) * 100
        mask = cfar_detect(img, guard=2, background=8, pfa=1e-6)
        # the planted pixel must be detected
        assert mask[128, 128] == 1

    def test_uniform_clutter_low_false_alarms(self, clutter_scene):
        mask = cfar_detect(clutter_scene, guard=2, background=8, pfa=1e-4)
        actual_pfa = mask.mean()
        # allow 10× headroom over the target PFA (image-edge effects)
        assert actual_pfa < 10 * 1e-4

    def test_output_dtype(self, clutter_scene):
        mask = cfar_detect(clutter_scene)
        assert mask.dtype == np.uint8
        assert set(np.unique(mask)).issubset({0, 1})

    def test_zero_image_no_detections(self):
        img  = np.zeros((64, 64), dtype=np.float32)
        mask = cfar_detect(img)
        assert mask.sum() == 0


# ── filter_detections ──────────────────────────────────────────────────────────

class TestFilterDetections:
    def _make_mask(self, positions: list[tuple[int, int]], shape=(128, 128)) -> np.ndarray:
        mask = np.zeros(shape, dtype=np.uint8)
        for r, c in positions:
            mask[r - 1:r + 2, c - 1:c + 2] = 1   # 3×3 blobs
        return mask

    def test_single_blob_returned(self):
        mask  = self._make_mask([(64, 64)])
        boxes = filter_detections(mask, min_area_px=1, max_area_px=9999)
        assert len(boxes) == 1

    def test_min_area_filters_noise(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[10, 10] = 1   # single-pixel blob (area = 1)
        mask[20:25, 20:25] = 1   # 5×5 blob (area = 25)
        boxes = filter_detections(mask, min_area_px=4, max_area_px=9999)
        assert all(b["area_px"] >= 4 for b in boxes)
        assert len(boxes) == 1

    def test_max_area_filters_land(self):
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[0:40, 0:40] = 1   # huge land blob
        mask[55:58, 55:58] = 1   # small vessel blob
        boxes = filter_detections(mask, min_area_px=4, max_area_px=500)
        assert all(b["area_px"] <= 500 for b in boxes)

    def test_two_separate_blobs(self):
        mask  = self._make_mask([(20, 20), (80, 80)])
        boxes = filter_detections(mask, min_area_px=1, max_area_px=9999)
        assert len(boxes) == 2

    def test_conf_none_without_linear_image(self):
        mask  = self._make_mask([(64, 64)])
        boxes = filter_detections(mask, min_area_px=1, max_area_px=9999)
        assert all(b["conf"] is None for b in boxes)

    def test_conf_positive_with_linear_image(self):
        mask  = self._make_mask([(64, 64)])
        linear = np.ones((128, 128), dtype=np.float32)
        boxes  = filter_detections(mask, min_area_px=1, max_area_px=9999, linear_image=linear)
        assert len(boxes) == 1
        assert boxes[0]["conf"] is not None
        assert boxes[0]["conf"] > 0

    def test_bright_target_conf_above_dim_background(self):
        # two blobs: one dim, one bright → bright blob conf > dim blob conf
        mask   = np.zeros((64, 64), dtype=np.uint8)
        mask[10:13, 10:13] = 1   # dim blob
        mask[40:43, 40:43] = 1   # bright blob
        linear = np.ones((64, 64), dtype=np.float32) * 0.1
        linear[10:13, 10:13] = 1.0
        linear[40:43, 40:43] = 100.0
        boxes  = filter_detections(mask, min_area_px=1, max_area_px=9999, linear_image=linear)
        confs  = {b["cx"]: b["conf"] for b in boxes}
        # bright blob centroid is around col 41, dim around col 11
        bright_conf = max(confs.values())
        dim_conf    = min(confs.values())
        assert bright_conf > dim_conf

    def test_empty_mask_returns_empty(self):
        mask  = np.zeros((64, 64), dtype=np.uint8)
        boxes = filter_detections(mask, min_area_px=1, max_area_px=9999)
        assert boxes == []


# ── nms_detections ─────────────────────────────────────────────────────────────

class TestNMS:
    def _box(self, x, y, w, h):
        return {"x": x, "y": y, "w": w, "h": h, "cx": x + w / 2,
                "cy": y + h / 2, "area_px": w * h}

    def test_non_overlapping_kept(self):
        boxes = [self._box(0, 0, 10, 10), self._box(50, 50, 10, 10)]
        assert len(nms_detections(boxes, iou_threshold=0.3)) == 2

    def test_heavily_overlapping_suppressed(self):
        boxes = [self._box(0, 0, 20, 20), self._box(1, 1, 20, 20)]
        kept  = nms_detections(boxes, iou_threshold=0.3)
        assert len(kept) == 1

    def test_empty_input(self):
        assert nms_detections([]) == []


# ── preprocessing unit tests ───────────────────────────────────────────────────

class TestDNToSigma0:
    def test_zero_dn_is_nan(self):
        dn = np.array([[0, 0], [0, 0]], dtype=np.uint16)
        out = dn_to_sigma0_db(dn)
        assert np.all(np.isnan(out))

    def test_non_zero_returns_finite(self):
        dn = np.array([[100, 200], [300, 400]], dtype=np.uint16)
        out = dn_to_sigma0_db(dn)
        assert np.all(np.isfinite(out))

    def test_monotone(self):
        dn  = np.array([100, 200, 400], dtype=np.float32)
        out = dn_to_sigma0_db(dn)
        assert np.all(np.diff(out) > 0)


class TestLeeFilter:
    def test_output_shape_preserved(self):
        img = np.random.default_rng(0).random((3, 64, 64)).astype(np.float32)
        out = lee_filter(img, kernel_size=7)
        assert out.shape == img.shape

    def test_smooths_speckle(self):
        rng = np.random.default_rng(1)
        img = rng.normal(loc=0.0, scale=5.0, size=(128, 128)).astype(np.float32)
        out = lee_filter(img, kernel_size=7)
        assert float(np.nanstd(out)) < float(np.nanstd(img))

    def test_nan_preserved(self):
        img = np.ones((32, 32), dtype=np.float32)
        img[0, 0] = np.nan
        out = lee_filter(img, kernel_size=3)
        assert np.isnan(out[0, 0])

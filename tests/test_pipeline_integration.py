"""
Integration tests for the Kraken pipeline using synthetic data.

No credentials, no downloads, no real SAR imagery required.
Tests exercise the full detect → AIS-match path end-to-end.

Run with:  python -m pytest tests/test_pipeline_integration.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.make_test_scene import make_synthetic_scene, VESSEL_POSITIONS
from src.detect.cfar_detect import detect_scene
from src.match.ais_match import run_ais_matching


# ── shared fixture ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def synthetic_scene(tmp_path_factory) -> Path:
    """Write one synthetic SAR scene shared across all integration tests."""
    out = tmp_path_factory.mktemp("scene") / "synthetic_processed.tif"
    make_synthetic_scene(out, size=256, n_vessels=10, seed=0)
    return out


@pytest.fixture(scope="module")
def detections(synthetic_scene, tmp_path_factory) -> list[dict]:
    """Run CFAR detection on the synthetic scene (no land mask for speed)."""
    out_dir = tmp_path_factory.mktemp("outputs")
    return detect_scene(
        synthetic_scene,
        pfa=1e-4,
        use_land_mask=False,
        out_dir=out_dir,
        dual_band=True,
    )


# ── scene generation ───────────────────────────────────────────────────────────

class TestSceneGeneration:
    def test_file_exists(self, synthetic_scene):
        assert synthetic_scene.exists()

    def test_two_bands(self, synthetic_scene):
        import rasterio
        with rasterio.open(synthetic_scene) as src:
            assert src.count == 2

    def test_epsg4326(self, synthetic_scene):
        import rasterio
        with rasterio.open(synthetic_scene) as src:
            assert src.crs.to_epsg() == 4326

    def test_finite_values_present(self, synthetic_scene):
        import rasterio
        with rasterio.open(synthetic_scene) as src:
            data = src.read()
        assert np.any(np.isfinite(data))


# ── CFAR detection ─────────────────────────────────────────────────────────────

class TestCFARIntegration:
    def test_detections_non_empty(self, detections):
        assert len(detections) > 0

    def test_detects_most_planted_targets(self, detections):
        # At pfa=1e-4 on a 256×256 scene we expect to recover most of 10 targets
        assert len(detections) >= 5

    def test_each_detection_has_lon_lat(self, detections):
        for d in detections:
            assert "lon" in d and "lat" in d
            assert -180 <= d["lon"] <= 180
            assert -90  <= d["lat"] <= 90

    def test_each_detection_has_dimensions(self, detections):
        for d in detections:
            assert "length_m" in d and "width_m" in d

    def test_each_detection_has_vessel_class(self, detections):
        valid_classes = {
            "small_craft", "fishing_vessel", "coastal_freighter",
            "cargo_ship", "vlcc_or_large",
        }
        for d in detections:
            assert d.get("vessel_class") in valid_classes

    def test_each_detection_has_conf(self, detections):
        for d in detections:
            assert d.get("conf") is not None
            assert d["conf"] > 0

    def test_csv_written(self, synthetic_scene, tmp_path):
        out_dir = tmp_path / "csv_check"
        detect_scene(synthetic_scene, pfa=1e-4, use_land_mask=False,
                     out_dir=out_dir, dual_band=True)
        stem = synthetic_scene.stem
        assert (out_dir / f"{stem}_cfar_detections.csv").exists()

    def test_geojson_written(self, synthetic_scene, tmp_path):
        out_dir = tmp_path / "geojson_check"
        detect_scene(synthetic_scene, pfa=1e-4, use_land_mask=False,
                     out_dir=out_dir, dual_band=True)
        stem = synthetic_scene.stem
        assert (out_dir / f"{stem}_cfar_detections.geojson").exists()


# ── AIS matching integration ───────────────────────────────────────────────────

class TestAISMatchingIntegration:
    @pytest.fixture
    def ais_csv(self, tmp_path) -> Path:
        import csv as _csv
        path = tmp_path / "ais.csv"
        with open(path, "w", newline="") as fh:
            writer = _csv.DictWriter(
                fh,
                fieldnames=["lat", "lon", "mmsi", "vessel_name", "flag", "timestamp"],
            )
            writer.writeheader()
            for i, (lon, lat) in enumerate(VESSEL_POSITIONS[:5]):
                writer.writerow({
                    "lat": lat, "lon": lon,
                    "mmsi": f"63600{i:04d}",
                    "vessel_name": f"MV Test-{i}",
                    "flag": "GH",
                    "timestamp": "2023-04-12T06:00:00",
                })
        return path

    def test_annotates_all_detections(self, detections, ais_csv):
        annotated = run_ais_matching(
            list(detections),  # copy — don't mutate the module-scoped fixture
            scene_path=Path(
                "S1A_IW_GRDH_1SDV_20230412T060000_20230412T060030_fake"
            ),
            radius_km=2.0,
            ais_csv_path=ais_csv,
        )
        assert len(annotated) == len(detections)
        for d in annotated:
            assert "ais_match" in d
            assert "dark_vessel" in d

    def test_some_matched_some_dark(self, detections, ais_csv):
        annotated = run_ais_matching(
            list(detections),
            scene_path=Path(
                "S1A_IW_GRDH_1SDV_20230412T060000_20230412T060030_fake"
            ),
            radius_km=2.0,
            ais_csv_path=ais_csv,
        )
        n_dark    = sum(1 for d in annotated if d["dark_vessel"])
        n_matched = sum(1 for d in annotated if d["ais_match"])
        # 5 AIS vessels planted at exactly the first 5 targets → expect matches
        assert n_matched >= 1
        # remaining targets are dark
        assert n_dark >= 1

    def test_no_ais_all_dark(self, detections):
        annotated = run_ais_matching(
            list(detections),
            scene_path=Path(
                "S1A_IW_GRDH_1SDV_20230412T060000_20230412T060030_fake"
            ),
            radius_km=1.0,
            ais_csv_path=None,
        )
        assert all(d["dark_vessel"] for d in annotated)

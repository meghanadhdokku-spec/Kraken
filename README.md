# KRAKEN — SAR Vessel / IUU Fishing Detector

Detect AIS-dark vessels and IUU fishing activity in open ocean using Sentinel-1 SAR imagery — two detectors, AIS cross-matching, GeoJSON output.

**CA-CFAR + YOLOv8 · 163 tests · dual-band detection · xView3-SAR · Gulf of Guinea**

---

## What is Kraken

Synthetic aperture radar sees through cloud and darkness. Vessels that disable their AIS transponders — fishing illegally, smuggling, evading sanctions — still cast a radar return. Kraken finds them, cross-references live AIS, and flags the ones that shouldn't be there.

| Component | What it does |
|---|---|
| **Sentinel-1 GRD** | Free, global coverage at ~10 m/px. VV + VH dual-polarisation. Downloaded from Copernicus Dataspace. |
| **CA-CFAR** | Constant false-alarm rate detector. No training data needed. Threshold `T = α · bg_mean` adapts to local clutter. Dual-band VV∨VH fusion. |
| **YOLOv8** | Fine-tuned on xView3-SAR with SAR-specific augmentation (no HSV, 90° rotations, mosaic). |
| **AIS matching** | Cross-references detections against AIS data (Global Fishing Watch or local CSV). Unmatched vessels flagged as **dark**. |
| **GeoJSON output** | Every detection carries lon/lat, class, confidence, bounding box, AIS match status. |

---

## Pipeline

```
Download → Preprocess → CA-CFAR / YOLOv8 → AIS Match → GeoJSON + Visualise
  (01)         (02)         (03a / 03b)         (04)            (05)
```

| Step | Module | Key operations |
|---|---|---|
| 01 Download | `src/download/sentinel_download.py` | Copernicus Dataspace OAuth2 → query → download |
| 02 Preprocess | `src/preprocess/sar_preprocess.py` | DN → σ⁰ dB · Lee speckle filter · WGS84 reproject · bbox clip |
| 03a CFAR | `src/detect/cfar_detect.py` | Water mask · area filter · IoU NMS · size classification · GeoJSON |
| 03b YOLO | `src/detect/yolo_detect.py` | Tiled inference · stitch · post-NMS · GeoJSON |
| 04 AIS match | `src/match/ais_match.py` | Haversine radius match · dark vessel flag · GFW API or local CSV |
| 05 Viz | `src/utils/viz.py` | Scene overview + 64 px detection crop grid |

---

## Detection Classes

### Size-based (CFAR output)

| `vessel_class` | Approximate length | Notes |
|---|---|---|
| `small_craft` | < 20 m | Pirogue, skiff, small fishing boat |
| `fishing_vessel` | 20–50 m | Industrial fishing vessel — **primary IUU target** |
| `coastal_freighter` | 50–100 m | Regional cargo / supply ship |
| `cargo_ship` | 100–200 m | Bulk carrier, container |
| `vlcc_or_large` | > 200 m | VLCC, large tanker |

### YOLO label classes (xView3)

| Class | Name | Description |
|---|---|---|
| `0` | `non_vessel` | Flotsam, ambiguous returns, sea clutter |
| `1` | `vessel` | Confirmed ship; fishing status unknown or negative |
| `2` | `fishing_vessel` | `is_vessel=True, is_fishing=True` in xView3 |

---

## Quick Start

### 1. Install

**Conda (recommended — resolves GDAL/rasterio natively)**

```bash
conda env create -f environment.yml
conda activate kraken
```

**pip only**

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure credentials

```bash
cp .env.example .env
```

Edit `.env`:

```dotenv
# Copernicus Dataspace — free account at dataspace.copernicus.eu
COPERNICUS_USER=your@email.com
COPERNICUS_PASSWORD=your_password

# Global Fishing Watch API (optional — for live AIS matching)
# Free token at globalfishingwatch.org/data-download/
GFW_API_TOKEN=your_token_here
```

### 3. Test without credentials (smoke test)

No account needed. Generates a synthetic SAR scene with 10 planted vessel targets and runs the full detection + AIS matching pipeline.

```bash
python scripts/smoke_test.py
```

Expected output:

```
[STEP 1] Generating synthetic SAR scene...
[OK]  Synthetic scene written → data/outputs/smoke_test/synthetic_scene_processed.tif

[STEP 2] Running CFAR detector...
[STEP 3] Running AIS matching...

==========================================================
  SMOKE TEST SUMMARY
==========================================================
  Detections:        10
  AIS-matched:       5
  Dark (no AIS):     5
==========================================================

  PASS — pipeline detected expected vessel targets
```

### 4. Download real scenes

```bash
# list available scenes (no download)
python -m src.download.sentinel_download --query-only

# download up to 5 scenes over Gulf of Guinea
python -m src.download.sentinel_download --limit 5
```

| Flag | Default | Description |
|---|---|---|
| `--start` | 30 days ago | Start date (YYYYMMDD) |
| `--end` | today | End date |
| `--bbox W S E N` | Gulf of Guinea | Custom AOI |
| `--limit N` | 5 | Max scenes to download |
| `--query-only` | — | List only, no download |

### 5. Run the pipeline

```bash
# CFAR only (no trained weights needed)
python -m src.pipeline \
    --skip-download --skip-preprocess \
    --scene data/processed/scene_processed.tif \
    --detector cfar

# Both detectors + AIS matching + save PNGs
python -m src.pipeline \
    --skip-download --skip-preprocess \
    --scene data/processed/scene_processed.tif \
    --detector both \
    --weights models/yolo_vessel/weights/best.pt \
    --viz
```

### 6. Train on xView3-SAR

Download chips + labels from [iuu.xview.us](https://iuu.xview.us), place under `data/annotations/xview3/`.

```bash
# prepare dataset (chip → PNG + YOLO labels)
python -m src.train.xview3_prep \
    --xview3-dir data/annotations/xview3

# fine-tune YOLOv8n
python -m src.train.train_yolo \
    --data data/annotations/xview3/dataset.yaml \
    --epochs 50
```

---

## Project Layout

```
Kraken/
├── config/
│   └── settings.py               # paths, AOI, CFAR + YOLO params, credentials
├── data/
│   ├── raw/                      # Sentinel-1 SAFE archives
│   ├── processed/                # σ⁰ dB GeoTIFFs (Lee-filtered)
│   └── annotations/xview3/      # xView3 chips + labels
├── scripts/
│   ├── make_test_scene.py        # generate synthetic SAR test scene
│   └── smoke_test.py             # end-to-end test without credentials
├── src/
│   ├── download/
│   │   └── sentinel_download.py  # step 1 — Copernicus OAuth2 API
│   ├── preprocess/
│   │   └── sar_preprocess.py     # step 2 — calibrate + Lee filter + clip
│   ├── detect/
│   │   ├── cfar_detect.py        # step 3a — dual-band CA-CFAR
│   │   └── yolo_detect.py        # step 3b — YOLOv8 tiled inference
│   ├── match/
│   │   └── ais_match.py          # step 4 — AIS cross-match, dark vessel flag
│   ├── train/
│   │   ├── xview3_prep.py        # scene→chip coordinate mapping
│   │   ├── train_yolo.py         # SAR-tuned fine-tuning
│   │   └── evaluate.py           # P/R/F1/mAP + IUU F-beta
│   ├── utils/
│   │   ├── geo_utils.py          # GeoJSON helpers, size classification
│   │   └── viz.py                # overview + crop grid
│   └── pipeline.py               # end-to-end orchestrator
├── models/                       # saved YOLO weights
├── outputs/                      # masks, CSV, GeoJSON, PNGs
├── notebooks/
│   └── 01_explore.ipynb          # walkthrough notebook
└── tests/                        # 163 tests, all passing
```

---

## Tests

```bash
# all tests
python -m pytest tests/ -v

# unit tests only (fast, no file I/O)
python -m pytest tests/ -v -k "not Integration"

# integration tests only (writes synthetic GeoTIFFs)
python -m pytest tests/test_pipeline_integration.py -v
```

| Test file | Count | Covers |
|---|---|---|
| `test_cfar.py` | 19 | alpha formula, CFAR detection, NMS, confidence score, Lee filter, calibration |
| `test_preprocess.py` | 14 | clip_to_bbox, write_geotiff, DN→σ⁰, Lee filter regression |
| `test_ais_match.py` | 21 | haversine, ISO parsing, scene time window, CSV fetch, detection matching |
| `test_xview3_prep.py` | 40 | CSV parsing, coord mapping, class assignment, label writing |
| `test_yolo_detect.py` | 16 | IoU, NMS — 6 `_iou_xyxy` + 10 `nms_yolo` cases |
| `test_geo_utils.py` | 38 | GeoJSON build, save, merge, class sizing, FeatureCollection I/O |
| `test_pipeline_integration.py` | 15 | full detect→AIS-match path on synthetic scene, no credentials needed |
| **Total** | **163** | **all passing** |

---

## Output Files

For each processed scene (`<stem>` = scene filename without extension):

| File | Format | Description |
|---|---|---|
| `<stem>_cfar_detections.csv` | CSV | One row per detection: lon, lat, length_m, width_m, vessel_class, conf, dark_vessel, ais_match |
| `<stem>_cfar_detections.geojson` | GeoJSON | Same detections as a FeatureCollection (Point geometry) |
| `<stem>_cfar_mask.tif` | GeoTIFF | Binary detection mask aligned to input scene |
| `<stem>_overview.png` | PNG | Scene overview with detection boxes overlaid |
| `<stem>_crops.png` | PNG | 64 px crop grid, one tile per detection |

---

## Area of Interest

Default AOI: **−5°W to 10°E · −5°S to 5°N** (Gulf of Guinea)

Target waters: Nigeria · Ghana · Ivory Coast · Cameroon
Primary concern: IUU fishing + AIS-dark cargo vessels

Change in `config/settings.py` → `GOG_BBOX` and `GOG_WKT`.

---

## Acknowledgements

- [Copernicus Dataspace](https://dataspace.copernicus.eu) — Sentinel-1 imagery
- [xView3-SAR](https://iuu.xview.us) — vessel detection training dataset
- [Global Fishing Watch](https://globalfishingwatch.org) — AIS event API
- [Natural Earth](https://www.naturalearthdata.com) — land polygon data (via `geodatasets`)

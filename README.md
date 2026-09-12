# KRAKEN v0 — SAR Vessel / IUU Fishing Detector

Detect AIS-dark vessels and IUU fishing activity in open ocean using Sentinel-1 SAR imagery — two detectors, one pipeline, GeoJSON output.

**CA-CFAR + YOLOv8 · 113 tests · 3-class detection · xView3-SAR**

---

## What is Kraken

Synthetic aperture radar sees through cloud and darkness. Vessels that disable their AIS transponders — fishing illegally, smuggling, evading sanctions — still cast a radar return. Kraken finds them.

| Component | What it does |
|---|---|
| **Sentinel-1 GRD** | Free, global coverage at ~10 m/px. VV + VH dual-polarisation. Downloads from Copernicus Dataspace. |
| **CA-CFAR** | Classical constant false-alarm rate detector. No training data needed. Threshold `T = α · bg_mean` adapts to local clutter. |
| **YOLOv8** | Fine-tuned on xView3-SAR with SAR-specific augmentation (no HSV, 90° rotations, mosaic). |
| **GeoJSON output** | Every detection carries lon/lat, class, confidence, bounding box. CFAR + YOLO results merge into one FeatureCollection. |

---

## Pipeline

```
Download → Preprocess → CA-CFAR / YOLOv8 → GeoJSON + Visualise
  (01)         (02)         (03a / 03b)            (04)
```

| Step | Module | Key operations |
|---|---|---|
| 01 Download | `src/download/sentinel_download.py` | Copernicus Dataspace query + download |
| 02 Preprocess | `src/preprocess/sar_preprocess.py` | DN → σ⁰ dB · Lee speckle filter · WGS84 reproject · bbox clip |
| 03a CFAR | `src/detect/cfar_detect.py` | Water mask · area filter · IoU NMS · GeoJSON |
| 03b YOLO | `src/detect/yolo_detect.py` | Tiled inference · stitch · post-NMS · GeoJSON |
| 04 Viz | `src/utils/viz.py` | Scene overview + 64 px detection crop grid |

---

## Detection Classes

| Class | Name | Description |
|---|---|---|
| `0` | `non_vessel` | Flotsam, ambiguous returns, sea clutter |
| `1` | `vessel` | Confirmed ship; fishing status unknown or negative |
| `2` | `fishing_vessel` | **Primary IUU target** — `is_vessel=True, is_fishing=True` in xView3 |

---

## Quick Start

### 1. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Set credentials

Register free at [dataspace.copernicus.eu](https://dataspace.copernicus.eu).

```bash
cp .env.example .env
# add COPERNICUS_USER + COPERNICUS_PASSWORD
```

### 3. Download scenes

```bash
# preview available scenes (no download)
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

### 4. Run the pipeline

```bash
# CFAR only (no trained weights needed)
python -m src.pipeline \
    --skip-download --skip-preprocess \
    --scene data/processed/scene_processed.tif \
    --detector cfar

# Both detectors + save PNGs
python -m src.pipeline \
    --skip-download --skip-preprocess \
    --scene data/processed/scene_processed.tif \
    --detector both \
    --weights models/yolo_vessel/weights/best.pt \
    --viz
```

### 5. Train on xView3-SAR

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
│   └── settings.py               # paths, AOI, CFAR + YOLO params
├── data/
│   ├── raw/                      # Sentinel-1 SAFE archives
│   ├── processed/                # σ⁰ dB GeoTIFFs (Lee-filtered)
│   └── annotations/xview3/      # xView3 chips + labels
├── src/
│   ├── download/
│   │   └── sentinel_download.py  # step 1 — Copernicus API
│   ├── preprocess/
│   │   └── sar_preprocess.py     # step 2 — calibrate + filter
│   ├── detect/
│   │   ├── cfar_detect.py        # step 3a — CA-CFAR
│   │   └── yolo_detect.py        # step 3b — YOLOv8 tiled
│   ├── train/
│   │   ├── xview3_prep.py        # scene→chip coordinate mapping
│   │   ├── train_yolo.py         # SAR-tuned fine-tuning
│   │   └── evaluate.py           # P/R/F1/mAP + IUU F-beta
│   ├── utils/
│   │   ├── geo_utils.py          # rasterio + GeoJSON helpers
│   │   └── viz.py                # overview + crop grid
│   └── pipeline.py               # end-to-end orchestrator
├── models/                       # saved weights
├── outputs/                      # masks, CSV, GeoJSON, PNGs
├── notebooks/
│   └── 01_explore.ipynb          # walkthrough notebook
└── tests/                        # 113 tests
```

---

## Tests

```bash
python -m pytest tests/ -q
```

| Test file | Count | Covers |
|---|---|---|
| `test_cfar.py` | 19 | alpha formula, detection, NMS, calibration, Lee filter |
| `test_xview3_prep.py` | 40 | CSV parsing, coord mapping, class assignment, label writing |
| `test_yolo_detect.py` | 16 | NMS, IoU — 6 `_iou_xyxy` + 10 `nms_yolo` cases |
| `test_geo_utils.py` | 38 | GeoJSON build, save, merge, pixel→lonlat, GeoTIFF I/O |
| **Total** | **113** | **all passing** |

---

## Area of Interest

Default AOI: **−5°W to 10°E · −5°S to 5°N** (Gulf of Guinea)

Target waters: Nigeria · Ghana · Ivory Coast · Cameroon
Primary concern: IUU fishing + dark cargo vessels

Configurable in `config/settings.py` → `GOG_BBOX`.

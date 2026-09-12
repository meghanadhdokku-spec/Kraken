# Kraken v0 — SAR Vessel / IUU Fishing Detector

Detect dark (AIS-dark) vessels in the Gulf of Guinea using Sentinel-1 GRD imagery and YOLOv8.

## Project layout

```
Kraken/
├── config/
│   └── settings.py           # Central config — paths, AOI, model params
├── data/
│   ├── raw/                  # Downloaded Sentinel-1 SAFE archives
│   ├── processed/            # Preprocessed GeoTIFFs (σ⁰ dB, Lee-filtered)
│   └── annotations/          # xView3-SAR labels
├── src/
│   ├── download/
│   │   └── sentinel_download.py   # Step 1 — download scenes from Copernicus
│   ├── preprocess/
│   │   └── sar_preprocess.py      # Step 2 — calibrate, clip, speckle-filter
│   ├── detect/
│   │   ├── cfar_detect.py         # Step 3a — CA-CFAR classical detector
│   │   └── yolo_detect.py         # Step 3b — YOLOv8 deep-learning detector
│   ├── train/
│   │   └── train_yolo.py          # Fine-tune YOLOv8 on xView3-SAR
│   └── utils/
│       ├── geo_utils.py           # Rasterio helpers
│       └── viz.py                 # Visualisation
├── models/                   # Saved model weights
├── outputs/                  # Detection masks, CSV results
└── notebooks/                # Exploration notebooks
```

## Quick start

### 1. Install dependencies

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Set Copernicus credentials

```bash
cp .env.example .env
# edit .env — add your Copernicus Dataspace username + password
```

Register free at <https://dataspace.copernicus.eu/>.

### 3. Download Sentinel-1 scenes

```bash
# List last 30 days of scenes over Gulf of Guinea (no download)
python -m src.download.sentinel_download --query-only

# Download up to 5 scenes
python -m src.download.sentinel_download --limit 5
```

Options:

| Flag | Default | Description |
|---|---|---|
| `--start` | 30 days ago | Start date (YYYYMMDD) |
| `--end` | today | End date (YYYYMMDD) |
| `--bbox W S E N` | Gulf of Guinea | Custom AOI |
| `--limit N` | 5 | Max scenes |
| `--query-only` | — | List only, no download |
| `--out-dir PATH` | `data/raw/` | Download destination |

### 4. Preprocess

```bash
python -m src.preprocess.sar_preprocess \
    --scene data/raw/S1A_IW_GRDH_1SDV_...SAFE.zip
```

Outputs a float32 GeoTIFF (VV + VH σ⁰ dB, Lee-filtered) in `data/processed/`.

### 5a. Classical detection (CA-CFAR)

```bash
python -m src.detect.cfar_detect \
    --scene data/processed/scene_processed.tif
```

### 5b. Deep-learning detection (YOLOv8)

```bash
python -m src.detect.yolo_detect \
    --scene data/processed/scene_processed.tif \
    --weights models/best.pt
```

### 6. Train on xView3-SAR

```bash
python -m src.train.train_yolo \
    --data-dir data/annotations/xview3 \
    --epochs 50
```

## Dataset

[xView3-SAR](https://iuu.xview.us/) — global ship detection benchmark on Sentinel-1.  
Download chips and labels from the xView3 portal and place under `data/annotations/xview3/`.

## Area of Interest

Gulf of Guinea bounding box: **-5°W to 10°E, -5°S to 5°N**  
Configurable in `config/settings.py`.

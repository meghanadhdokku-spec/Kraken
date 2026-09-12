#!/usr/bin/env bash
# Kraken one-command setup for Linux / macOS
set -e

if command -v conda &>/dev/null; then
    echo "[1/3] Creating conda environment from environment.yml ..."
    conda env create -f environment.yml 2>/dev/null || conda env update -f environment.yml --prune
    echo "[2/3] Activate with: conda activate kraken"
else
    echo "[1/3] conda not found — installing via pip ..."
    pip install \
        rasterio shapely pyproj geopandas \
        numpy scipy opencv-python pillow \
        matplotlib python-dotenv tqdm pyyaml requests \
        ultralytics
fi

echo "[3/3] Verifying key imports ..."
python -c "import rasterio, scipy, numpy, cv2, ultralytics, matplotlib, dotenv; print('[OK] All imports successful')"

echo ""
echo "Setup complete. Next steps:"
echo "  1. cp .env.example .env   # add Copernicus credentials"
echo "  2. python -m src.download.sentinel_download --query-only --limit 5"

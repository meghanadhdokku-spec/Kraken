@echo off
REM Kraken one-command setup for Windows (requires miniforge / conda)

echo [1/3] Creating conda environment from environment.yml ...
conda env create -f environment.yml
IF ERRORLEVEL 1 (
    echo.
    echo Environment already exists. Updating instead ...
    conda env update -f environment.yml --prune
)

echo.
echo [2/3] Activating environment ...
call conda activate kraken

echo.
echo [3/3] Verifying key imports ...
python -c "import rasterio, scipy, numpy, cv2, ultralytics, matplotlib, dotenv; print('[OK] All imports successful')"

echo.
echo Setup complete. Next steps:
echo   1. copy .env.example .env
echo      ^(then edit .env with your Copernicus credentials^)
echo   2. conda activate kraken
echo   3. python -m src.download.sentinel_download --query-only --limit 5

"""
SAR preprocessing pipeline.

Steps applied per Sentinel-1 SAFE scene:
  1. Read VV and VH bands from the SAFE zip (rasterio)
  2. Convert DN → sigma-naught (linear, then dB)
  3. Clip to AOI bounding box (optional)
  4. Apply Lee speckle filter (7×7 kernel)
  5. Write float32 GeoTIFF to processed/

Usage:
    python -m src.preprocess.sar_preprocess --scene data/raw/S1A_IW_GRD_...SAFE.zip
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
from scipy.ndimage import uniform_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import PROCESSED_DIR, GOG_BBOX


# ── signal maths ───────────────────────────────────────────────────────────────

def dn_to_sigma0_db(dn: np.ndarray, nodata: float = 0.0) -> np.ndarray:
    """Convert GRD integer DN to σ⁰ in dB.  Nodata pixels become NaN."""
    arr = dn.astype(np.float32)
    valid = arr != nodata
    sigma_linear = np.where(valid, arr ** 2, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_db = 10.0 * np.log10(sigma_linear)
    return sigma_db


def lee_filter(img: np.ndarray, kernel_size: int = 7) -> np.ndarray:
    """
    Single-look Lee speckle filter.
    Works band-by-band on a (bands, rows, cols) or (rows, cols) array.
    """
    def _filter_2d(band: np.ndarray) -> np.ndarray:
        mask = np.isnan(band)
        band_filled = np.where(mask, 0.0, band)
        mean  = uniform_filter(band_filled, kernel_size)
        mean2 = uniform_filter(band_filled ** 2, kernel_size)
        var   = mean2 - mean ** 2
        noise_var = np.nanmean(var)
        weight = var / (var + noise_var + 1e-10)
        filtered = mean + weight * (band_filled - mean)
        return np.where(mask, np.nan, filtered)

    if img.ndim == 2:
        return _filter_2d(img)
    return np.stack([_filter_2d(img[b]) for b in range(img.shape[0])])


# ── I/O ────────────────────────────────────────────────────────────────────────

def read_grd_bands(scene_path: Path) -> tuple[np.ndarray, dict]:
    """
    Open a Sentinel-1 GRD SAFE zip and read the VV + VH bands.

    Returns:
        data  – float32 ndarray (2, rows, cols) [VV, VH] in dB
        meta  – rasterio profile dict
    """
    with rasterio.open(scene_path) as src:
        if src.count < 2:
            raise ValueError(f"Expected ≥2 bands in {scene_path}, got {src.count}")
        vv = src.read(1)
        vh = src.read(2)
        meta = src.profile.copy()

    vv_db = dn_to_sigma0_db(vv)
    vh_db = dn_to_sigma0_db(vh)

    data = np.stack([vv_db, vh_db], axis=0)
    meta.update(count=2, dtype="float32", nodata=np.nan)
    return data, meta


def clip_to_bbox(
    data: np.ndarray,
    meta: dict,
    bbox: tuple[float, float, float, float],
) -> tuple[np.ndarray, dict]:
    """Clip raster to a (west, south, east, north) bounding box."""
    from rasterio.transform import array_bounds
    from rasterio.windows import from_bounds as fb

    transform = meta["transform"]
    crs       = meta["crs"]
    height, width = data.shape[-2], data.shape[-1]

    window = fb(bbox[0], bbox[1], bbox[2], bbox[3], transform=transform)
    window = window.crop(height, width)

    col_off = int(window.col_off)
    row_off = int(window.row_off)
    col_end = col_off + int(window.width)
    row_end = row_off + int(window.height)

    clipped = data[..., row_off:row_end, col_off:col_end]
    new_transform = rasterio.transform.from_bounds(
        bbox[0], bbox[1], bbox[2], bbox[3],
        clipped.shape[-1], clipped.shape[-2],
    )
    meta = meta.copy()
    meta.update(
        width=clipped.shape[-1],
        height=clipped.shape[-2],
        transform=new_transform,
    )
    return clipped, meta


def write_geotiff(data: np.ndarray, meta: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **meta) as dst:
        if data.ndim == 2:
            dst.write(data, 1)
        else:
            for b in range(data.shape[0]):
                dst.write(data[b], b + 1)
    print(f"[OK]   Written → {out_path}")


# ── pipeline ───────────────────────────────────────────────────────────────────

def preprocess_scene(
    scene_path: Path,
    out_dir: Path = PROCESSED_DIR,
    clip_bbox: tuple | None = GOG_BBOX,
    speckle_kernel: int = 7,
) -> Path:
    print(f"[INFO] Preprocessing {scene_path.name} …")

    data, meta = read_grd_bands(scene_path)
    print(f"  Bands: VV + VH  |  Shape: {data.shape}")

    if clip_bbox:
        data, meta = clip_to_bbox(data, meta, clip_bbox)
        print(f"  Clipped to bbox: {data.shape}")

    data = lee_filter(data, kernel_size=speckle_kernel)
    print(f"  Speckle-filtered (kernel={speckle_kernel})")

    stem = scene_path.stem.split(".")[0]
    out_path = out_dir / f"{stem}_processed.tif"
    write_geotiff(data, meta, out_path)
    return out_path


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Preprocess a Sentinel-1 GRD scene.")
    parser.add_argument("--scene", type=Path, required=True, help="Path to SAFE zip or .tif")
    parser.add_argument("--out-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--no-clip", action="store_true", help="Skip AOI clipping")
    parser.add_argument("--kernel", type=int, default=7, help="Lee filter kernel size")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    clip = None if args.no_clip else GOG_BBOX
    preprocess_scene(args.scene, args.out_dir, clip, args.kernel)


if __name__ == "__main__":
    main()

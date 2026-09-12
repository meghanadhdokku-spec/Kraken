"""
SAR preprocessing pipeline for Sentinel-1 GRD scenes.

Steps applied per SAFE scene (zip or directory):
  1. Locate VV + VH measurement GeoTIFFs inside the SAFE archive
  2. Convert DN → σ⁰ linear → dB  (GRD calibration)
  3. Reproject to EPSG:4326 if not already geographic
  4. Clip to AOI bounding box (optional)
  5. Apply Lee speckle filter
  6. Write 2-band float32 GeoTIFF → data/processed/

Usage:
    python -m src.preprocess.sar_preprocess \
        --scene data/raw/S1A_IW_GRDH_1SDV_...SAFE.zip
"""

import argparse
import sys
import zipfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from scipy.ndimage import uniform_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import PROCESSED_DIR, GOG_BBOX

TARGET_CRS = CRS.from_epsg(4326)


# ── SAFE archive navigation ────────────────────────────────────────────────────

def find_safe_measurements(safe_path: Path) -> dict[str, str]:
    """
    Locate VV and VH measurement GeoTIFFs inside a SAFE zip or directory.

    Returns a dict like {'VV': '<rasterio-openable path>', 'VH': '...'}.
    Paths for zip files use GDAL's /vsizip/ virtual filesystem.
    Raises FileNotFoundError if both polarisations are not found.
    """
    pols: dict[str, str] = {}

    if safe_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(safe_path) as zf:
            names = zf.namelist()
        tiffs = [n for n in names if "/measurement/" in n and n.endswith(".tiff")]
        for entry in tiffs:
            lower = entry.lower()
            if "-vv-" in lower:
                pols["VV"] = f"/vsizip/{safe_path}/{entry}"
            elif "-vh-" in lower:
                pols["VH"] = f"/vsizip/{safe_path}/{entry}"
    else:
        # Unpacked .SAFE directory
        meas_dir = safe_path / "measurement"
        if not meas_dir.is_dir():
            raise FileNotFoundError(f"No measurement/ directory in {safe_path}")
        for tiff in meas_dir.glob("*.tiff"):
            lower = tiff.name.lower()
            if "-vv-" in lower:
                pols["VV"] = str(tiff)
            elif "-vh-" in lower:
                pols["VH"] = str(tiff)

    missing = [p for p in ("VV", "VH") if p not in pols]
    if missing:
        raise FileNotFoundError(
            f"Could not find {missing} measurement tiffs in {safe_path}"
        )
    return pols


# ── radiometric calibration ────────────────────────────────────────────────────

def dn_to_sigma0_db(dn: np.ndarray, nodata: float = 0.0) -> np.ndarray:
    """
    GRD DN → σ⁰ in dB.

    For Sentinel-1 GRD, σ⁰ (linear) = DN².
    Nodata (DN == 0) → NaN.  Very small values → NaN to avoid -inf.
    """
    arr = dn.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_lin = np.where(arr > 0, arr ** 2, np.nan)
        sigma_db  = 10.0 * np.log10(sigma_lin)
    return sigma_db.astype(np.float32)


# ── speckle filter ─────────────────────────────────────────────────────────────

def lee_filter(img: np.ndarray, kernel_size: int = 7) -> np.ndarray:
    """
    Single-look Lee speckle filter operating in dB domain.
    Handles NaN (no-data) pixels by masking before filtering.
    Works on (bands, H, W) or (H, W).
    """
    def _filter_2d(band: np.ndarray) -> np.ndarray:
        nan_mask   = np.isnan(band)
        filled     = np.where(nan_mask, 0.0, band)
        local_mean = uniform_filter(filled, kernel_size, mode="reflect")
        local_sq   = uniform_filter(filled ** 2, kernel_size, mode="reflect")
        local_var  = np.maximum(local_sq - local_mean ** 2, 0.0)
        noise_var  = float(np.nanmean(local_var))
        weight     = local_var / (local_var + noise_var + 1e-10)
        filtered   = local_mean + weight * (filled - local_mean)
        return np.where(nan_mask, np.nan, filtered).astype(np.float32)

    if img.ndim == 2:
        return _filter_2d(img)
    return np.stack([_filter_2d(img[b]) for b in range(img.shape[0])])


# ── reprojection ───────────────────────────────────────────────────────────────

def reproject_to_wgs84(
    data: np.ndarray,
    meta: dict,
) -> tuple[np.ndarray, dict]:
    """
    Reproject (bands, H, W) data to EPSG:4326 if it is not already geographic.
    Returns reprojected data and updated profile.
    """
    src_crs = meta.get("crs")
    if src_crs and CRS(src_crs).to_epsg() == 4326:
        return data, meta

    n_bands, src_h, src_w = data.shape
    src_transform = meta["transform"]

    left, bottom, right, top = rasterio.transform.array_bounds(
        src_h, src_w, src_transform
    )
    dst_transform, dst_w, dst_h = calculate_default_transform(
        src_crs, TARGET_CRS, src_w, src_h,
        left=left, bottom=bottom, right=right, top=top,
    )

    out = np.full((n_bands, dst_h, dst_w), np.nan, dtype=np.float32)

    for b in range(n_bands):
        reproject(
            source=data[b],
            destination=out[b],
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=TARGET_CRS,
            resampling=Resampling.bilinear,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )

    new_meta = meta.copy()
    new_meta.update(
        crs=TARGET_CRS,
        transform=dst_transform,
        width=dst_w,
        height=dst_h,
    )
    return out, new_meta


# ── AOI clipping ───────────────────────────────────────────────────────────────

def clip_to_bbox(
    data: np.ndarray,
    meta: dict,
    bbox: tuple[float, float, float, float],
) -> tuple[np.ndarray, dict]:
    """
    Clip to (west, south, east, north) bounding box in the dataset's CRS.
    Intersects bbox with actual scene extent before clipping so the window
    is always valid even when bbox partially or fully falls outside the scene.
    Returns the clipped array and updated profile.
    """
    transform = meta["transform"]
    H, W = data.shape[-2], data.shape[-1]

    # array_bounds returns (west, south, east, north) but for south-up rasters
    # (transform.e > 0, common with GCP-derived Sentinel-1 TIFFs) the second and
    # fourth values are swapped geographically. Normalise before intersecting.
    raw = rasterio.transform.array_bounds(H, W, transform)
    s_left  = min(raw[0], raw[2])
    s_right = max(raw[0], raw[2])
    s_south = min(raw[1], raw[3])
    s_north = max(raw[1], raw[3])

    # Intersect with requested bbox
    west  = max(bbox[0], s_left)
    south = max(bbox[1], s_south)
    east  = min(bbox[2], s_right)
    north = min(bbox[3], s_north)

    if west >= east or south >= north:
        raise ValueError(
            f"AOI bbox {bbox} does not intersect scene extent "
            f"({s_left:.3f}, {s_south:.3f}, {s_right:.3f}, {s_north:.3f})"
        )

    # Use inverse transform to map bbox corners → pixel coords.
    # window_from_bounds rejects transforms with e > 0 (south-up), which
    # GCP-derived transforms from raw Sentinel-1 TIFFs often produce.
    inv = ~transform
    corners_geo = [(west, north), (east, north), (east, south), (west, south)]
    corners_px  = [inv * pt for pt in corners_geo]
    col_vals = [c[0] for c in corners_px]
    row_vals = [c[1] for c in corners_px]

    col0 = max(0, int(np.floor(min(col_vals))))
    row0 = max(0, int(np.floor(min(row_vals))))
    col1 = min(W, int(np.ceil(max(col_vals))))
    row1 = min(H, int(np.ceil(max(row_vals))))

    clipped = data[..., row0:row1, col0:col1]
    new_transform = rasterio.transform.from_bounds(
        west, south, east, north,
        clipped.shape[-1], clipped.shape[-2],
    )
    new_meta = meta.copy()
    new_meta.update(
        width=clipped.shape[-1],
        height=clipped.shape[-2],
        transform=new_transform,
    )
    return clipped, new_meta


# ── I/O ────────────────────────────────────────────────────────────────────────

def read_grd_bands(safe_path: Path) -> tuple[np.ndarray, dict]:
    """
    Open VV and VH polarisation bands from a Sentinel-1 GRD SAFE archive.

    Returns:
        data – float32 (2, H, W) array: band 1 = VV dB, band 2 = VH dB
        meta – rasterio profile (from the VV tiff, updated for 2 bands)

    Sentinel-1 GRD measurement TIFFs embed no CRS; their geolocation is
    stored as GCPs (Ground Control Points). When crs is None we derive an
    approximate affine transform and assign EPSG:4326 from those GCPs so
    the downstream reprojection step gets valid bounds.
    """
    pols = find_safe_measurements(safe_path)

    bands = []
    meta  = None
    for pol in ("VV", "VH"):
        with rasterio.open(pols[pol]) as src:
            dn = src.read(1)
            if meta is None:
                meta = src.profile.copy()
                # Raw Sentinel-1 GRD TIFFs have crs=None; derive from GCPs
                if meta.get("crs") is None:
                    gcps, gcp_crs = src.gcps
                    if gcps:
                        from rasterio.transform import from_gcps
                        meta["crs"]       = gcp_crs if gcp_crs else TARGET_CRS
                        meta["transform"] = from_gcps(gcps)
                    else:
                        raise ValueError(
                            f"No CRS and no GCPs found in {pols[pol]}. "
                            "Cannot determine scene geolocation."
                        )
        bands.append(dn_to_sigma0_db(dn))

    data = np.stack(bands, axis=0)  # (2, H, W)
    meta.update(count=2, dtype="float32", nodata=float("nan"))
    return data, meta


def write_geotiff(data: np.ndarray, meta: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if data.ndim == 2:
        meta = {**meta, "count": 1}
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(data, 1)
    else:
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(data)
    print(f"[OK]   Written → {out_path}")


# ── full pipeline ──────────────────────────────────────────────────────────────

def preprocess_scene(
    scene_path: Path,
    out_dir: Path = PROCESSED_DIR,
    clip_bbox: tuple | None = GOG_BBOX,
    speckle_kernel: int = 7,
    reproject: bool = True,
) -> Path:
    """
    Full preprocessing pipeline for one Sentinel-1 GRD SAFE scene.

    Returns path to the output GeoTIFF.
    """
    print(f"\n[PREPROCESS] {scene_path.name}")

    data, meta = read_grd_bands(scene_path)
    print(f"  Read VV+VH  shape={data.shape}  crs={meta.get('crs')}")

    if reproject:
        data, meta = reproject_to_wgs84(data, meta)
        print(f"  Reprojected → EPSG:4326  shape={data.shape}")

    if clip_bbox:
        data, meta = clip_to_bbox(data, meta, clip_bbox)
        print(f"  Clipped to AOI  shape={data.shape}")

    data = lee_filter(data, speckle_kernel)
    print(f"  Lee filter (kernel={speckle_kernel}) applied")

    stem     = scene_path.stem.split(".")[0]
    out_path = out_dir / f"{stem}_processed.tif"
    write_geotiff(data, meta, out_path)
    return out_path


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Preprocess a Sentinel-1 GRD scene.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--scene",    type=Path, required=True,
                   help="SAFE zip, SAFE directory, or pre-calibrated GeoTIFF")
    p.add_argument("--out-dir",  type=Path, default=PROCESSED_DIR)
    p.add_argument("--no-clip",  action="store_true", help="Skip AOI clip")
    p.add_argument("--no-reproject", action="store_true", help="Skip reprojection")
    p.add_argument("--kernel",   type=int, default=7, help="Lee filter kernel size")
    return p.parse_args(argv)


def main(argv=None):
    args  = parse_args(argv)
    clip  = None if args.no_clip else GOG_BBOX
    preprocess_scene(
        args.scene, args.out_dir, clip,
        args.kernel, reproject=not args.no_reproject,
    )


if __name__ == "__main__":
    main()

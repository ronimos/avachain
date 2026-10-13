"""
snowpack_io.py — SNOWPACK data loading and boundary I/O.

Provides:
    load_dataset()       Load xsnow dataset from Zarr cache or .pro files
    build_zarr_cache()   Build / resume Zarr cache from .pro files in batches
    load_boundaries()    Load start zone KML + release area GeoJSON → UTM polys
    kml_to_mask()        Rasterize a KML polygon to a boolean numpy mask

No CLI. Import from pipeline.py, analysis_pipeline.py, and diagnostic scripts.
"""

from __future__ import annotations

import json
import time
import tempfile
import os
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr
import xsnow
from pyproj import Transformer

from config import RASTER_EPSG


# -----------------------------------------------------------------------
# Dataset loading
# -----------------------------------------------------------------------

def load_dataset(pro_dir: Path,
                 zarr_path: Optional[Path] = None,
                 n_cpus: int = 1) -> xr.Dataset:
    """
    Load the SNOWPACK xsnow dataset.

    Preference order:
      1. Zarr cache at zarr_path (fast, lazy via Dask)
      2. Direct read from .pro files in pro_dir (slow, eager)

    Parameters
    ----------
    pro_dir   : directory containing cluster_*.pro files
    zarr_path : path to Zarr store; if it exists, load from there
    n_cpus    : workers for xsnow.read() when reading .pro files directly

    Returns
    -------
    xr.Dataset wrapped by xsnow.xsnowDataset
    """
    if zarr_path and zarr_path.exists():
        print(f"Loading dataset from Zarr: {zarr_path}...")
        dr = xr.open_zarr(str(zarr_path))
        ds = xsnow.xsnowDataset(dr)
        print(f"  dims: { {k: v for k, v in ds.sizes.items()} }")
        return ds

    pro_files = list(pro_dir.glob("cluster_*.pro"))
    if not pro_files:
        raise FileNotFoundError(f"No cluster .pro files found in {pro_dir}")

    print(f"Loading {len(pro_files)} .pro files from {pro_dir}...")
    ds = xsnow.read(str(pro_dir), lazy=False, n_cpus_use=n_cpus)
    print(f"  dims: { {k: v for k, v in ds.sizes.items()} }")
    return ds


# -----------------------------------------------------------------------
# Zarr cache builder
# -----------------------------------------------------------------------

def _resume_state(zarr_out: Path, rebuild: bool) -> tuple[set, bool]:
    """
    Decide what is already in `zarr_out` and whether to append to it.

    Returns (written_locs, append). `append` False means the first write uses
    mode='w' and overwrites whatever is there.

    Raises RuntimeError if the store exists but cannot be read. That case used
    to print "starting fresh" and then append anyway — because the append mode
    was taken from `zarr_out.exists()` rather than from whether the resume
    state had actually been recovered — which silently grafted a second full
    copy of every cluster onto the store. Nine such runs are why
    slope_snowpack.zarr reached 78 GB for 6565 distinct clusters. Appending to
    a store we cannot read is never the right guess, and neither is wiping
    hours of SNOWPACK output without being asked, so refuse and say so.
    """
    if rebuild or not zarr_out.exists():
        return set(), False

    try:
        with xr.open_zarr(str(zarr_out)) as existing:
            written_locs = set(str(v) for v in existing.coords['location'].values)
    except Exception as exc:
        raise RuntimeError(
            f"Zarr store {zarr_out} exists but could not be read ({exc}). "
            f"Refusing to append to it — that is what produced the duplicated "
            f"locations in the 78 GB store. Delete it, or pass rebuild=True "
            f"(--rebuild) to overwrite it."
        ) from exc

    print(f"Resuming — {len(written_locs)} locations already written")
    return written_locs, True


def build_zarr_cache(pro_dir: Path,
                     zarr_out: Path,
                     batch_size: int = 100,
                     n_cpus: int = 31,
                     max_layers: int = 338,
                     rebuild: bool = False) -> None:
    """
    Build a Zarr cache from .pro files, processing in batches.

    Resumable: skips locations already written, per file rather than per batch.
    Squeezes singleton slope/realization dims and pads layers to max_layers.

    Parameters
    ----------
    pro_dir    : directory containing cluster_*_cluster_*.pro files
    zarr_out   : path where Zarr store will be written
    batch_size : .pro files per batch (100 uses ~4 GB RAM)
    n_cpus     : parallel workers passed to xsnow.read()
    max_layers : pad all datasets to this layer count (use global max)
    rebuild    : overwrite an existing store instead of resuming into it
    """
    pro_files = sorted(pro_dir.glob("cluster_*_cluster_*.pro"))
    n_total   = len(pro_files)
    n_batches = int(np.ceil(n_total / batch_size))
    print(f"Files: {n_total}  batch_size: {batch_size}  "
          f"batches: {n_batches}  workers: {n_cpus}")

    written_locs, append = _resume_state(zarr_out, rebuild)

    t_total = time.time()

    for batch_idx in range(n_batches):
        batch_files = pro_files[batch_idx * batch_size:
                                (batch_idx + 1) * batch_size]

        # .pro files are named {station}_{experiment}.pro (e.g.
        # cluster_0001_cluster_0001.pro) but the Zarr location coordinate
        # stores just the station name (cluster_0001). Strip the _<experiment>
        # suffix so the resume check matches.
        #
        # Filter file by file, not all-or-nothing per batch: a batch whose
        # write was interrupted part-way leaves some of its locations in the
        # store, and re-appending the whole batch duplicates exactly those.
        keep, keep_locs, seen = [], [], set()
        for f in batch_files:
            loc = f.stem.split('_cluster_')[0]
            if loc in written_locs or loc in seen:   # seen guards repeats
                continue                             # within the batch itself
            keep.append(f)
            keep_locs.append(loc)
            seen.add(loc)

        if not keep:
            print(f"  Batch {batch_idx+1}/{n_batches}: SKIP")
            continue

        n_skipped   = len(batch_files) - len(keep)
        batch_files = keep

        print(f"  Batch {batch_idx+1}/{n_batches}: "
              f"{batch_files[0].name} … {batch_files[-1].name}"
              + (f"  ({n_skipped} already written, skipped)" if n_skipped else ""))

        with tempfile.TemporaryDirectory() as tmp:
            for f in batch_files:
                os.symlink(str(f.resolve()), os.path.join(tmp, f.name))
            t0 = time.time()
            try:
                ds = xsnow.read(tmp, lazy=False, n_cpus_use=n_cpus,
                                max_layers=max_layers)
            except Exception as exc:
                print(f"    ERROR reading batch: {exc}")
                continue
            t_read = time.time() - t0

        # Normalise dims
        sq_dims = [d for d in ['slope', 'realization'] if d in ds.dims]
        if sq_dims:
            ds = ds.squeeze(sq_dims, drop=True)

        n_layers = ds.sizes.get('layer', 0)
        if n_layers < max_layers:
            ds = ds.pad(layer=(0, max_layers - n_layers),
                        mode='constant', constant_values=np.nan)
        elif n_layers > max_layers:
            ds = ds.isel(layer=slice(0, max_layers))

        t1 = time.time()
        # Append only once we know the store holds a recovered resume state:
        # `zarr_out.exists()` is not that, and using it is what let an
        # unreadable store be appended to.
        written_now = [str(v) for v in ds.coords['location'].values]
        try:
            if append:
                ds.to_zarr(str(zarr_out), mode='a', append_dim='location')
            else:
                ds.to_zarr(str(zarr_out), mode='w')
        except Exception as exc:
            print(f"    ERROR writing Zarr: {exc}")
            continue
        finally:
            del ds

        # Only now are these locations really in the store; a failed write
        # above must leave them eligible for a later resume.
        written_locs.update(written_now)
        append = True

        t_write = time.time() - t1
        elapsed = time.time() - t_total
        frac    = (batch_idx + 1) / n_batches
        eta_min = elapsed / frac * (1 - frac) / 60
        print(f"    read={t_read:.1f}s  write={t_write:.1f}s  "
              f"ETA={eta_min:.0f}min")

    print(f"\nDone in {(time.time()-t_total)/60:.1f}min → {zarr_out}")


# -----------------------------------------------------------------------
# Boundary helpers
# -----------------------------------------------------------------------

def load_boundaries(project_dir: Path,
                    target_epsg: int = RASTER_EPSG) -> dict:
    """
    Load start zone KML and release area GeoJSON, reproject to target CRS.

    Returns a dict with any of these keys, depending on what files exist:
        'start_zone'         : (xs, ys) tuple of UTM coordinate lists
        'release_area'       : (xs, ys) tuple for a single polygon
        'release_area_multi' : list of (xs, ys) tuples for multi-polygon

    Parameters
    ----------
    project_dir  : project root (data/boundaries/ is expected inside)
    target_epsg  : EPSG code for output CRS (default config.RASTER_EPSG)
    """
    from xml.etree import ElementTree as ET
    from shapely.geometry import shape
    from shapely.ops import unary_union

    bnd_dir    = project_dir / 'data' / 'boundaries'
    boundaries = {}

    def _make_transformer(src_crs_str):
        """Build a pyproj Transformer from a CRS string to target_epsg."""
        return Transformer.from_crs(src_crs_str, f'EPSG:{target_epsg}',
                                    always_xy=True)

    # Transformer for the KML (always lon/lat WGS84)
    t_lonlat = _make_transformer('EPSG:4326')

    def to_utm(pts_lonlat):
        return [t_lonlat.transform(x, y) for x, y in pts_lonlat]

    # --- Start zone KML ---
    kml_path = bnd_dir / 'Litte_prof_start_zone.kml'
    if kml_path.exists():
        tree = ET.parse(str(kml_path))
        ns   = '{http://www.opengis.net/kml/2.2}'
        ct   = (tree.find('.//coordinates') or
                tree.find(f'.//{ns}coordinates'))
        if ct is not None:
            raw = [tuple(map(float, p.split(',')[:2]))
                   for p in ct.text.strip().split() if ',' in p]
            utm = to_utm(raw)
            boundaries['start_zone'] = (
                [x for x, y in utm],
                [y for x, y in utm])

    # --- Release area GeoJSON ---
    # Try candidate filenames in priority order; first match wins.
    _release_candidates = [
        'avalanche_release_area.geojson',
        'avalanche_release_area_hand_drawn.geojson',
        'avalanche_release_area_20260118.geojson',
    ]
    gj_path = next((bnd_dir / n for n in _release_candidates
                    if (bnd_dir / n).exists()), None)
    if gj_path is not None:
        with open(str(gj_path)) as f:
            gj = json.load(f)

        # Respect the CRS declared in the file; fall back to EPSG:4326 (the
        # GeoJSON spec default) only when no crs block is present.
        declared = (gj.get('crs') or {}).get('properties', {}).get('name', '')
        src_crs = declared if declared else 'EPSG:4326'
        t_gj = _make_transformer(src_crs)

        polys  = [shape(feat['geometry']) for feat in gj['features']]
        merged = unary_union(polys)

        def _reproj(poly):
            from shapely.geometry import Polygon, MultiPolygon
            def ring(r): return [t_gj.transform(x, y) for x, y in r.coords]
            if poly.geom_type == 'Polygon':
                return Polygon(ring(poly.exterior),
                               [ring(i) for i in poly.interiors])
            return MultiPolygon([_reproj(p) for p in poly.geoms])

        utm_poly = _reproj(merged)
        if utm_poly.geom_type == 'Polygon':
            boundaries['release_area'] = (
                list(utm_poly.exterior.xy[0]),
                list(utm_poly.exterior.xy[1]))
        elif utm_poly.geom_type == 'MultiPolygon':
            boundaries['release_area_multi'] = [
                (list(p.exterior.xy[0]), list(p.exterior.xy[1]))
                for p in utm_poly.geoms]

    return boundaries


def kml_to_mask(kml_path: Path,
                shape: tuple,
                transform,
                target_epsg: int = RASTER_EPSG) -> np.ndarray:
    """
    Rasterize a KML polygon to a boolean mask matching a raster grid.

    Parameters
    ----------
    kml_path     : path to the KML file
    shape        : (nrows, ncols) of the output raster
    transform    : rasterio Affine transform of the raster
    target_epsg  : EPSG for rasterisation (must match raster CRS)

    Returns
    -------
    Boolean np.ndarray of `shape`, True inside the polygon.
    """
    import rasterio.features
    from shapely.geometry import Polygon
    from xml.etree import ElementTree as ET

    t = Transformer.from_crs('EPSG:4326', f'EPSG:{target_epsg}',
                              always_xy=True)

    tree = ET.parse(str(kml_path))
    ns   = '{http://www.opengis.net/kml/2.2}'
    ct   = (tree.find('.//coordinates') or
            tree.find(f'.//{ns}coordinates'))
    if ct is None:
        raise ValueError(f"No <coordinates> found in {kml_path}")

    pts_lonlat = [tuple(map(float, p.split(',')[:2]))
                  for p in ct.text.strip().split() if ',' in p]
    pts_utm    = [t.transform(x, y) for x, y in pts_lonlat]
    poly       = Polygon(pts_utm)

    mask = rasterio.features.geometry_mask(
        [poly.__geo_interface__],
        out_shape=shape,
        transform=transform,
        invert=True)   # True = inside polygon
    return mask


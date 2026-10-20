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
from datetime import datetime
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

def unique_location_index(ds: xr.Dataset,
                          keep_ids: "set[int] | None" = None) -> np.ndarray:
    """Positional index of the first entry for each wanted cluster.

    `keep_ids` None keeps every cluster in the store, which is the dedup case;
    a set restricts to those cluster ids, which is the dev-fixture case.

    First occurrence rather than any: the duplicate entries left by the resume
    bug are byte-identical (verified in the production store across index
    strides of 6565 — equal lat/lon/alt and equal data), so the earliest copy
    is as good as the last, and taking the first makes the result independent
    of how many times the store was re-appended.
    """
    names = np.array([str(v) for v in ds.coords['location'].values])
    ids   = np.array([int(n.split('_')[-1]) for n in names])

    seen, idx = set(), []
    for i, cid in enumerate(ids):
        if (keep_ids is None or cid in keep_ids) and cid not in seen:
            seen.add(cid)
            idx.append(i)

    if keep_ids is not None:
        missing = keep_ids - seen
        if missing:
            print(f"  WARNING: {len(missing)} selected clusters absent from "
                  f"the store, e.g. {sorted(missing)[:5]}")
    n_dup = len(names) - len(set(ids))
    if n_dup:
        print(f"  store holds {len(names)} locations for {len(set(ids))} "
              f"clusters ({n_dup} duplicate entries dropped)")
    return np.array(idx, dtype=int)


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


def _chunk_encoding(ds, max_layers: int, loc_chunk: int = 256) -> dict:
    """Per-variable chunk encoding: one timestep, whole layer column.

    Matches what dedup_zarr produces -- (location=256, time=1, layer=338) --
    so a rebuild and a rechunk agree. Wide-time chunks compress better, so the
    store is smaller with the bad layout; size is not the signal here.
    """
    enc = {}
    for name, var in ds.data_vars.items():
        chunks = []
        for d in var.dims:
            if d == 'location':
                chunks.append(loc_chunk)
            elif d == 'time':
                chunks.append(1)
            elif d == 'layer':
                chunks.append(max_layers)
            else:
                chunks.append(ds.sizes[d])
        enc[name] = {'chunks': tuple(chunks)}
    return enc


def _pro_block_time(line: str):
    """Timestamp of a `0500,dd.mm.yyyy HH:MM:SS` profile-block line."""
    return np.datetime64(datetime.strptime(line[5:].strip(), "%d.%m.%Y %H:%M:%S"))


def _write_pro_tail(src: Path, dst: Path, since) -> list:
    """Copy the header of `src` plus only the profile blocks dated after `since`.

    Returns the timestamps written. xsnow then parses only the window, so an
    incremental update costs time proportional to the re-run, not the season.
    """
    since = np.datetime64(since)
    times, keep, in_data = [], True, False
    with open(src) as fin, open(dst, "w") as fout:
        for line in fin:
            if not in_data:
                fout.write(line)
                in_data = line.strip() == "[DATA]"
                continue
            if line.startswith("0500,"):
                t = _pro_block_time(line)
                keep = t > since
                if keep:
                    times.append(t)
            if keep:
                fout.write(line)
    return times


def _read_pro_batch(files: list, n_cpus: int, max_layers: int, since=None):
    """xsnow-read a batch of .pro files, squeezed and padded to max_layers.

    With `since`, each file is first cut to its blocks after `since`. Returns
    None when no file has any such block.
    """
    with tempfile.TemporaryDirectory() as tmp:
        n_blocks = 0
        for f in files:
            dst = os.path.join(tmp, f.name)
            if since is None:
                os.symlink(str(f.resolve()), dst)
                n_blocks = 1
            else:
                n_blocks += len(_write_pro_tail(f, Path(dst), since))
        if n_blocks == 0:
            return None
        ds = xsnow.read(tmp, lazy=False, n_cpus_use=n_cpus, max_layers=max_layers)

    sq_dims = [d for d in ['slope', 'realization'] if d in ds.dims]
    if sq_dims:
        ds = ds.squeeze(sq_dims, drop=True)
    n_layers = ds.sizes.get('layer', 0)
    if n_layers < max_layers:
        ds = ds.pad(layer=(0, max_layers - n_layers),
                    mode='constant', constant_values=np.nan)
    elif n_layers > max_layers:
        ds = ds.isel(layer=slice(0, max_layers))
    return ds


def _resize_time(zarr_out: Path, n: int) -> None:
    """Resize every array that has a time dimension to n timesteps."""
    import zarr
    g = zarr.open_group(str(zarr_out), mode='r+')
    for _, arr in g.arrays():
        names = getattr(arr.metadata, 'dimension_names', None) \
            or arr.attrs.get('_ARRAY_DIMENSIONS')
        if names and 'time' in names:
            shape = list(arr.shape)
            shape[list(names).index('time')] = n
            arr.resize(tuple(shape))
    zarr.consolidate_metadata(str(zarr_out))


def update_zarr_since(pro_dir: Path,
                      zarr_out: Path,
                      since,
                      batch_size: int = 256,
                      n_cpus: int = 31,
                      max_layers: int = 338) -> int:
    """
    Replace everything in the store after `since` with the .pro blocks after it.

    This is the incremental update for a SNOWPACK re-run from a restart point:
    the daily anchor (yesterday's T_stable) or the survey anchor (the previous
    survey). SNOWPACK rewrites each .pro from its begin date, so the blocks
    after `since` are the new truth and everything the store held after
    `since` is discarded. That includes earlier inter-survey estimates and
    forecast steps, which must never survive as analysis.

    build_zarr_cache cannot do this: it resumes by skipping locations already
    in the store and only appends along location, so after the first build no
    incremental run ever extended the store in time, and analyze silently read
    the latest stored profile.

    The time axis is shrunk to the last step <= since, grown by the new steps,
    and filled by region writes in location batches aligned to the 256-location
    chunks. Locations without a block at some new step stay NaN there. Returns
    the number of timesteps written.
    """
    since = np.datetime64(since)
    with xr.open_zarr(str(zarr_out)) as ex:
        old_times = ex['time'].values
        locs = [str(v) for v in ex['location'].values]
    keep_n = int((old_times <= since).sum())
    if keep_n == 0:
        raise ValueError(f"{zarr_out} has no timestep <= {since}; rebuild it "
                         f"with build_zarr_cache instead")

    pro_by_loc = {f.stem.split('_cluster_')[0]: f
                  for f in pro_dir.glob("cluster_*_cluster_*.pro")}
    new_times = set()
    with tempfile.TemporaryDirectory() as tmp:
        for loc in locs:
            if loc in pro_by_loc:
                new_times.update(_write_pro_tail(
                    pro_by_loc[loc], Path(tmp) / "probe.pro", since))
    new_times = np.array(sorted(new_times), dtype='datetime64[ns]')
    n_new = len(new_times)

    _resize_time(zarr_out, keep_n + n_new)
    if n_new == 0:
        print(f"update_zarr_since: no profiles after {since}; store cut to "
              f"{keep_n} timesteps")
        return 0

    # The time coordinate, written in the store's own encoding.
    import zarr
    import pandas as pd
    g = zarr.open_group(str(zarr_out), mode='r+')
    units = g['time'].attrs['units']                     # "<unit> since <origin>"
    unit, origin = units.split(' since ')
    step = {'hours': 'h', 'minutes': 'm', 'seconds': 's', 'days': 'D'}[unit]
    vals = (new_times - np.datetime64(pd.Timestamp(origin))) / np.timedelta64(1, step)
    g['time'][keep_n:] = vals.astype(g['time'].dtype)

    region_t = slice(keep_n, keep_n + n_new)
    t_total = time.time()
    for i0 in range(0, len(locs), batch_size):
        batch_locs = locs[i0:i0 + batch_size]
        files = [pro_by_loc[l] for l in batch_locs if l in pro_by_loc]
        ds = _read_pro_batch(files, n_cpus, max_layers, since) if files else None
        if ds is None:
            continue
        ds = ds.reindex(location=batch_locs, time=new_times)
        ds = ds.reset_coords([c for c in ds.coords
                              if 'time' in ds[c].dims and c != 'time'])
        ds = ds[[v for v in ds.data_vars if 'time' in ds[v].dims]]
        ds = ds.drop_vars([c for c in ('location', 'time', 'layer') if c in ds.coords])
        for v in ds.variables.values():
            v.encoding = {}
        ds.to_zarr(str(zarr_out), mode='r+',
                   region={'location': slice(i0, i0 + len(batch_locs)),
                           'time': region_t})
    zarr.consolidate_metadata(str(zarr_out))
    print(f"update_zarr_since: kept {keep_n} timesteps <= {since}, wrote "
          f"{n_new} ({str(new_times[0])[:16]} -> {str(new_times[-1])[:16]}) in "
          f"{time.time() - t_total:.0f}s")
    return n_new


def build_zarr_cache(pro_dir: Path,
                     zarr_out: Path,
                     batch_size: int = 100,
                     n_cpus: int = 31,
                     max_layers: int = 338,
                     rebuild: bool = False,
                     since=None) -> None:
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
    since      : if given, keep only profiles dated after it. This is how the
                 NWP forecast store is built from the forecast pass; pass
                 rebuild=True with it, since resuming a windowed store makes
                 no sense.
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

        t0 = time.time()
        try:
            ds = _read_pro_batch(batch_files, n_cpus, max_layers, since)
        except Exception as exc:
            print(f"    ERROR reading batch: {exc}")
            continue
        if ds is None:
            print(f"    no profiles after {since}; nothing to write")
            continue
        t_read = time.time() - t0

        t1 = time.time()
        # Append only once we know the store holds a recovered resume state:
        # `zarr_out.exists()` is not that, and using it is what let an
        # unreadable store be appended to.
        written_now = [str(v) for v in ds.coords['location'].values]
        try:
            if append:
                ds.to_zarr(str(zarr_out), mode='a', append_dim='location')
            else:
                # Chunk encoding is fixed when the store is created, so it can
                # only be set on this first write. time=1 because EVERY
                # consumer reads a single snapshot (isel(time=tidx) /
                # sel(time=ts) in analysis_pipeline, analyze_release_zone,
                # release_geometry, snowpack_analysis); inheriting the reader's
                # (25, 126, 85) chunks makes a one-timestep read decompress
                # 126 timesteps. This used to be corrected after the fact by
                # dedup_zarr's rechunk, so any plain rebuild silently
                # reintroduced the slow layout.
                ds.to_zarr(str(zarr_out), mode='w',
                           encoding=_chunk_encoding(ds, max_layers))
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


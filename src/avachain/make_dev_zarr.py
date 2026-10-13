"""
make_dev_zarr.py — cut a small, fast Zarr fixture out of the full store for
development iteration.

The full store is slow to iterate against for three separate reasons, and only
one of them is its cluster count:

  1. Duplicated locations. Appends that re-ran the whole cluster set left
     slope_snowpack.zarr with 59280 `location` entries covering 6565 distinct
     clusters — each present 9-12 times with byte-identical data. Selecting
     clusters by `int(name.split('_')[-1]) in ids` matches every copy, so the
     1608-cluster Jan 18 start zone pulls 14502 locations and the per-cluster
     loop runs ~9x. See snowpack_io._resume_state for the cause and the fix;
     this script drops the duplicates on the way out, so a fixture cut from a
     duplicated store is already clean.

  2. Chunking against the access pattern. Chunks are
     (location=25, time=126, layer=85) but `analyze` reads one snapshot, so
     every read decompresses 126 timesteps to use one. Measured on 500
     locations: one timestep costs 1.4 s, 126 timesteps cost 0.7 s — the same
     chunks either way. Rechunking to time=1 is the single biggest win and it
     does not change which clusters you are testing on.

  3. Cluster count. Real, but linear, and the only one of the three that
     changes what is under test.

Selection is a *dilated* start zone, not the start zone itself: the release
BFS grows outward from its trigger and _build_scour_depth_map wants every
cluster in the domain, with a full-depth fallback for path and deposition
clusters. A fixture cut to the start zone exactly makes those reads miss, and
a missing location fails as if it were physics.

The fixture is for iteration only. The full domain stays the acceptance gate —
a subset structurally cannot see BFS-escape or anything else domain-scale, the
same reason the 40-50 Pa tau_g band needs a unit test rather than Jan 18.

Usage
-----
    python src/avachain/make_dev_zarr.py \
        --toml slopes/little_prof/slope_config.toml \
        --dates 2026-01-18 2026-03-07 \
        --pad-days 2 --buffer-m 200

    # then point the pipeline at it
    python src/avachain/analysis_pipeline.py analyze \
        --toml slopes/little_prof/slope_config.toml \
        --zarr-path snowpack/little_prof/output/dev_snowpack.zarr \
        --snapshot-date 2026-01-18 --force
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import ProjectConfig                      # noqa: E402
from snowpack_io import kml_to_mask                   # noqa: E402

# Layers above this are NaN pad in practice: the median Jan 18 profile uses
# 102 of 338 and the deepest uses 258. Trimming is safe but is the one step
# that can silently truncate a profile, so the real max is measured at run
# time and this is only the ceiling.
DEFAULT_MAX_LAYERS = 338


def dilated_start_zone_clusters(cfg, buffer_m: float) -> set[int]:
    """Cluster ids whose pixels fall inside the start zone dilated by buffer_m.

    The buffer is what keeps the release BFS and the scour-depth scan from
    reading locations the fixture does not contain.
    """
    dem_tif = cfg.resampled_dir / "dem_1m.tif"
    with rasterio.open(str(dem_tif)) as src:
        shape     = (src.height, src.width)
        transform = src.transform
        px_m      = abs(transform.a)

    mask = kml_to_mask(cfg.start_zone_kml, shape, transform)
    n_core = int(mask.sum())

    if buffer_m > 0:
        iters = max(1, int(round(buffer_m / px_m)))
        mask = ndimage.binary_dilation(mask, iterations=iters)

    cluster_map = np.load(str(cfg.analysis_dir / "cluster_map.npy"))
    ids = np.unique(cluster_map[mask])
    ids = set(int(c) for c in ids if c > 0)

    print(f"  start zone: {n_core} px → {int(mask.sum())} px "
          f"after {buffer_m:g} m buffer ({px_m:g} m/px)")
    print(f"  clusters selected: {len(ids)}")
    return ids


def unique_location_index(ds: xr.Dataset, keep_ids: set[int]) -> np.ndarray:
    """Positional index of the first entry for each wanted cluster.

    First rather than any: the duplicates are byte-identical (verified across
    index strides of 6565 in the production store), so the earliest copy is as
    good as the last, and taking one makes the result independent of how many
    times the store was re-appended.
    """
    names = np.array([str(v) for v in ds.coords['location'].values])
    ids   = np.array([int(n.split('_')[-1]) for n in names])

    seen, idx = set(), []
    for i, cid in enumerate(ids):
        if cid in keep_ids and cid not in seen:
            seen.add(cid)
            idx.append(i)

    missing = keep_ids - seen
    if missing:
        print(f"  WARNING: {len(missing)} selected clusters absent from the "
              f"store, e.g. {sorted(missing)[:5]}")
    n_dup = len(names) - len(set(ids))
    if n_dup:
        print(f"  store holds {len(names)} locations for {len(set(ids))} "
              f"clusters ({n_dup} duplicate entries dropped)")
    return np.array(idx, dtype=int)


def time_index(ds: xr.Dataset, dates: list[str], pad_days: int) -> np.ndarray:
    """Positional index of every timestep within pad_days of a wanted date.

    The window is whole days, so --dates 2026-01-18 --pad-days 0 keeps all of
    the 18th rather than only the 00:00 step — SNOWPACK output is 6-hourly and
    the snapshot a date resolves to is not necessarily midnight.
    """
    times = pd.DatetimeIndex(ds.coords['time'].values)
    keep  = np.zeros(len(times), dtype=bool)
    pad   = pd.Timedelta(days=pad_days)
    day   = pd.Timedelta(days=1)

    for d in dates:
        t = pd.Timestamp(d).normalize()
        window = (times >= t - pad) & (times < t + pad + day)
        if not window.any():
            # Nearest, so a date just outside the SNOWPACK run still yields a
            # usable snapshot rather than an empty fixture.
            nearest = int(np.abs(times - t).argmin())
            print(f"  WARNING: no timesteps within {pad_days} d of {d}; "
                  f"falling back to nearest ({times[nearest].date()})")
            window = np.zeros(len(times), dtype=bool)
            window[nearest] = True
        keep |= window

    print(f"  timesteps: {int(keep.sum())} of {len(times)} "
          f"({dates}, ±{pad_days} d)")
    return np.where(keep)[0]


def measure_max_layer(ds: xr.Dataset, probe_var: str = 'z') -> int:
    """Deepest layer index holding data, so the pad can be trimmed safely."""
    if probe_var not in ds:
        return ds.sizes['layer']
    arr   = ds[probe_var].compute().values
    valid = ~np.isnan(arr)
    axes  = tuple(i for i, d in enumerate(ds[probe_var].dims) if d != 'layer')
    used  = valid.any(axis=axes)
    n     = int(np.where(used)[0].max()) + 1 if used.any() else ds.sizes['layer']
    print(f"  layers in use: {n} of {ds.sizes['layer']}")
    return n


def build(src_zarr: Path, out_zarr: Path, cfg, dates: list[str],
          pad_days: int, buffer_m: float, loc_chunk: int,
          trim_layers: bool) -> None:
    t0 = time.time()
    print(f"Source: {src_zarr}")
    ds = xr.open_zarr(str(src_zarr))
    print(f"  dims: { {k: v for k, v in ds.sizes.items()} }")

    print("Selecting clusters...")
    keep_ids = dilated_start_zone_clusters(cfg, buffer_m)
    loc_idx  = unique_location_index(ds, keep_ids)
    if loc_idx.size == 0:
        raise ValueError("No locations selected — check cluster_map.npy and "
                         "the start zone KML against the store.")

    print("Selecting timesteps...")
    t_idx = time_index(ds, dates, pad_days)

    sub = ds.isel(location=loc_idx, time=t_idx)

    if trim_layers:
        print("Measuring layer use...")
        n_layer = measure_max_layer(sub)
        sub = sub.isel(layer=slice(0, n_layer))

    # time=1 is the whole point: `analyze` reads one snapshot at a time.
    chunks = {'location': min(loc_chunk, sub.sizes['location']),
              'time': 1,
              'layer': sub.sizes['layer']}
    chunks = {d: c for d, c in chunks.items() if d in sub.dims}
    sub = sub.chunk(chunks)

    # Dask chunk layout wins over whatever encoding came off the source store.
    for v in list(sub.data_vars) + list(sub.coords):
        sub[v].encoding.pop('chunks', None)
        sub[v].encoding.pop('preferred_chunks', None)

    print(f"Writing {out_zarr}")
    print(f"  dims: { {k: v for k, v in sub.sizes.items()} }  chunks: {chunks}")
    if out_zarr.exists():
        raise FileExistsError(f"{out_zarr} exists — delete it first")
    out_zarr.parent.mkdir(parents=True, exist_ok=True)
    sub.to_zarr(str(out_zarr), mode='w')

    size_mb = sum(f.stat().st_size for f in out_zarr.rglob('*') if f.is_file())
    size_mb /= 1024 ** 2
    print(f"Done in {time.time()-t0:.0f}s — {size_mb:.0f} MB")


def main():
    p = argparse.ArgumentParser(
        description="Cut a deduplicated, rechunked dev Zarr fixture out of the "
                    "full SNOWPACK store.")
    p.add_argument('--toml', default=None,
                   help='Slope config TOML (as passed to analysis_pipeline).')
    p.add_argument('--project-dir', default='.')
    p.add_argument('--zarr-path', default=None,
                   help='Source store (default: cfg.zarr_path).')
    p.add_argument('--out', default=None,
                   help='Output store (default: <pro_dir>/dev_snowpack.zarr).')
    p.add_argument('--dates', nargs='+', default=['2026-01-18'],
                   help='Snapshot dates to keep.')
    p.add_argument('--pad-days', type=int, default=2,
                   help='Keep timesteps within this many days of each date, '
                        'so reinit and dHS windows still work (default 2).')
    p.add_argument('--buffer-m', type=float, default=200.0,
                   help='Dilate the start zone by this many metres before '
                        'selecting clusters, so the release BFS and the '
                        'scour-depth scan stay inside the fixture '
                        '(default 200).')
    p.add_argument('--loc-chunk', type=int, default=256)
    p.add_argument('--no-trim-layers', action='store_true',
                   help='Keep all 338 layers instead of trimming NaN pad.')
    args = p.parse_args()

    cfg = (ProjectConfig.from_toml(args.toml) if args.toml
           else ProjectConfig(project_dir=Path(args.project_dir)))

    src = Path(args.zarr_path) if args.zarr_path else cfg.zarr_path
    out = Path(args.out) if args.out else src.parent / "dev_snowpack.zarr"

    build(src, out, cfg, args.dates, args.pad_days, args.buffer_m,
          args.loc_chunk, not args.no_trim_layers)


if __name__ == '__main__':
    main()

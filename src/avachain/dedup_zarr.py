"""
dedup_zarr.py — repair a SNOWPACK Zarr store that holds duplicate locations,
and rechunk it for the way the pipeline actually reads it.

Why this exists
---------------
`build_zarr_cache` used to append a second full copy of every cluster whenever
it could not read the resume state of an existing store (and to re-append a
whole batch whose write had been interrupted). Both holes are fixed in
snowpack_io, but the store they produced is still on disk:
slope_snowpack.zarr is 78 GB holding 59280 `location` entries for 6565
distinct clusters, each present 9-12 times with byte-identical data.

Nothing downstream is *wrong* because of it — the duplicates are identical, so
the per-cluster dict collapses and the ravel over stacked copies lands on the
same answer — but every full-domain read pays for it. Selection by
`int(name.split('_')[-1]) in ids` matches every copy, so the 1608-cluster
Jan 18 start zone pulls 14502 locations and `analyze` takes 25 minutes.

The second cost is chunking. Chunks are (location=25, time=126, layer=85) and
every consumer reads a single timestep — `isel(time=tidx)` or
`sel(time=ts, method='nearest')` in analysis_pipeline, analyze_release_zone,
release_geometry and snowpack_analysis, with no 3D read spanning a time slab.
So each snapshot decompresses 126 timesteps to use one. Rechunking to time=1
is therefore safe for every current consumer; if a future one wants long time
series from a 3D variable, that is the trade to revisit.

The same applies to the two 2D (location, time) variables, HS and
profile_status: every zarr read of them is at a single timestep too
(`ds_t['HS']`), and the HS history used for dHS/dt comes from the SMET and
forcing side, not from here. A full-time-series read of HS does get slower —
measured 0.1 s to 3.2 s for 1608 locations, from chunks (100, 501) to
(256, 1) — so that is the number to weigh if a consumer ever wants one.

Two costs of time=1 worth knowing before rebuilding a store this way:
time=1 chunks compress worse than time=126 chunks, because the temporal
redundancy a codec can exploit within a chunk is gone — the deduped store came
out at 12.5 GB where 78 GB / 9 predicts 8.7 GB. And a cold read of a freshly
written store is slow: the first full-domain `analyze` against it took 7m16s
against ~17 s on every subsequent warm run.

What it does
------------
Writes a new store holding the first entry of each cluster, all timesteps and
all layers, rechunked. It never touches the source. The swap is a separate,
explicit step (`--swap`) that renames the old store aside rather than deleting
it, because this is operational input and the rebuild from .pro files is hours.

Usage
-----
    # write + verify, leaving both stores in place
    python src/avachain/dedup_zarr.py --toml slopes/little_prof/slope_config.toml

    # then, once the verification output looks right
    python src/avachain/dedup_zarr.py --toml ... --swap

Verification compares the deduped store against the source directly: cluster
set, dims, and every variable over a sample of clusters at a sample of
timesteps, at zero tolerance. The real acceptance test is a full `analyze` run
whose feature CSVs match the pre-dedup ones bit for bit.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import ProjectConfig                              # noqa: E402
from snowpack_io import unique_location_index                 # noqa: E402


def _dir_size_gb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob('*') if f.is_file()) / 1024 ** 3


def as_slice(idx: np.ndarray) -> "slice | np.ndarray":
    """Prefer a contiguous slice over fancy indexing when the index is one.

    Worth the check: the duplicates were appended whole-set at a time, so the
    first copy of every cluster is the leading block and the dedup collapses to
    isel(location=slice(0, n)) — a contiguous read of ~11% of the store rather
    than a scattered gather across all of it.
    """
    if idx.size and np.array_equal(idx, np.arange(idx[0], idx[0] + idx.size)):
        return slice(int(idx[0]), int(idx[-1]) + 1)
    return idx


def dedup(src: Path, out: Path, loc_chunk: int, time_chunk: int) -> None:
    t0 = time.time()
    print(f"Source: {src}")
    ds = xr.open_zarr(str(src))
    print(f"  dims: { {k: v for k, v in ds.sizes.items()} }  "
          f"vars: {len(ds.data_vars)}")

    idx = unique_location_index(ds)
    if idx.size == ds.sizes['location']:
        print("  no duplicate locations — nothing to dedup")
    sel = as_slice(idx)
    print(f"  keeping {idx.size} locations "
          f"({'contiguous slice' if isinstance(sel, slice) else 'gather'})")

    sub = ds.isel(location=sel)

    chunks = {'location': min(loc_chunk, sub.sizes['location']),
              'time': min(time_chunk, sub.sizes['time']),
              'layer': sub.sizes['layer']}
    chunks = {d: c for d, c in chunks.items() if d in sub.dims}
    sub = sub.chunk(chunks)
    for v in list(sub.data_vars) + list(sub.coords):
        sub[v].encoding.pop('chunks', None)
        sub[v].encoding.pop('preferred_chunks', None)

    if out.exists():
        raise FileExistsError(f"{out} exists — delete it first")
    print(f"Writing {out}")
    print(f"  dims: { {k: v for k, v in sub.sizes.items()} }  chunks: {chunks}")
    out.parent.mkdir(parents=True, exist_ok=True)
    sub.to_zarr(str(out), mode='w')
    ds.close()

    print(f"Written in {(time.time()-t0)/60:.1f} min — "
          f"{_dir_size_gb(out):.1f} GB")


def verify(src: Path, out: Path, n_clusters: int, n_times: int,
           seed: int = 0) -> bool:
    """Compare the deduped store against the source at zero tolerance."""
    print(f"\nVerifying {out} against {src}")
    a = xr.open_zarr(str(src))
    b = xr.open_zarr(str(out))
    ok = True

    def check(label, cond, detail=""):
        nonlocal ok
        print(f"  {'OK  ' if cond else 'FAIL'}  {label}{detail}")
        ok = ok and cond

    a_ids = sorted({int(str(v).split('_')[-1])
                    for v in a.coords['location'].values})
    b_names = [str(v) for v in b.coords['location'].values]
    b_ids = sorted(int(n.split('_')[-1]) for n in b_names)

    check("cluster set preserved", a_ids == b_ids,
          f"  ({len(b_ids)} clusters)")
    check("no duplicate locations", len(b_names) == len(set(b_names)))
    for d in ('time', 'layer'):
        check(f"{d} dim preserved", a.sizes[d] == b.sizes[d],
              f"  ({b.sizes[d]})")
    check("variable set preserved",
          set(a.data_vars) == set(b.data_vars), f"  ({len(b.data_vars)} vars)")
    check("time coord identical",
          np.array_equal(a.coords['time'].values, b.coords['time'].values))

    # Per-variable value check on a random sample of clusters and timesteps.
    #
    # Positional indexing throughout: the source's `location` index has
    # duplicate labels, and .sel() on a non-unique index raises
    # InvalidIndexError rather than returning the copies, so label selection
    # cannot be used on the thing we are checking.
    def first_positions(ds_):
        pos = {}
        for i, v in enumerate(ds_.coords['location'].values):
            pos.setdefault(int(str(v).split('_')[-1]), i)
        return pos

    a_pos, b_pos = first_positions(a), first_positions(b)

    rng  = np.random.default_rng(seed)
    cids = sorted(rng.choice(b_ids, size=min(n_clusters, len(b_ids)),
                             replace=False).tolist())
    tidx = np.sort(rng.choice(b.sizes['time'],
                              size=min(n_times, b.sizes['time']),
                              replace=False))
    a_idx = [a_pos[c] for c in cids]
    b_idx = [b_pos[c] for c in cids]

    print(f"  comparing all {len(b.data_vars)} variables over "
          f"{len(cids)} clusters x {len(tidx)} timesteps...")
    bad = []
    for v in sorted(b.data_vars):
        x = np.asarray(
            a[v].isel(location=a_idx, time=tidx).compute().values)
        y = np.asarray(
            b[v].isel(location=b_idx, time=tidx).compute().values)
        if x.shape != y.shape:
            bad.append(f"{v} (shape {x.shape} vs {y.shape})")
            continue
        if x.dtype.kind in 'fc':
            if not np.array_equal(x, y, equal_nan=True):
                bad.append(f"{v} (max |diff| "
                           f"{np.nanmax(np.abs(x - y))})")
        elif not np.array_equal(x, y):
            bad.append(v)

    check("sampled values identical", not bad,
          "" if not bad else f"  — {bad[:6]}")

    a.close()
    b.close()
    print(f"\n{'VERIFIED' if ok else 'VERIFICATION FAILED'}")
    return ok


def swap(src: Path, out: Path) -> None:
    """Move the deduped store into place, keeping the original alongside.

    Rename, never delete: rebuilding from .pro files is hours of work, and the
    point of this tool is to stop losing data to a confident guess.
    """
    aside = src.with_name(src.name + ".duplicated")
    if aside.exists():
        raise FileExistsError(f"{aside} already exists — move or remove it")
    print(f"\nMoving {src}\n    -> {aside}")
    shutil.move(str(src), str(aside))
    print(f"Moving {out}\n    -> {src}")
    shutil.move(str(out), str(src))
    print(f"\nDone. The duplicated store is kept at:\n  {aside}\n"
          f"Delete it once you are satisfied:  rm -rf {aside}")


def main():
    p = argparse.ArgumentParser(
        description="Deduplicate and rechunk a SNOWPACK Zarr store.")
    p.add_argument('--toml', default=None)
    p.add_argument('--project-dir', default='.')
    p.add_argument('--zarr-path', default=None,
                   help='Store to repair (default: cfg.zarr_path).')
    p.add_argument('--out', default=None,
                   help='Where to write the deduped store '
                        '(default: <store>.dedup).')
    p.add_argument('--loc-chunk', type=int, default=256)
    p.add_argument('--time-chunk', type=int, default=1,
                   help='Timesteps per chunk. 1 matches every current '
                        'consumer, which all read one snapshot at a time '
                        '(default 1).')
    p.add_argument('--verify-clusters', type=int, default=24)
    p.add_argument('--verify-times', type=int, default=6)
    p.add_argument('--skip-write', action='store_true',
                   help='Only verify an already-written deduped store.')
    p.add_argument('--swap', action='store_true',
                   help='After verifying, move the deduped store into place '
                        'and rename the original to <name>.duplicated.')
    args = p.parse_args()

    cfg = (ProjectConfig.from_toml(args.toml) if args.toml
           else ProjectConfig(project_dir=Path(args.project_dir)))

    src = Path(args.zarr_path) if args.zarr_path else cfg.zarr_path
    out = Path(args.out) if args.out else src.with_name(src.name + ".dedup")

    if not args.skip_write:
        dedup(src, out, args.loc_chunk, args.time_chunk)

    if not verify(src, out, args.verify_clusters, args.verify_times):
        sys.exit("Refusing to continue — verification failed.")

    if args.swap:
        swap(src, out)
    else:
        print(f"\nNot swapping (pass --swap). Deduped store at:\n  {out}")


if __name__ == '__main__':
    main()

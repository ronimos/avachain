"""
analysis_pipeline.py — Post-SNOWPACK analysis pipeline (Pipeline B).

Steps (run after SNOWPACK finishes and Zarr cache is available):
    zarr_build      Build / resume Zarr cache from .pro files
    analyze         Snapshot comparison, Meloche features, deviation plots
    scenarios       Generate AvaFrame com1DFA release scenario ensemble

Usage:
    python analysis_pipeline.py zarr_build
    python analysis_pipeline.py analyze --snapshot-date 2026-01-18
    python analysis_pipeline.py scenarios --snapshot-date 2026-01-18
    python analysis_pipeline.py all --snapshot-date 2026-01-18

Requires Pipeline A (forcing_pipeline.py) to have been run first to produce:
    outputs/analysis/cluster_map.npy
    outputs/analysis/release_zone_groups.json  (from step_avalanche)

Note: cluster_update (mid-season splitting) lives in forcing_pipeline.py —
it is a pre-SNOWPACK data-prep step, not a post-SNOWPACK analysis step.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

# ---------------------------------------------------------------------------
# Module imports — all logic lives in the modules below
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import ProjectConfig, RASTER_EPSG, replay_as_of
from snowpack_io import (load_dataset, load_boundaries, kml_to_mask,
                         snapshot_time_index)
from snowpack_analysis import (
    assign_cluster_groups,
    profile_features,
    extract_group_timeseries,
    compute_meloche_features,
    GROUP_COLORS, GROUP_LABELS,
)
import release_geometry as _rg
import energy_path as _ep
from release_geometry import (
    compute_slope_aspect,
    make_release_polygon_2d,
    propagate_release,
    depth_from_snowpack,
    rasterize_release_polygon,
    plot_release_comparison,
)
from scenario_writer import (
    write_trigger_locations,
    write_scenario,
    write_scenario_weights,
    write_summary_csv,
    write_metadata,
)

# ---------------------------------------------------------------------------
# Defaults — path and parameter defaults are in config.py.
# ---------------------------------------------------------------------------

def default_snapshot_date() -> str:
    """"Now" as YYYY-MM-DD: the replay clock if set, else today's UTC date.

    The default used to be the hard-coded validation date 2026-01-18, and the
    operational scripts call analyze/scenarios without --snapshot-date, so
    every operational run would have analysed January 18.
    """
    as_of = replay_as_of()
    if as_of is not None:
        return as_of.date().isoformat()
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).date().isoformat()


def resolve_snapshot_date(value) -> str:
    return value if value else default_snapshot_date()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _load_dem(cfg):
    dem_tif = cfg.resampled_dir / "dem_1m.tif"
    with rasterio.open(str(dem_tif)) as src:
        dem       = src.read(1).astype(np.float32)
        dem[dem == src.nodata] = np.nan
        transform = src.transform
        crs_wkt   = src.crs.to_wkt()
        profile   = src.profile.copy()
    return dem, transform, crs_wkt, profile


def _load_groups(cfg, dem, transform, snapshot_date=None):
    """Load or build release/adjacent/reference cluster group assignments.

    Groups are date-specific: all observed release GeoJSONs up to
    snapshot_date are unioned to form the release mask.  When no events
    exist for the date (operational no-avalanche day), the release group
    is empty and reference terrain-matching is skipped.
    """
    cache_name = (f"release_zone_groups_{snapshot_date}.json"
                  if snapshot_date else "release_zone_groups.json")
    groups_path = cfg.analysis_dir / cache_name
    # A cache older than any release polygon it would be built from is stale
    # (the validation target changing must not be masked by the cache).
    sources = (cfg.release_geojsons_for_date(snapshot_date) if snapshot_date
               else [cfg.release_geojson] if cfg.release_geojson.exists() else [])
    if groups_path.exists() and any(
            p.stat().st_mtime > groups_path.stat().st_mtime for p in sources):
        print(f"  {cache_name} is older than its release polygon(s) — rebuilding")
    elif groups_path.exists():
        with open(str(groups_path)) as f:
            raw = json.load(f)
        return {k: set(v) for k, v in raw.items()}

    print("Building cluster group assignments from boundaries...")
    from snowpack_analysis import geojson_to_mask
    cluster_map = np.load(str(cfg.analysis_dir / "cluster_map.npy"))

    release_mask = np.zeros(dem.shape, dtype=bool)
    if snapshot_date:
        gj_paths = cfg.release_geojsons_for_date(snapshot_date)
        for gj_path in gj_paths:
            release_mask |= geojson_to_mask(gj_path, dem.shape, transform)
        if gj_paths:
            print(f"  Release mask from {len(gj_paths)} event GeoJSON(s), "
                  f"{release_mask.sum()} cells")
        else:
            print("  No release area GeoJSONs for this date — empty release group")
    else:
        if cfg.release_geojson.exists():
            release_mask = geojson_to_mask(cfg.release_geojson, dem.shape, transform)

    start_zone_mask = kml_to_mask(cfg.start_zone_kml, dem.shape, transform)
    domain_mask     = ~np.isnan(dem)

    groups = assign_cluster_groups(
        cluster_map, release_mask, start_zone_mask, dem, domain_mask)

    groups_path.write_text(
        json.dumps({k: [int(c) for c in v] for k, v in groups.items()}, indent=2))
    print(f"  Groups saved: { {k: len(v) for k, v in groups.items()} }")
    return groups


def _rank_triggers(cand_df: pd.DataFrame, n: int, sk38_col: str,
                   rc_col: str = 'rc_wl') -> pd.DataFrame:
    """
    Top-n trigger candidates: lowest Sk38 first, ties broken by lowest r_c.

    SNOWPACK reports Sk38 to two decimals, so places 3-5 regularly sit inside
    a tie (4-way at 0.79 on 2026-01-18, 10-way on 2026-03-07). nsmallest()
    settled those by CSV row order, i.e. by nothing physical. The critical
    cut length r_c at the same weak layer breaks them by propagation
    propensity instead (lower = more propagation-prone). Only exact Sk38 ties
    are affected; Sk38 still decides every place it separates. Clusters with
    no r_c sort after those with one; anything still tied keeps row order (the
    multi-column sort is stable).

    rc_col is optional so older feature CSVs without it still rank by Sk38.
    """
    ranked = cand_df.dropna(subset=[sk38_col])
    keys = [sk38_col] + ([rc_col] if rc_col in ranked.columns else [])
    return ranked.sort_values(keys, na_position='last').head(n)


def _trigger_weights(n: int, sk38_values: list) -> list:
    """
    Inverse-Sk38 weights for trigger locations.
    Lower Sk38 = more unstable = higher probability of being the trigger.
    """
    inv = [1.0 / max(s, 0.01) for s in sk38_values]
    total = sum(inv)
    return [v / total for v in inv]


def _size_weights(size_factors: list) -> list:
    """
    Log-normal weights centred on the median size factor.
    Tails (P10/P90) get less weight than median.
    """
    from scipy.stats import norm
    n = len(size_factors)
    # Map factors to quantiles assuming log-normal
    log_f = [np.log(f) for f in size_factors]
    mu    = np.mean(log_f)
    sigma = max(np.std(log_f), 0.01)
    probs = [norm.pdf(lf, mu, sigma) for lf in log_f]
    total = sum(probs)
    return [p / total for p in probs]


def _depth_weights(depth_pcts: list) -> list:
    """Simple weights: P50 most likely, P10/P90 less so."""
    weights = {10: 0.20, 25: 0.15, 50: 0.60, 75: 0.15, 90: 0.20}
    raw = [weights.get(p, 1.0 / len(depth_pcts)) for p in depth_pcts]
    total = sum(raw)
    return [r / total for r in raw]


def _load_observed_release_polygon(cfg, snapshot_date=None):
    """
    Load the most recent observed release area GeoJSON up to snapshot_date
    and reproject to UTM.  Returns a single shapely Polygon.
    """
    import json
    from shapely.geometry import shape
    from shapely.ops import unary_union
    from pyproj import Transformer

    if snapshot_date:
        gj_paths = cfg.release_geojsons_for_date(snapshot_date)
        gj_path = gj_paths[-1] if gj_paths else None
    else:
        gj_path = cfg.release_geojson if cfg.release_geojson.exists() else None

    if gj_path is None or not gj_path.exists():
        raise FileNotFoundError(
            f"No observed release GeoJSON found"
            + (f" for date ≤ {snapshot_date}" if snapshot_date else "")
        )

    with open(str(gj_path)) as f:
        gj = json.load(f)

    polys  = [shape(feat['geometry']) for feat in gj['features']]
    merged = unary_union(polys)

    import re as _re
    src_epsg = 4326
    crs_node = gj.get('crs', {}).get('properties', {}).get('name', '')
    _m = _re.search(r'EPSG:+(\d+)', crs_node, _re.IGNORECASE)
    if _m:
        src_epsg = int(_m.group(1))

    dst_epsg = RASTER_EPSG
    if src_epsg != dst_epsg:
        t = Transformer.from_crs(f'EPSG:{src_epsg}', f'EPSG:{dst_epsg}', always_xy=True)

        def _reproj(poly):
            from shapely.geometry import Polygon, MultiPolygon
            def ring(r): return [t.transform(x, y) for x, y in r.coords]
            if poly.geom_type == 'Polygon':
                return Polygon(ring(poly.exterior),
                               [ring(i) for i in poly.interiors])
            return MultiPolygon([_reproj(p) for p in poly.geoms])

        utm_poly = _reproj(merged)
    else:
        utm_poly = merged

    if utm_poly.geom_type == 'MultiPolygon':
        utm_poly = max(utm_poly.geoms, key=lambda p: p.area)

    print(f"Observed release polygon: {utm_poly.area:.0f} m²  "
          f"({gj_path.name})")
    return utm_poly


# ---------------------------------------------------------------------------
# Step: zarr_build
# ---------------------------------------------------------------------------

def step_zarr_build(cfg, args):
    """Build or resume Zarr cache from .pro files."""
    from snowpack_io import build_zarr_cache
    build_zarr_cache(
        pro_dir    = args.pro_dir,
        zarr_out   = args.zarr_path,
        batch_size = args.batch_size,
        n_cpus     = args.workers,
        max_layers = args.max_layers,
        rebuild    = getattr(args, 'rebuild', False),
    )


# ---------------------------------------------------------------------------
# Step: analyze
# ---------------------------------------------------------------------------

def step_analyze(cfg, args):
    """
    Snapshot feature extraction.

    Always produces:
        outputs/analysis/all_start_zone_features_YYYY-MM-DD.csv
        outputs/analysis/meloche_features_all_YYYY-MM-DD.csv

    With --plots also produces:
        outputs/plots/release_zone_comparison_YYYY-MM-DD.png
        outputs/plots/release_zone_deviations_YYYY-MM-DD.png
        outputs/plots/meloche_comparison_YYYY-MM-DD.png
    """
    import subprocess

    zarr_path = Path(args.zarr_path)
    if zarr_path.resolve() == Path(cfg.zarr_path).resolve():
        import snowpack_cycle
        try:
            snowpack_cycle.require_no_pending_cycle(cfg)
        except snowpack_cycle.CycleError as exc:
            sys.exit(f"ERROR: {exc}")

    # Skip if both output CSVs exist and are newer than the Zarr store.
    # Pass --force to bypass this check.
    fdir     = features_dir(cfg, args)
    feat_csv = fdir / f"all_start_zone_features_{args.snapshot_date}.csv"
    mel_csv  = fdir / f"meloche_features_all_{args.snapshot_date}.csv"
    if not getattr(args, 'force', False) and feat_csv.exists() and mel_csv.exists():
        zarr_mtime = zarr_path.stat().st_mtime if zarr_path.exists() else 0
        if feat_csv.stat().st_mtime > zarr_mtime and mel_csv.stat().st_mtime > zarr_mtime:
            print(f"  analyze {args.snapshot_date}: CSVs up-to-date, skipping "
                  f"(pass --force to rerun)")
            return

    script = Path(__file__).resolve().parent / "analyze_release_zone.py"
    cmd    = [sys.executable, str(script),
              '--project-dir', str(cfg.project_dir),
              '--pro-dir',     str(args.pro_dir),
              '--zarr-path',   str(args.zarr_path),
              '--snapshot-date', args.snapshot_date,
              '--features-dir', str(fdir)]
    if getattr(args, 'snapshot_ts', None):
        cmd += ['--snapshot-ts', str(args.snapshot_ts)]
    if getattr(args, 'toml', None):
        cmd += ['--toml', str(args.toml)]
    if args.classifier:
        cmd.append('--classifier')
    if getattr(args, 'feature_buffer_m', 0.0):
        cmd += ['--feature-buffer-m', str(args.feature_buffer_m)]
    if args.plots:
        cmd.append('--plots')
    else:
        # Operational mode: skip the ~46k-location release/adjacent/reference
        # group comparison; only produce the two CSVs step_scenarios needs.
        cmd.append('--operational')
    print(f"Running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Scour depth helpers (used by step_scenarios for trigger and domain map)
# ---------------------------------------------------------------------------

def _scour_depth_for_cluster(ds_t, slab_thick_m):
    """
    Extract entrainable scour depth from a single cluster's SNOWPACK profile.

    ds_t         : xarray Dataset for one location × one timestep (squeezed)
    slab_thick_m : slab thickness (m) defining the failure plane, or None to
                   scan from the surface (full-depth entrainable case)

    Returns (hh_depth_m, hh_depth_mid_m, hh_value).
    All three are None on failure (missing variables, empty profile, etc.).
    hh_depth_m     — scour depth from the bottom of the failure plane (m)
    hh_depth_mid_m — scour depth from the midpoint of the failure plane (m)
    hh_value       — SNOWPACK hand hardness of the stopping layer (negative int)
    """
    if 'hand_hardness' not in ds_t:
        return None, None, None

    hh   = ds_t['hand_hardness'].values.ravel()
    # HS and element heights are metres in the Zarr store (xsnow units 'm').
    # This used to treat them as cm: with a slab the failure plane landed at
    # HS - 100 h, below ground, so every scour depth came out 0; without one
    # it was 100x too small (cluster 1772, Jan 18: 0.019 m for 1.9 m).
    hs_m = float(np.nanmean(ds_t['HS'].values))

    z_var = next((v for v in ('height', 'z', 'layer_height') if v in ds_t), None)
    if z_var is None:
        return None, None, None

    z        = ds_t[z_var].values.ravel()
    n_layers = min(len(hh), len(z))
    hh       = hh[:n_layers]
    z        = z[:n_layers]
    has_hh   = ~np.isnan(hh) & (hh < 0)

    # Failure plane: bottom at fp_height_m, midpoint at fp_mid_m
    if slab_thick_m is not None and not np.isnan(slab_thick_m) and slab_thick_m > 0:
        fp_height_m = hs_m - slab_thick_m
        fp_mid_m    = hs_m - slab_thick_m / 2.0
        # Layer index at the failure plane (layers are bottom-to-top)
        fp_layer_idx = 0
        for li in range(n_layers - 1, -1, -1):
            if z[li] <= fp_height_m:
                fp_layer_idx = li
                break
    else:
        # No failure plane — scan from surface (full entrainable depth)
        fp_height_m  = hs_m
        fp_mid_m     = hs_m
        fp_layer_idx = n_layers - 1

    # Scan downward from failure plane for first hard layer (harder than 1F → hh < -3)
    hard_layer_idx = None
    for li in range(fp_layer_idx, -1, -1):
        if has_hh[li] and hh[li] < -3:
            hard_layer_idx = li
            break

    if hard_layer_idx is not None:
        hard_z         = float(z[hard_layer_idx])
        hh_depth_m     = max(fp_height_m - hard_z, 0.0)
        hh_depth_mid_m = max(fp_mid_m    - hard_z, 0.0)
        hh_value       = float(hh[hard_layer_idx])
    else:
        # No hard layer — entrainable to ground
        hh_depth_m     = max(fp_height_m, 0.0)
        hh_depth_mid_m = max(fp_mid_m,    0.0)
        hh_value       = None

    return hh_depth_m, hh_depth_mid_m, hh_value


def _build_scour_depth_map(ds, cluster_map, snap_features, snap_ts):
    """
    Build per-cluster entrainable scour depth for every cluster in the domain.

    Uses the failure-plane scan for clusters with slab_thickness in snap_features
    (start zone); falls back to full-depth scan for path/deposition clusters.

    Returns dict {cid: scour_depth_m (float)}.  Missing or failed clusters are
    absent from the dict (caller should treat absence as NaN in the raster).
    """
    locs = ds.coords['location'].values
    loc_index = {}   # cid -> loc_str
    for loc in locs:
        loc_str = str(loc)
        try:
            cid = int(loc_str.split('_')[-1])
            loc_index[cid] = loc_str
        except ValueError:
            continue

    domain_cids = sorted(int(c) for c in np.unique(cluster_map) if c > 0)
    scour_map   = {}
    n_ok = n_fail = n_missing = 0

    # Pre-compute the snapshot time index and load all variables at that time
    # step into memory once — avoids per-cluster lazy zarr reads (which incur
    # dask scheduler overhead 6500+ times and make this loop very slow).
    tidx_sd  = snapshot_time_index(ds.coords['time'].values, snap_ts)
    sq_dims  = [d for d in ('slope', 'realization') if d in ds.dims]
    import warnings as _warnings
    with _warnings.catch_warnings():
        _warnings.simplefilter('ignore')
        ds_snap = ds.isel(time=tidx_sd).compute()
    if sq_dims:
        ds_snap = ds_snap.squeeze(sq_dims)

    def _process_cluster(cid):
        loc_str = loc_index.get(cid)
        if loc_str is None:
            return cid, None, 'missing'

        slab_thick_m = None
        if cid in snap_features.index and 'slab_thickness' in snap_features.columns:
            st = snap_features.loc[cid, 'slab_thickness']
            if isinstance(st, pd.Series):
                st = st.iloc[0]
            try:
                val = float(st)
                if not np.isnan(val) and val > 0:
                    slab_thick_m = val
            except (ValueError, TypeError):
                pass

        try:
            ds_t = ds_snap.sel(location=loc_str)
            sq   = [d for d in ('slope', 'realization') if d in ds_t.dims]
            if sq:
                ds_t = ds_t.squeeze(sq)
            depth_m, _, _ = _scour_depth_for_cluster(ds_t, slab_thick_m)
            if depth_m is not None and depth_m > 0:
                return cid, depth_m, 'ok'
            return cid, None, 'fail'
        except Exception:
            return cid, None, 'fail'

    from concurrent.futures import ThreadPoolExecutor as _TPE
    with _TPE(max_workers=min(16, len(domain_cids))) as _pool:
        _results = list(_pool.map(_process_cluster, domain_cids))

    for cid, depth, status in _results:
        if status == 'missing':
            n_missing += 1
        elif status == 'ok':
            scour_map[cid] = depth
            n_ok += 1
        else:
            n_fail += 1

    print(f"  Scour depth map: {n_ok} clusters mapped, "
          f"{n_fail} failed/zero, {n_missing} missing from Zarr")
    if scour_map:
        depths = list(scour_map.values())
        print(f"  Scour depth: median={np.median(depths):.2f}m  "
              f"range=[{min(depths):.2f}, {max(depths):.2f}]m")
    return scour_map


# ---------------------------------------------------------------------------
# Step: scenarios
# ---------------------------------------------------------------------------

def frac_outside_mask(poly, mask: np.ndarray, transform) -> float:
    """Share of the polygon's pixels outside the start-zone mask (NaN if no polygon)."""
    import rasterio.features
    if poly is None or poly.is_empty:
        return float('nan')
    inside = rasterio.features.geometry_mask(
        [poly.__geo_interface__], out_shape=mask.shape,
        transform=transform, invert=True)
    n = int(inside.sum())
    return float((inside & ~mask).sum() / n) if n else float('nan')


def _write_arrest_state(state: dict, sdir: Path):
    """Per-scenario arrest diagnostics for the non-legacy arrest modes."""
    sdir.mkdir(parents=True, exist_ok=True)
    rej = pd.Series(state.get('rejected', {}), name='reason', dtype=object) \
        .rename_axis('cluster_id')
    if 'cells' in state:
        pd.DataFrame.from_dict(state['cells'], orient='index') \
            .rename_axis('cluster_id').to_csv(sdir / 'energy_cells.csv')
        rej.to_csv(sdir / 'energy_rejected.csv')
    else:
        rej.to_csv(sdir / 'rejected.csv')


def features_dir(cfg, args) -> Path:
    """Folder of the per-date feature CSVs: --features-dir, else the analysis dir."""
    d = getattr(args, 'features_dir', None)
    return Path(d) if d else cfg.analysis_dir


def resolve_snapshot_ts(cfg, args) -> None:
    """--snapshot-ts, else [scenarios] snapshot_hour_utc on the snapshot date,
    else None (the historical per-read hours)."""
    if getattr(args, 'snapshot_ts', None) is None and cfg.snapshot_hour_utc is not None:
        args.snapshot_ts = f"{args.snapshot_date}T{cfg.snapshot_hour_utc:02d}:00"


def scenario_set_dir(cfg, args) -> Path:
    """Where a scenario set is published: <scenarios_dir>/<date>[/<horizon>].

    <date> is the cycle date when one is given (a forecast cycle groups its
    days T_0..T_n under it), else the snapshot date."""
    base = cfg.scenarios_dir / (getattr(args, 'cycle_date', None) or args.snapshot_date)
    return base / args.forecast_horizon if args.forecast_horizon else base


SHOT_SET_NAME = "shots"


def shot_set_dir(cfg, args) -> Path:
    """Where the forced-shot set is published: <scenarios_dir>/<date>/shots[/<horizon>].

    Under the date folder but not named T_n, so runout_step and runout's
    discover_horizons never take it for a horizon, and handoff.publish of
    the model set at <date>/ carries it across instead of dropping it."""
    base = (cfg.scenarios_dir / (getattr(args, 'cycle_date', None) or args.snapshot_date)
            / SHOT_SET_NAME)
    return base / args.forecast_horizon if args.forecast_horizon else base


def _forcing_shots(cfg, args) -> bool:
    """True when a forced-shot set is to be built: asked for, and the path
    has shot locations. Asked for without any, there is no shot set and no
    shot runout; say so rather than skip it silently."""
    flag = getattr(args, 'force_explosive_triggers', None)
    if flag is None:
        flag = getattr(cfg, 'force_explosive_triggers', False)
    if not flag:
        return False
    if not getattr(cfg, 'explosive_trigger_points', None):
        print("  WARNING: force_explosive_triggers is on but [scenarios] "
              "explosive_trigger_points is empty for this path — no forced-shot "
              "set (and no shot runout) is published")
        return False
    return True


def store_span(zarr_path):
    """(first, last) time in a Zarr store, or None when there is none."""
    if not zarr_path or not Path(zarr_path).exists():
        return None
    import xarray as xr
    t = xr.open_zarr(str(zarr_path))['time'].values
    if len(t) == 0:
        return None
    return pd.Timestamp(t.min()), pd.Timestamp(t.max())


# One SNOWPACK profile step (PROF_DAYS_BETWEEN = 0.25 d).
PROFILE_STEP = pd.Timedelta(hours=6)


def fresh_forecast_span(main_span, fc_span):
    """The forecast store's span if it was built from the current observed
    state, else None.

    The cycle rebuilds the forecast store only when the NWP pass runs; when
    nwp_ingest fails it keeps the previous cycle's, which still covers the
    coming days. A store is current when it starts within one profile step
    after the observed store ends (T_stable).
    """
    if not fc_span or not main_span:
        return None
    if main_span[1] < fc_span[0] <= main_span[1] + PROFILE_STEP:
        return fc_span
    return None


def plan_forecast_days(cycle_date: str, n_days: int, hour: int,
                       main_span, fc_span) -> list:
    """Which store and time each day T+0..T+n-1 of a cycle reads.

    Day n is <cycle_date + n> at <hour> UTC (Ron, 2026-11-04: 18 UTC). It
    reads the observed store when that covers the time (an afternoon survey
    run), else the NWP forecast store. Past the forecast store's end the time
    is clamped to its last step ("if T+2 is past the WRF outlook, use its last
    forward-looking timestamp"); a day whose clamped time is not after the
    previous day's is skipped, as is any day with no store covering it.
    Returns one dict per day: day, label, date, ts, store ('observed' |
    'forecast' | None), clamped, reason.
    """
    base = pd.Timestamp(cycle_date)
    out, prev = [], None
    for n in range(n_days):
        date = (base + pd.Timedelta(days=n)).strftime("%Y-%m-%d")
        ts = pd.Timestamp(f"{date}T{hour:02d}:00")
        d = {"day": n, "label": f"T_{n}", "date": date, "ts": ts,
             "store": None, "clamped": False, "reason": None}
        if main_span and ts <= main_span[1]:
            d["store"] = "observed"
        elif fc_span and ts <= fc_span[1]:
            d["store"] = "forecast"
        elif fc_span and (prev is None or fc_span[1] > prev):
            d.update(ts=fc_span[1], store="forecast", clamped=True)
        else:
            d["reason"] = ("no forecast store" if not fc_span else
                           f"the WRF outlook ends {fc_span[1]:%Y-%m-%dT%H:%M}, "
                           f"not after the previous day")
        if d["store"]:
            prev = d["ts"]
        out.append(d)
    return out


def step_forecast(cfg, args):
    """analyze + scenarios for each day of a forecast cycle (T_0..T_{n-1}).

    The cycle date is the snapshot date. Each day's set goes to
    <scenarios_dir>/<cycle_date>/T_n/. Days read from the forecast store write
    their feature CSVs to <analysis_dir>/forecast/<cycle_date>/T_n/, so the
    observed analysis of that date never finds a forecast's CSVs. A day that
    fails is reported and the next one still runs; the step fails only if no
    day published a set.
    """
    import copy
    if cfg.snapshot_hour_utc is None:
        sys.exit("ERROR: forecast needs [scenarios] snapshot_hour_utc in the TOML")
    n_days = args.days or cfg.forecast_days or 3
    main_span = store_span(cfg.zarr_path)
    fc_span = store_span(cfg.forecast_zarr_path)
    if fc_span and fresh_forecast_span(main_span, fc_span) is None:
        print(f"  WARNING: the forecast store ({fc_span[0]} .. {fc_span[1]}) was not "
              f"built from this cycle (observations end "
              f"{main_span[1] if main_span else '-'}): not used, so only days the "
              f"observations cover are produced")
        fc_span = None
    plan = plan_forecast_days(args.snapshot_date, n_days, cfg.snapshot_hour_utc,
                              main_span, fc_span)
    print(f"\n=== step_forecast  cycle={args.snapshot_date}  days={n_days} ===")
    print(f"  observed store: {main_span[0] if main_span else '-'} .. {main_span[1] if main_span else '-'}")
    print(f"  forecast store: {fc_span[0] if fc_span else '-'} .. {fc_span[1] if fc_span else '-'}")
    published = 0
    for d in plan:
        tag = f"{d['label']} {d['date']}"
        if not d["store"]:
            print(f"\n  {tag}: skipped ({d['reason']})")
            continue
        a = copy.copy(args)
        a.cycle_date = args.snapshot_date
        a.snapshot_date = d["date"]
        a.snapshot_ts = f"{d['ts']:%Y-%m-%dT%H:%M}"
        a.forecast_horizon = d["label"]
        a.store_kind = d["store"]
        a.clamped = d["clamped"]
        if d["store"] == "forecast":
            a.zarr_path = cfg.forecast_zarr_path
            a.features_dir = cfg.analysis_dir / "forecast" / args.snapshot_date / d["label"]
            a.force = True                       # the forecast store is rebuilt every cycle
        else:
            a.zarr_path = cfg.zarr_path
        print(f"\n  {tag}: {d['store']} store at {a.snapshot_ts} UTC"
              + (" (clamped to the end of the WRF outlook)" if d["clamped"] else ""), flush=True)
        try:
            step_analyze(cfg, a)
            step_scenarios(cfg, a)
            published += 1
        except (Exception, SystemExit) as e:
            print(f"  {tag}: FAILED: {e}", flush=True)
    # The forecast store is rebuilt with mode='w' next cycle: keep the
    # timesteps the T_n days were read at (season record, Ron 2026-11-08).
    fc_ts = [d["ts"] for d in plan if d["store"] == "forecast"]
    if fc_ts:
        import metrics
        metrics.keep_forecast_store(cfg, args.snapshot_date, fc_ts)
    if not published:
        sys.exit("ERROR: no forecast day published a scenario set")
    print(f"\n  forecast cycle {args.snapshot_date}: {published} of {n_days} days published")


def step_scenarios(cfg, args):
    """
    Build the scenario set in staging and publish it as a handoff set
    (handoff.py): manifest.json, rename into place, READY last. Runout waits
    for READY, so it never reads a set still being written or replaced. A
    failed build discards its staging and leaves any published set alone.
    """
    import copy
    import handoff
    import calibration
    # [calibration]: c_fit for the BFS arrest; size_scale on every size factor
    # (weights are centred on the median log factor, so they do not change).
    calibration.apply(cfg)
    k = calibration.size_scale(cfg)
    if k != 1.0:
        args = copy.copy(args)
        args.size_factors = [round(f * k, 6) for f in args.size_factors]
    final = scenario_set_dir(cfg, args)
    stage = handoff.staging_dir(final)
    shot_final = shot_set_dir(cfg, args) if _forcing_shots(cfg, args) else None
    shot_stage = None
    if shot_final is not None:
        shot_final.parent.mkdir(parents=True, exist_ok=True)
        shot_stage = handoff.staging_dir(shot_final)
    try:
        if shot_stage is None:
            _build_scenarios(cfg, args, stage)
        else:
            _build_scenarios(cfg, args, stage, shot_dir=shot_stage)
    except BaseException:
        handoff.discard(stage)
        if shot_stage is not None:
            handoff.discard(shot_stage)
        raise
    _publish_set(cfg, args, stage, final)
    print(f"Published scenario set → {final}")
    import metrics
    metrics.cycle_record(cfg, final, funnel=dict(_FUNNEL), near_miss=list(_NEAR_MISS))
    if shot_stage is not None:
        # After the model set: publishing <date>/ moves an existing
        # <date>/shots across, and the shot set then replaces itself in place.
        _publish_set(cfg, args, shot_stage, shot_final)
        print(f"Published forced-shot set → {shot_final}")
        metrics.cycle_record(cfg, shot_final)
    return final


def _calibration_version(cfg):
    import calibration
    return calibration.version(cfg)


def _calibration_knobs(cfg):
    import calibration
    cal = getattr(cfg, 'calibration', None) or {}
    return {k: cal[k] for k in calibration.KNOBS if k in cal} or None


def _publish_set(cfg, args, stage, final):
    import handoff
    meta = stage / "metadata.json"
    meta_d = json.loads(meta.read_text()) if meta.exists() else {}
    n = meta_d.get("n_scenarios", 0)
    keep_old = None
    if final.exists():
        # A rerun replaces what forecasters were shown: keep it (Ron, 11-08).
        import metrics
        try:
            rel = final.relative_to(cfg.scenarios_dir)
            cycle = getattr(args, 'cycle_date', None) or args.snapshot_date
            keep_old = metrics.superseded_dir(cfg, cycle) / "release" / rel
        except (ValueError, Exception) as e:     # noqa: BLE001
            print(f"WARNING: superseded set will not be kept: {e}")
    handoff.publish(stage, final, "release", toml=cfg.toml_path, keep_old=keep_old,
                    path=cfg.slope_name, snapshot_date=args.snapshot_date,
                    horizon=args.forecast_horizon,
                    status="ok" if n else "no_scenarios", n_scenarios=n,
                    trigger_source=meta_d.get("trigger_source"),
                    cycle_date=getattr(args, 'cycle_date', None),
                    snapshot_ts=getattr(args, 'snapshot_ts', None),
                    store=getattr(args, 'store_kind', None),
                    clamped=getattr(args, 'clamped', None),
                    calibration_version=_calibration_version(cfg),
                    calibration=_calibration_knobs(cfg))


# What the last _build_scenarios saw at each trigger gate, for the season
# record (metrics.cycle_record). Printed only until 2026-11-08.
_FUNNEL: dict = {}
_NEAR_MISS: list = []
N_NEAR_MISS = 10


def _near_misses(cids, snap_features, meloche_df, val, mean_slope, sk38_col,
                 min_tau_g, min_slope, max_sk38, min_slab, max_slab):
    """The N_NEAR_MISS lowest-Sk38 start-zone clusters with each gate's value
    and threshold, before any gate. Never raises: it is a record, not a gate."""
    try:
        rows = []
        for cid in cids:
            sk = val(snap_features, cid, sk38_col)
            if np.isnan(sk):
                continue
            rows.append((sk, cid))
        out = []
        for sk, cid in sorted(rows)[:N_NEAR_MISS]:
            r = {'cid': int(cid), 'sk38': round(sk, 3),
                 'tau_g_pa': round(val(meloche_df, cid, 'tau_g'), 1),
                 'slope_deg': round(mean_slope(cid), 1),
                 'slab_m': round(val(snap_features, cid, 'slab_thickness'), 3)}
            try:
                r['s_t'] = round(float(_rg.cluster_tensile_support(cid, meloche_df,
                                                                   snap_features)), 3)
            except Exception:                    # noqa: BLE001
                r['s_t'] = None
            out.append({k: (None if isinstance(v, float) and np.isnan(v) else v)
                        for k, v in r.items()})
        if out:
            out.append({'thresholds': {'sk38_lt': max_sk38, 'tau_g_ge': min_tau_g,
                                       'slope_ge': min_slope,
                                       'slab_m': [min_slab, max_slab], 's_t_ge': 1.0}})
        return out
    except Exception as e:                       # noqa: BLE001
        print(f"WARNING: near-miss record failed: {e}")
        return []


def _build_scenarios(cfg, args, out_dir: Path, shot_dir: Path = None):
    """
    Generate AvaFrame com1DFA release scenario ensemble into out_dir.

    Axes:
      - Trigger location:   top-N weakest Sk38 clusters in start zone
      - Release size:       A_ca × size_factor (Meloche brittle scaling)
      - Slab depth:         SNOWPACK HS × depth_scale at given percentile
    """
    print(f"\n=== step_scenarios  snapshot={args.snapshot_date} ===")
    _FUNNEL.clear()
    _NEAR_MISS.clear()

    # make_release_polygon_2d reads these module flags, so set them before any run.
    _rg.ARREST_MODE  = getattr(args, 'arrest_mode', _rg.ARREST_MODE)
    _rg.MASK_MODE    = getattr(args, 'mask_mode', _rg.MASK_MODE)
    _ep.ENERGY_DELTA = getattr(args, 'energy_delta', _ep.ENERGY_DELTA)
    if getattr(args, 'slab_tensile_gate', None) is not None:
        _rg.USE_SLAB_TENSILE_GATE = args.slab_tensile_gate
    if _rg.USE_SLAB_TENSILE_GATE:
        print(f"  slab tensile-failure gate ON ({_rg.SLAB_TENSILE_SIGMA_COL})")
    if _rg.ARREST_MODE != 'legacy' or _rg.MASK_MODE != 'gate':
        print(f"  arrest_mode={_rg.ARREST_MODE}  mask_mode={_rg.MASK_MODE}  "
              f"energy_delta={_ep.ENERGY_DELTA}")

    # --- Load inputs ---
    dem, transform, crs_wkt, raster_profile = _load_dem(cfg)
    cluster_map = np.load(str(cfg.analysis_dir / "cluster_map.npy"))
    ds          = load_dataset(args.pro_dir, zarr_path=args.zarr_path)

    start_zone_mask = kml_to_mask(cfg.start_zone_kml, dem.shape, transform)

    # --- Load pre-computed analysis outputs ---
    # The analyze step now produces full start zone features by default.
    # Fall back to group-level CSVs for backward compatibility with
    # older analysis outputs.
    fdir         = features_dir(cfg, args)
    feat_csv_all = fdir / f"all_start_zone_features_{args.snapshot_date}.csv"
    feat_csv_grp = fdir / f"release_zone_features_{args.snapshot_date}.csv"
    mel_csv_all  = fdir / f"meloche_features_all_{args.snapshot_date}.csv"
    mel_csv_grp  = fdir / f"meloche_features_{args.snapshot_date}.csv"

    if feat_csv_all.exists():
        feat_csv = feat_csv_all
    elif feat_csv_grp.exists():
        feat_csv = feat_csv_grp
        print(f"  WARNING: using group-level features ({feat_csv_grp.name}). "
              f"Rerun 'analyze' step to generate full start zone coverage.")
    else:
        # analyze writes no features CSV when no start-zone cluster has a
        # snow profile. On bare ground that is the right answer, not an
        # error: publish an empty set that says so, so the run ends DONE and
        # the report shows "no snow cover" (first live night, 2026-11-08).
        # With snow present a missing CSV is still a real fault and fails.
        max_hs = _start_zone_max_hs(ds, cluster_map, start_zone_mask, args)
        if max_hs is not None and max_hs < MIN_RELEASE_DEPTH_M:
            print(f"  No snow cover: start-zone max HS {max_hs:.2f} m "
                  f"< {MIN_RELEASE_DEPTH_M} m — publishing an empty scenario set")
            _write_no_snow_metadata(out_dir, args, max_hs)
            if shot_dir is not None:
                _write_no_snow_metadata(shot_dir, args, max_hs, shots=True)
            return out_dir
        raise FileNotFoundError(
            f"Run 'analyze' step first — missing features CSV"
            + ("" if max_hs is None else
               f" (start-zone max HS {max_hs:.2f} m, so this is not bare ground)"))

    mel_csv = mel_csv_all if mel_csv_all.exists() else mel_csv_grp

    snap_features = pd.read_csv(str(feat_csv), index_col=0)
    snap_features = snap_features[~snap_features.index.duplicated(keep='first')]
    meloche_df    = pd.read_csv(str(mel_csv),  index_col=0) \
                   if mel_csv.exists() else pd.DataFrame()
    if not meloche_df.empty:
        meloche_df = meloche_df[~meloche_df.index.duplicated(keep='first')]
    print(f"  Features: {len(snap_features)} clusters from {feat_csv.name}")
    print(f"  Meloche:  {len(meloche_df)} clusters from {mel_csv.name}")

    if shot_dir is not None:
        _build_shot_set(cfg, args, shot_dir, cluster_map, snap_features,
                        meloche_df, ds, dem, transform, crs_wkt,
                        raster_profile, start_zone_mask)

    # --- Find trigger locations ---
    # Candidate pool: ALL clusters within the start zone that have features.
    # Group assignment (release/adjacent/reference) is for validation
    # comparison, NOT for trigger selection — operationally we don't know
    # which clusters will be in the release area.
    # Pa — minimum gravitational driving shear stress to accept as a trigger.
    # Distinct from snowpack_analysis.TAU_G_FEATURE_FLOOR (50 Pa), which gates
    # whether arrest indices are computed at all; see the note there.
    MIN_TAU_G   = 40.0
    MIN_SLOPE   = 30.0   # degrees — well above the 27° friction angle
    MAX_SK38    = 1.0    # Sk38 >= 1.0 = stable for skier triggering (Schweizer & Jamieson 2007)
    MIN_SLAB_THICKNESS = 0.5  # metres — too thin = D1 at most
    # metres — the deepest slab we expect explosive control to trigger at the
    # basal WL. This is an operational choice, not a physical result, and it
    # binds hard. Sk38 at the WL falls almost monotonically with slab
    # thickness (Spearman -0.93 within the 2026-01-18 pool), so ranking by
    # Sk38 picks the thickest slabs under this cap. 2.0 -> 2.5 m replaced all
    # five triggers on both 2026-01-18 and 2026-03-07. 1.5 m left only two
    # candidates on 2026-03-07 (median slab 1.8 m). Change it only for an
    # operational reason.
    MAX_SLAB_THICKNESS = args.max_slab_thickness

    # All clusters that overlap the start zone mask.
    # Iterate the feature index, not the set: ties that survive the r_c
    # tie-break in _rank_triggers fall back to row order, and set iteration
    # order would hand them to a different cluster than the standalone does.
    sz_cids = set(int(c) for c in np.unique(cluster_map[start_zone_mask])
                  if c > 0)
    candidate_ids = [cid for cid in snap_features.index
                     if cid in sz_cids]
    print(f"  Start zone clusters with features: {len(candidate_ids)}")
    _FUNNEL['start_zone_with_features'] = len(candidate_ids)

    def _scalar_val(df, cid, col):
        """Safely get a scalar from a possibly multi-row index."""
        if cid not in df.index or col not in df.columns:
            return np.nan
        v = df.loc[cid, col]
        if isinstance(v, pd.Series): v = v.iloc[0]
        return float(v)

    MIN_SLOPE_TRIGGER = args.stauchwall_deg + 2.0  # must be above stauchwall

    if not meloche_df.empty and 'tau_g' in meloche_df.columns:
        # Pre-compute per-cluster mean slope from DEM
        slope_grid, _ = compute_slope_aspect(dem)
        def _mean_slope(cid):
            pxs = np.argwhere(cluster_map == cid)
            if len(pxs) == 0: return 0.0
            return float(slope_grid[pxs[:,0], pxs[:,1]].mean())

        # Also filter by Sk38 — Sk38 >= 0.5 is stable terrain
        sk38_col = 'min_sk38'
        _NEAR_MISS.extend(_near_misses(candidate_ids, snap_features, meloche_df,
                                       _scalar_val, _mean_slope, sk38_col,
                                       MIN_TAU_G, MIN_SLOPE_TRIGGER, MAX_SK38,
                                       MIN_SLAB_THICKNESS, MAX_SLAB_THICKNESS))
        candidate_ids = [
            cid for cid in candidate_ids
            if cid in meloche_df.index
            and _scalar_val(meloche_df, cid, 'tau_g') >= MIN_TAU_G
            and not np.isnan(_scalar_val(meloche_df, cid, 'tau_g'))
            and _mean_slope(cid) >= MIN_SLOPE_TRIGGER
            and (cid not in snap_features.index
                 or _scalar_val(snap_features, cid, sk38_col) < MAX_SK38)
        ]
        print(f"  Candidates after tau_g >= {MIN_TAU_G} Pa, "
              f"slope >= {MIN_SLOPE_TRIGGER:.1f}°, "
              f"Sk38 < {MAX_SK38} filter: "
              f"{len(candidate_ids)}")
        _FUNNEL['tau_g_slope_sk38'] = len(candidate_ids)
        candidate_ids = [
            cid for cid in candidate_ids
            if ((_scalar_val(snap_features, cid, 'slab_thickness') >= MIN_SLAB_THICKNESS
                 and _scalar_val(snap_features, cid, 'slab_thickness') <= MAX_SLAB_THICKNESS)
                or np.isnan(_scalar_val(snap_features, cid, 'slab_thickness')))
        ]
        print(f"  Candidates after slab_thickness {MIN_SLAB_THICKNESS}-{MAX_SLAB_THICKNESS}m filter: "
            f"{len(candidate_ids)}")
        _FUNNEL['slab_thickness'] = len(candidate_ids)
    # Slab tensile gate on the trigger: rank only triggers whose slab can carry
    # a crack to r_c, so the ranking still yields n_triggers releases.
    if (_rg.USE_SLAB_TENSILE_GATE and _rg.ARREST_MODE == 'legacy'
            and _rg.SLAB_TENSILE_SIGMA_COL not in snap_features.columns):
        print(f"  WARNING: slab tensile gate is on but the features have no "
              f"'{_rg.SLAB_TENSILE_SIGMA_COL}' column, so it never fires; "
              f"re-run analyze")
    if _rg.USE_SLAB_TENSILE_GATE and _rg.ARREST_MODE == 'legacy':
        candidate_ids = [
            cid for cid in candidate_ids
            if not (_rg.cluster_tensile_support(cid, meloche_df, snap_features) < 1.0)]
        print(f"  Candidates after slab tensile support S_t >= 1: {len(candidate_ids)}")
        _FUNNEL['slab_tensile'] = len(candidate_ids)
    # Optional: restrict to observed release area (validation mode only)
    if args.restrict_to_release_area:
        from snowpack_analysis import geojson_to_mask
        gj_path = cfg.release_geojson
        if gj_path.exists():
            release_area_mask = geojson_to_mask(gj_path, dem.shape, transform)
            candidate_ids = [
                cid for cid in candidate_ids
                if np.any(release_area_mask[cluster_map == cid])]
            print(f"  Candidates after release area restriction: "
                  f"{len(candidate_ids)}")
        else:
            print(f"  Warning: --restrict-to-release-area set but "
                  f"{gj_path} not found — skipping")

    # Elevation filter: triggers must be in upper portion of candidate zone
    # Use P75 in validation mode (release area = known crown terrain)
    # Use P50 in operational mode (full start zone)
    elev_pct = 75 if args.restrict_to_release_area else 50
    if candidate_ids:
        elevs = []
        for cid in candidate_ids:
            pxs = np.argwhere(cluster_map == cid)
            if len(pxs):
                elevs.append((cid, float(dem[pxs[:,0], pxs[:,1]].mean())))
        if elevs:
            elev_vals = [e for _, e in elevs]
            elev_threshold = np.percentile(elev_vals, elev_pct)
            candidate_ids = [cid for cid, e in elevs
                             if e >= elev_threshold]
            print(f"  Candidates after elevation >= P{elev_pct} "
                  f"({elev_threshold:.0f}m) filter: {len(candidate_ids)}")
            _FUNNEL['elevation'] = len(candidate_ids)

    sk38_col = 'min_sk38' if 'min_sk38' in snap_features.columns else 'sk38_min'
    trigger_source = 'model'
    if not candidate_ids:
        # No cluster passed the gates: every thin early-season day does this.
        # Trigger at the slope's explosive-control shot locations instead
        # (Ron, 2026-11-04). They take no gate, rank or Pi1 test and get equal
        # weight. The release model still decides what each one releases: a
        # trigger with no features gets the oriented rectangle with depth
        # 0.8 x HS, and a trigger that fails the slab tensile gate releases
        # nothing. With no shot locations configured, publish an empty set.
        shot_cids = _clusters_at_points(cfg.explosive_trigger_points,
                                        cluster_map, transform)
        triggers = _explosive_triggers(shot_cids, cluster_map, snap_features,
                                       sk38_col)
        if triggers.empty:
            print("  No candidate trigger clusters in the start zone and no "
                  "explosive_trigger_points on the cluster map — "
                  "publishing an empty scenario set")
            return out_dir
        trigger_source = 'explosive_fallback'
        print(f"  No candidate trigger clusters in the start zone — using the "
              f"explosive-control locations {list(triggers.index)}")
    else:
        triggers = _model_triggers(candidate_ids, snap_features, meloche_df,
                                   sk38_col, args, cfg, cluster_map, dem,
                                   transform, start_zone_mask)
    return _write_scenario_set(cfg, args, out_dir, triggers, trigger_source,
                               sk38_col, snap_features, meloche_df, ds,
                               cluster_map, dem, transform, crs_wkt,
                               raster_profile, start_zone_mask)


# metres — a scenario whose release holds less snow than this on average is
# dropped, whatever its trigger. Ron, 2026-11-04: first for the shot-location
# fallback, then for model triggers too (rehearsal 2025-11-30: both model
# triggers fell back to rectangles averaging 0.01-0.03 m). No Jan 18 scenario
# is near it (shallowest 0.79 m of 150).
MIN_RELEASE_DEPTH_M = 0.10
# kg/m^3 — release density when no cluster in the polygon has a slab density
# (the value the old release-group median fell back to).
DEFAULT_SLAB_DENSITY = 280.0
DEFAULT_SLAB_DENSITY_STD = 20.0


def _start_zone_max_hs(ds, cluster_map, start_zone_mask, args):
    """Largest modelled HS (m) over the start-zone clusters at the snapshot,
    or None when the store has no HS or no start-zone cluster."""
    if 'HS' not in ds:
        return None
    sz_ids = set(int(c) for c in np.unique(cluster_map[start_zone_mask]) if c > 0)
    keep = np.array([int(str(n).split('_')[-1]) in sz_ids
                     for n in ds['location'].values])
    if not keep.any():
        return None
    ts = pd.Timestamp(getattr(args, 'snapshot_ts', None) or args.snapshot_date)
    hs = ds['HS'].isel(location=keep).sel(time=ts, method='nearest').values
    hs = np.asarray(hs, dtype=float)
    if not np.isfinite(hs).any():
        return None
    return float(np.nanmax(hs))


def _write_no_snow_metadata(out_dir, args, max_hs, shots=False):
    """metadata.json of an empty set published because there is no snow."""
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {'snapshot_date': args.snapshot_date, 'n_scenarios': 0,
            'n_trigger_locations': 0, 'trigger_source': 'no_snow',
            'n_below_depth_floor': 0, 'max_hs_m': round(max_hs, 3),
            'min_release_depth_m': MIN_RELEASE_DEPTH_M}
    if shots:
        meta['shot_outcomes'] = {}
        meta['p_no_release'] = 1.0
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))


def _release_mean_depth(release_depth: np.ndarray) -> float:
    """Mean depth over the polygon's valid cells, as write_scenario computes it."""
    valid = release_depth[~np.isnan(release_depth)]
    return float(valid.mean()) if valid.size else 0.0


def _polygon_density(polygon, cluster_map, snap_features, transform):
    """(mean, std) slab density of the clusters inside a release polygon.

    The median and spread over clusters, as the release-group density was
    computed, but from this scenario's own polygon. The release group came
    from the observed release GeoJSONs, so on any day with no observed
    avalanche it was empty and the density was NaN, which runout passes to
    com1DFA. Falls back to DEFAULT_SLAB_DENSITY(_STD) when the polygon's
    clusters have no slab density; std falls back when fewer than two do.
    """
    import rasterio.features
    mask = rasterio.features.geometry_mask(
        [polygon.__geo_interface__], out_shape=cluster_map.shape,
        transform=transform, invert=True)
    cids = [int(c) for c in np.unique(cluster_map[mask]) if c > 0]
    if 'slab_density' not in snap_features.columns:
        return DEFAULT_SLAB_DENSITY, DEFAULT_SLAB_DENSITY_STD
    rho = snap_features['slab_density'].reindex(cids).dropna()
    if rho.empty:
        return DEFAULT_SLAB_DENSITY, DEFAULT_SLAB_DENSITY_STD
    std = float(rho.std()) if len(rho) > 1 else DEFAULT_SLAB_DENSITY_STD
    return float(rho.median()), std


def _no_release_reason(cid, meloche_df, snap_features) -> str:
    """Why a trigger gave no polygon: the slab tensile gate, else no propagation."""
    if _rg.USE_SLAB_TENSILE_GATE and _rg.ARREST_MODE == 'legacy':
        s_t = _rg.cluster_tensile_support(cid, meloche_df, snap_features)
        if np.isfinite(s_t) and s_t < 1.0:
            return 'slab_tensile'
    if meloche_df.empty or cid not in meloche_df.index:
        return 'no_arrest_indices'
    return 'no_propagation'


def _clusters_at_points(points, cluster_map, transform) -> list:
    """Cluster id under each (lat, lon) WGS84 point, in order, no duplicates.

    A point off the raster or on a cell with no cluster is reported and
    skipped. Two points in one cluster give one trigger.
    """
    from pyproj import Transformer
    from rasterio.transform import rowcol
    to_xy = Transformer.from_crs(4326, RASTER_EPSG, always_xy=True)
    nrow, ncol = cluster_map.shape
    cids = []
    for lat, lon in points:
        x, y = to_xy.transform(lon, lat)
        cid = 0
        if np.isfinite(x) and np.isfinite(y):      # inf for an impossible lat
            r, c = rowcol(transform, x, y)
            if 0 <= r < nrow and 0 <= c < ncol:
                cid = int(cluster_map[r, c])
        if cid <= 0:
            print(f"  WARNING: shot location ({lat}, {lon}) is on no cluster "
                  f"(E {x:.1f}, N {y:.1f}) — skipped")
            continue
        if cid not in cids:
            cids.append(cid)
    return cids


def _explosive_triggers(cids, cluster_map, snap_features, sk38_col):
    """Fallback triggers: the configured shot locations present on the map.

    Returns their feature rows in the configured order, all-NaN for a
    cluster with no features, and always with the Sk38 column.
    """
    on_map = set(int(c) for c in np.unique(cluster_map) if c > 0)
    missing = [c for c in cids if c not in on_map]
    if missing:
        print(f"  WARNING: shot-location clusters {missing} are not on "
              f"the cluster map — skipped")
    keep = [int(c) for c in dict.fromkeys(cids) if c in on_map]
    triggers = snap_features.reindex(keep)
    if sk38_col not in triggers.columns:
        triggers[sk38_col] = np.nan
    return triggers


def _build_shot_set(cfg, args, shot_dir, cluster_map, snap_features,
                    meloche_df, ds, dem, transform, crs_wkt, raster_profile,
                    start_zone_mask):
    """The forced-shot scenario set, written into shot_dir.

    A shot always initiates, so the shot clusters take none of the trigger
    gates (tau_g, slope, Sk38, slab thickness, elevation, Pi1, ranking). What
    stays is everything that decides propagation: the slab tensile gate on
    the trigger, the BFS arrest, the start-zone and Stauchwall clip, and the
    0.10 m release-depth floor. There is no rectangle fallback: a shot whose
    BFS does not propagate releases nothing, and metadata.json's
    shot_outcomes says so per shot.
    """
    print(f"\n--- Forced explosive triggers → {shot_dir} ---")
    shot_dir.mkdir(parents=True, exist_ok=True)
    points = list(cfg.explosive_trigger_points)
    cids = _clusters_at_points(points, cluster_map, transform)
    sk38_col = 'min_sk38' if 'min_sk38' in snap_features.columns else 'sk38_min'
    triggers = _explosive_triggers(cids, cluster_map, snap_features, sk38_col)
    if triggers.empty:
        print("  No shot location is on the cluster map — empty shot set")
        (shot_dir / "metadata.json").write_text(json.dumps(
            {'snapshot_date': args.snapshot_date, 'n_scenarios': 0,
             'n_trigger_locations': 0, 'trigger_source': 'explosive_forced',
             'n_below_depth_floor': 0, 'shot_outcomes': {}}, indent=2))
        return
    _write_scenario_set(cfg, args, shot_dir, triggers, 'explosive_forced',
                        sk38_col, snap_features, meloche_df, ds, cluster_map,
                        dem, transform, crs_wkt, raster_profile,
                        start_zone_mask, rectangle_fallback=False)


def _model_triggers(candidate_ids, snap_features, meloche_df, sk38_col, args,
                    cfg, cluster_map, dem, transform, start_zone_mask):
    """Pi1 gate, Sk38 ranking and the propagation check on the candidates."""
    cand_df  = snap_features.loc[candidate_ids].copy()

    # Join Pi1 from meloche features for propagation gate.
    # A trigger that can't propagate (low Pi1) produces a degenerate release —
    # good initiation potential (low Sk38) alone is not sufficient.
    if not meloche_df.empty and 'Pi1_elastic' in meloche_df.columns:
        pi1_vals = {}
        for cid in cand_df.index:
            if cid in meloche_df.index:
                v = meloche_df.loc[cid, 'Pi1_elastic']
                pi1_vals[cid] = float(v.iloc[0]) if isinstance(v, pd.Series) else float(v)
        cand_df['Pi1_elastic'] = pd.Series(pi1_vals)

        pi1_valid = cand_df['Pi1_elastic'].dropna()
        if len(pi1_valid) > 0:
            pi1_median = float(pi1_valid.median())
            pi1_mean   = float(pi1_valid.mean())
            pre_gate   = cand_df.dropna(subset=[sk38_col]).shape[0]
            cand_df    = cand_df[cand_df['Pi1_elastic'] >= pi1_median]
            print(f"  Pi1 propagation gate (>= median {pi1_median:.2f}, "
                  f"mean {pi1_mean:.2f}): "
                  f"{pre_gate} → {cand_df.shape[0]} candidates")

    triggers = _rank_triggers(cand_df, args.n_triggers, sk38_col)

    # Validate triggers: drop any that can't produce a release polygon
    # Test with median size_factor — if it fails at median, it fails at all
    if not args.use_observed_release and not args.no_propagation:
        valid_triggers = []
        _mid_sf = args.size_factors[len(args.size_factors) // 2]
        for t_cid_check in triggers.index:
            _a_ca_check = 50.0
            if 'A_ca_brittle' in meloche_df.columns and t_cid_check in meloche_df.index:
                _v = meloche_df.loc[t_cid_check, 'A_ca_brittle']
                if isinstance(_v, pd.Series): _v = _v.iloc[0]
                if not np.isnan(float(_v)):
                    _a_ca_check = min(float(_v), 300.0)
            _poly_check = make_release_polygon_2d(
                trigger_cluster_id=t_cid_check,
                A_ca=_a_ca_check,
                meloche_df=meloche_df,
                cluster_map=cluster_map,
                dem=dem,
                transform=transform,
                start_zone_mask=start_zone_mask,
                snap_features=snap_features,
                size_factor=_mid_sf,
                stauchwall_deg=args.stauchwall_deg,
                max_clusters=args.max_clusters,
                use_propagation=True,
            )
            if _poly_check is not None:
                valid_triggers.append(t_cid_check)
            else:
                print(f"  Dropped trigger cid={t_cid_check} — "
                      f"propagate_release returned None at "
                      f"size_factor={_mid_sf}")
        if valid_triggers:
            triggers = triggers.loc[valid_triggers]
        else:
            print("  WARNING: all triggers failed propagation — "
                  "keeping original set with rectangle fallback")
    return triggers


def _write_scenario_set(cfg, args, out_dir, triggers, trigger_source,
                        sk38_col, snap_features, meloche_df, ds, cluster_map,
                        dem, transform, crs_wkt, raster_profile,
                        start_zone_mask, rectangle_fallback=True):
    """Write the scenario ensemble for the chosen triggers into out_dir.

    rectangle_fallback=False (forced shots): a trigger whose BFS does not
    propagate gives no polygon. Each trigger's outcome is recorded in
    metadata.json's shot_outcomes when it is False, and the weights are
    absolute: each shot holds 1/N over its size x depth grid, nothing is
    renormalised, and what is not released is p_no_release (Ron, 2026-11-06)."""
    print(f"Trigger clusters ({trigger_source}): {list(triggers.index)} "
          f"Sk38={list(triggers[sk38_col].round(3))}")

    # --- Output directory (staging; step_scenarios publishes it) ---
    (out_dir / "scenarios").mkdir(exist_ok=True)

    # Write trigger locations GeoJSON
    write_trigger_locations(
        triggers, cluster_map, transform,
        out_dir / "trigger_locations.geojson")

    # --- Slab depth grid for snapshot ---
    full_depth = depth_from_snowpack(cluster_map, ds, args.snapshot_date,
                                     snap_features=snap_features,
                                     snapshot_ts=getattr(args, 'snapshot_ts', None))

    # --- Domain-wide entrainable scour depth raster ---
    # Run the failure-plane-to-hard-layer scan for every cluster in the domain
    # and write as scour_depth.tif alongside depth.tif in the output directory.
    # com1DFA can use this as a spatially variable entrainable-depth input
    # (each cell caps how deep the flowing avalanche can scour old snow).
    print(f"\nBuilding domain scour depth map ...")
    snap_ts_sd = np.datetime64(getattr(args, 'snapshot_ts', None)
                               or f"{args.snapshot_date}T18:00")
    scour_depth_map = _build_scour_depth_map(ds, cluster_map, snap_features, snap_ts_sd)

    scour_raster = np.full(cluster_map.shape, np.nan, dtype=np.float32)
    for cid, depth in scour_depth_map.items():
        scour_raster[cluster_map == cid] = float(depth)

    scour_tif_path = out_dir / "scour_depth.tif"
    _sd_profile = raster_profile.copy()
    _sd_profile.update(dtype='float32', count=1, nodata=float('nan'),
                       compress='lzw', predictor=2)
    with rasterio.open(str(scour_tif_path), 'w', **_sd_profile) as _dst:
        _dst.write(scour_raster[np.newaxis, ...])
    print(f"  Written: {scour_tif_path}")

    # --- Precompute weights per axis ---
    sk38_vals     = list(triggers[sk38_col])
    # Shot locations carry no instability ranking (Sk38 is often NaN on a
    # thin pack), so each gets the same weight.
    t_weights     = (_trigger_weights(len(triggers), sk38_vals)
                     if trigger_source == 'model'
                     else [1.0 / len(triggers)] * len(triggers))
    s_weights     = _size_weights(args.size_factors)
    d_weights     = _depth_weights(args.depth_pcts)

    # --- Load observed release polygon if validation mode ---
    observed_polygon = None
    if args.use_observed_release:
        observed_polygon = _load_observed_release_polygon(cfg, args.snapshot_date)
        print("Mode: OBSERVED RELEASE (validation — flow model only)")
    else:
        print("Mode: MELOCHE-DERIVED RELEASE (full chain)")

    # --- Generate scenarios ---
    rows            = []
    n_below_floor   = 0
    weight_map      = {}
    a_ca_list       = []
    median_polygons = []   # (polygon, size_factor) for comparison plot
    most_likely     = {'weight': -1.0, 'scenario_id': None, 'polygon': None,
                       'size_factor': None, 'trigger': None}  # highest-weight scenario

    outcomes = {}   # per trigger: scenarios kept / below floor / why none
    for t_rank, (t_cid, _) in enumerate(triggers.iterrows()):
        outcome = outcomes.setdefault(str(int(t_cid)), {
            'n_scenarios': 0, 'n_below_depth_floor': 0, 'n_no_release': 0,
            'no_release_reason': None})
        # A_ca from Meloche features, fallback to 50m
        A_ca    = 50.0   # default fallback (m)
        A_CA_MAX = 300.0 # cap — start zone is ~250m wide; beyond this
                         # the scaling law is unreliable (very low tau_g)
        # No column at all when no cluster reached TAU_G_FEATURE_FLOOR.
        if 'A_ca_brittle' in meloche_df.columns and t_cid in meloche_df.index:
            val = meloche_df.loc[t_cid, 'A_ca_brittle']
            if isinstance(val, pd.Series):
                val = val.iloc[0]
            if not np.isnan(float(val)):
                A_ca = min(float(val), A_CA_MAX)
        a_ca_list.append(A_ca)

        # Extract scour depth at trigger cluster from SNOWPACK profile
        hh_depth_m, hh_depth_mid_m, hh_value = None, None, None
        t_loc_str = next(
            (str(loc) for loc in ds.coords['location'].values
             if int(str(loc).split('_')[-1]) == t_cid), None)
        if t_loc_str is None:
            print(f"    scour_depth: trigger cid={t_cid} not found in Zarr")
        else:
            try:
                tidx  = snapshot_time_index(ds.coords['time'].values, snap_ts_sd)
                ds_t  = ds.sel(location=t_loc_str).isel(time=tidx)
                sq    = [d for d in ('slope', 'realization') if d in ds_t.dims]
                if sq:
                    ds_t = ds_t.squeeze(sq)

                slab_thick_m = None
                if t_cid in snap_features.index and \
                   'slab_thickness' in snap_features.columns:
                    st = snap_features.loc[t_cid, 'slab_thickness']
                    if isinstance(st, pd.Series):
                        st = st.iloc[0]
                    try:
                        slab_thick_m = float(st)
                    except (ValueError, TypeError):
                        pass

                hh_depth_m, hh_depth_mid_m, hh_value = \
                    _scour_depth_for_cluster(ds_t, slab_thick_m)

                if hh_depth_m is not None:
                    hs_m = float(np.nanmean(ds_t['HS'].values))
                    print(f"    scour_depth trigger cid={t_cid}: "
                          f"HS={hs_m:.2f}m  "
                          f"slab={slab_thick_m if slab_thick_m is None else round(slab_thick_m, 2)}m  "
                          f"scour={hh_depth_m:.2f}m  "
                          f"HH={hh_value}")
                else:
                    print(f"    scour_depth: extraction failed for "
                          f"trigger cid={t_cid}")

            except Exception as e:
                print(f"    scour_depth extraction failed for "
                      f"cid={t_cid}: {e}")

        for s_idx, size_f in enumerate(args.size_factors):
            for d_idx, d_pct in enumerate(args.depth_pcts):

                # Build release polygon
                if args.use_observed_release:
                    # Use observed Jan 18 GeoJSON — validation mode
                    # Still scale by size_factor to bracket uncertainty
                    from shapely import affinity
                    centroid = observed_polygon.centroid
                    polygon  = affinity.scale(
                        observed_polygon,
                        xfact=size_f, yfact=size_f,
                        origin=centroid)
                else:
                    arrest_state = {} if _rg.ARREST_MODE != 'legacy' else None
                    polygon = make_release_polygon_2d(
                        trigger_cluster_id = t_cid,
                        A_ca               = A_ca,
                        meloche_df         = meloche_df,
                        cluster_map        = cluster_map,
                        dem                = dem,
                        transform          = transform,
                        start_zone_mask    = start_zone_mask,
                        snap_features      = snap_features,
                        size_factor        = size_f,
                        stauchwall_deg     = args.stauchwall_deg,
                        max_clusters       = args.max_clusters,
                        use_propagation    = not args.no_propagation,
                        state_out          = arrest_state,
                        rectangle_fallback = rectangle_fallback,
                    )

                if polygon is None:
                    print(f"  Skip scenario (degenerate polygon): "
                          f"trigger={t_cid} size_f={size_f}")
                    outcome['n_no_release'] += 1
                    outcome['no_release_reason'] = _no_release_reason(
                        t_cid, meloche_df, snap_features)
                    continue

                # Scale depth by percentile
                depth_scale  = cfg.depth_scales.get(d_pct, 1.0)
                scaled_depth = full_depth * depth_scale
                release_depth = rasterize_release_polygon(
                    polygon, scaled_depth, dem.shape, transform)

                # On a thin or patchy pack a release polygon can hold almost
                # no snow: shot locations on 11-27 (0-3 mm), and model
                # triggers whose rectangle fallback spans thin cells on 11-30
                # (10-30 mm). Not a release, and not worth a com1DFA run
                # (Ron, 2026-11-04).
                if _release_mean_depth(release_depth) < MIN_RELEASE_DEPTH_M:
                    n_below_floor += 1
                    outcome['n_below_depth_floor'] += 1
                    continue

                density_mean, density_std = _polygon_density(
                    polygon, cluster_map, snap_features, transform)

                # Scenario weight = trigger × size × depth
                weight = t_weights[t_rank] * s_weights[s_idx] * d_weights[d_idx]

                scenario_id = f"scenario_{len(rows)+1:03d}"
                # Capture reference polygon ONCE per trigger
                # Use middle size_factor in the list (closest to 1.0)
                _mid = args.size_factors[len(args.size_factors) // 2]
                if abs(size_f - _mid) < 1e-9 and d_idx == 0:
                    label = (f"T{t_rank+1}  cid={t_cid}  "
                             f"Sk38={sk38_vals[t_rank]:.2f}  "
                             f"Aca={A_ca:.0f}m")
                    median_polygons.append((polygon, size_f, label))

                row = write_scenario(
                    scenario_dir       = out_dir / "scenarios",
                    scenario_id        = scenario_id,
                    release_polygon    = polygon,
                    depth_raster       = release_depth,
                    dem_shape          = dem.shape,
                    transform          = transform,
                    crs_wkt            = crs_wkt,
                    density_mean       = density_mean,
                    density_std        = density_std,
                    trigger_cluster_id = t_cid,
                    A_ca               = A_ca,
                    size_factor        = size_f,
                    depth_percentile   = d_pct,
                    weight             = weight,
                    mu                 = args.mu,
                    xi                 = args.xi,
                    profile            = raster_profile,
                    scour_depth_m         = hh_depth_m,
                    scour_depth_mid_m     = hh_depth_mid_m,
                    hand_hardness_value   = hh_value,
                    ensemble_weight_total = None if rectangle_fallback else 1.0,
                )
                # Augment row with trigger rank and sk38
                row['density_kgm3'] = round(density_mean, 1)
                row['sk38_rank'] = t_rank + 1
                row['min_sk38']  = float(sk38_vals[t_rank])
                if _rg.MASK_MODE == 'diagnostic':
                    row['frac_outside_mask'] = frac_outside_mask(
                        polygon, start_zone_mask, transform)
                if not args.use_observed_release and arrest_state:
                    _write_arrest_state(arrest_state,
                                        out_dir / "scenarios" / scenario_id)
                rows.append(row)
                outcome['n_scenarios'] += 1
                weight_map[scenario_id] = weight

                # Track the highest-weight scenario for the comparison plot.
                # (argmax is unaffected by the later normalisation.)
                if weight > most_likely['weight']:
                    most_likely = {'weight': weight, 'scenario_id': scenario_id,
                                   'polygon': polygon, 'size_factor': size_f,
                                   'trigger': t_cid}

    if n_below_floor:
        print(f"  Dropped {n_below_floor} scenarios ({trigger_source} triggers) "
              f"with mean release depth < {MIN_RELEASE_DEPTH_M} m")
    if not rows and n_below_floor:
        print("  No release holds enough snow — publishing an empty scenario set")
    elif not rows:
        print("WARNING: No scenarios generated — check trigger clusters and Meloche features")
    if not rows:
        # Still record why the set is empty: the manifest reads this.
        empty = {'snapshot_date': args.snapshot_date, 'n_scenarios': 0,
                 'n_trigger_locations': len(triggers),
                 'trigger_source': trigger_source,
                 'n_below_depth_floor': n_below_floor}
        if not rectangle_fallback:
            empty['shot_outcomes'] = outcomes
            empty['p_no_release'] = 1.0
        (out_dir / "metadata.json").write_text(json.dumps(empty, indent=2))
        return out_dir

    # Normalise weights to sum to 1, except forced shots: their weights are
    # absolute (1/N per shot), and the share that did not release is
    # P(no release), not something to spread over the shots that did.
    total_w = sum(weight_map.values())
    if rectangle_fallback:
        weight_map = {k: v / total_w for k, v in weight_map.items()}
    p_no_release = max(0.0, 1.0 - total_w)
    for row in rows:
        row['weight'] = weight_map.get(row['scenario_id'], row['weight'])

    # Write ensemble outputs
    write_scenario_weights(weight_map, out_dir / "scenario_weights.json",
                           normalise=rectangle_fallback)
    write_summary_csv(rows, out_dir / "summary.csv",
                      extra_cols=('frac_outside_mask',) if _rg.MASK_MODE == 'diagnostic' else ())
    write_metadata(
        out_path          = out_dir / "metadata.json",
        snapshot_date     = args.snapshot_date,
        n_scenarios       = len(rows),
        n_triggers        = len(triggers),
        trigger_source    = trigger_source,
        n_below_depth_floor = n_below_floor,
        size_factors      = args.size_factors,
        depth_percentiles = args.depth_pcts,
        mu                = args.mu,
        xi                = args.xi,
        a_ca_stats        = {
            'mean_m':   round(float(np.mean(a_ca_list)), 1),
            'min_m':    round(float(np.min(a_ca_list)),  1),
            'max_m':    round(float(np.max(a_ca_list)),  1),
        },
    )
    if not rectangle_fallback:
        meta_p = out_dir / "metadata.json"
        meta_d = json.loads(meta_p.read_text())
        meta_d['shot_outcomes'] = outcomes
        meta_d['p_no_release'] = round(p_no_release, 6)
        meta_p.write_text(json.dumps(meta_d, indent=2))

    # --- Release comparison plot: written on every scenarios run ---
    import traceback
    print(f"\nGenerating release comparison plot...")
    print(f"  median_polygons collected: {len(median_polygons)}")
    mode_str  = "observed_" if args.use_observed_release else ""
    plot_path = out_dir / f"release_comparison_{mode_str}{args.snapshot_date}_physical_model.png"
    print(f"  Output path: {plot_path}")
    try:
        # A live date has no observed release: plot the model alone. This
        # used to raise, so every operational run logged a traceback and
        # wrote no plot.
        try:
            obs_poly = _load_observed_release_polygon(cfg, args.snapshot_date)
        except FileNotFoundError:
            obs_poly = None
            print("  No observed release for this date: plotting the model only")
        poly_pairs = [(p, sf) for p, sf, _ in median_polygons]
        t_labels   = [lbl for _, _, lbl in median_polygons]

        # Compute trigger centroids from cluster_map (stable position)
        t_centroids = []
        for t_cid_plot in triggers.index:
            px = np.argwhere(cluster_map == t_cid_plot)
            if len(px):
                cx = transform.c + px[:, 1].mean() * transform.a
                cy = transform.f + px[:, 0].mean() * transform.e
                t_centroids.append((cx, cy))
            else:
                t_centroids.append(None)

        ml_poly  = most_likely['polygon']
        ml_label = None
        if most_likely['scenario_id'] is not None:
            ml_label = (f"Most likely: {most_likely['scenario_id']} "
                        f"(p={weight_map.get(most_likely['scenario_id'], 0.0):.3f}, "
                        f"sf={most_likely['size_factor']:.2f})")

        plot_release_comparison(
            meloche_polygons = poly_pairs,
            observed_polygon = obs_poly,
            dem              = dem,
            transform        = transform,
            start_zone_mask  = start_zone_mask,
            trigger_labels   = t_labels,
            trigger_centroids = t_centroids,
            most_likely_polygon = ml_poly,
            most_likely_label   = ml_label,
            out_path         = plot_path,
            title            = (f"Release polygons — {args.snapshot_date} | {cfg.slope_name}"
                               + (" | observed release" if obs_poly is not None else "")
                               + "\nBlue = Model-derived  "
                               + ("Red = Observed  " if obs_poly is not None else "")
                               + "Green = Start zone  Black dashed = Most likely"),
        )
    except Exception:
        print("WARNING: comparison plot failed — full traceback:")
        traceback.print_exc()

    print(f"\n{len(rows)} scenarios written → {out_dir}")
    print(f"  Triggers: {len(triggers)}  "
          f"Size factors: {len(args.size_factors)}  "
          f"Depth pcts: {len(args.depth_pcts)}")
    print(f"  A_ca range: {min(a_ca_list):.1f}–{max(a_ca_list):.1f} m")
    _rho = [r['density_kgm3'] for r in rows if 'density_kgm3' in r]
    if _rho:
        print(f"  Density (per release polygon): median {np.median(_rho):.1f}, "
              f"range {min(_rho):.1f}–{max(_rho):.1f} kg/m³")
    return out_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        description="Post-SNOWPACK analysis pipeline (Pipeline B)")
    parser.add_argument('step',
        choices=['zarr_build', 'analyze', 'scenarios', 'forecast', 'all'],
        help="Pipeline step to run")

    # Common
    parser.add_argument('--project-dir', type=Path, default=Path('.'))
    parser.add_argument('--pro-dir',     type=Path, default=None)
    parser.add_argument('--zarr-path',   type=Path, default=None)
    parser.add_argument('--snapshot-date', default=None,
                        help='SNOWPACK snapshot date YYYY-MM-DD (default: the '
                             'replay clock $AVACHAIN_AS_OF, else today UTC)')
    parser.add_argument('--snapshot-ts', default=None,
                        help='Snapshot time YYYY-MM-DDTHH:MM (UTC) for every read '
                             '(default: [scenarios] snapshot_hour_utc on the snapshot '
                             'date; without it the historical per-read hours)')
    parser.add_argument('--features-dir', type=Path, default=None,
                        help='Folder of the per-date feature CSVs (default: the '
                             'analysis dir). Forecast days use their own.')

    # zarr_build
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--workers',    type=int, default=31)
    parser.add_argument('--max-layers', type=int, default=338)
    parser.add_argument('--rebuild', action='store_true',
                        help='Overwrite an existing Zarr store instead of '
                             'resuming into it. Required if the store is '
                             'unreadable — zarr_build refuses to append to a '
                             'store whose resume state it cannot recover.')

    # analyze
    parser.add_argument('--classifier', action='store_true')
    parser.add_argument('--force', action='store_true',
                        help='Force rerun of analyze even if CSVs are newer than the Zarr.')
    parser.add_argument('--plots', action='store_true',
                        help='Generate research/diagnostic plots '
                             '(analyze: comparison, deviations, Meloche; '
                             'the scenarios release comparison is always written). '
                             'Off by default.')

    # forecast (analyze + scenarios per forecast day)
    parser.add_argument('--days', type=int, default=None,
                        help='forecast: number of days T_0.. (default [scenarios] '
                             'forecast_days, else 3)')

    # scenarios
    parser.add_argument('--forecast-horizon', default=None,
                        help='Forecast horizon label (e.g. T_1, T_2). '
                             'When set, scenarios are written to '
                             'outputs/scenarios/YYYY-MM-DD/<horizon>/. '
                             'Omit for T+0 (writes directly to YYYY-MM-DD/).')
    parser.add_argument('--n-triggers',    type=int,   default=None)
    parser.add_argument('--max-clusters',  type=int,   default=None,
                        help='BFS region-size cap (default: config max_bfs_clusters = 2000). '
                             'A [CAP-BOUND] warning is printed when it binds.')
    parser.add_argument('--size-factors',  type=float, nargs='+',
                        default=None,
                        help='Size factor multipliers for arrest thresholds. '
                             '>1.0 = relaxed = larger release; '
                             '<1.0 = tighter = smaller release; '
                             '1.0 = baseline.')
    parser.add_argument('--depth-pcts',    type=int,   nargs='+',
                        default=None)
    parser.add_argument('--mu',            type=float, default=None)
    parser.add_argument('--xi',            type=float, default=None)
    parser.add_argument('--stauchwall-deg',type=float, default=None)
    parser.add_argument('--max-slab-thickness', type=float, default=2.0,
                        help='Maximum slab thickness to accept as a trigger (m). '
                             'Default 2.0, sized for explosive control. Use 1.5 for '
                             'skier-triggered scenarios (skier stress is negligible '
                             'below ~1.5 m) or 10.0 for natural triggers.')
    parser.add_argument('--no-propagation', action='store_true',
                        help='Skip BFS propagation, use rectangle fallback')
    parser.add_argument('--restrict-to-release-area', action='store_true',
                        help='Restrict trigger candidates to clusters inside '
                             'the observed release area GeoJSON. Use for '
                             'validation only — in operational mode this '
                             'file does not exist.')
    parser.add_argument('--use-observed-release', action='store_true',
                        help='Use observed Jan 18 release GeoJSON as polygon '
                             'instead of Meloche-derived geometry. '
                             'Validates flow model independently of '
                             'release area estimation.')
    parser.add_argument('--toml', default=None,
                        help="Path to slope_config.toml; overrides --project-dir when provided")
    parser.add_argument('--feature-buffer-m', type=float, default=0.0,
                        help='analyze: also extract features within this distance (m) '
                             'of the start zone, for --mask-mode diagnostic (default 0).')
    parser.add_argument('--arrest-mode', default=_rg.ARREST_MODE,
                        choices=('legacy', 'energy', 'slope_only', 'stability_connected'),
                        help=f"BFS arrest rule (default: {_rg.ARREST_MODE}). 'energy' is the "
                             "path-energy hypothesis (energy_path.py) and writes "
                             "energy_cells.csv / energy_rejected.csv per scenario; "
                             "'slope_only' and 'stability_connected' are mechanics-free "
                             "baselines (baselines.py) and write rejected.csv.")
    parser.add_argument('--mask-mode', default=_rg.MASK_MODE, choices=('gate', 'diagnostic'),
                        help=f"Start-zone KML use (default: {_rg.MASK_MODE}). 'diagnostic' "
                             "runs the BFS without it and adds frac_outside_mask to summary.csv.")
    parser.add_argument('--slab-tensile-gate', action=argparse.BooleanOptionalAction,
                        default=None,
                        help='legacy mode: also arrest where the slab breaks in tension '
                             'before the crack reaches r_c (Schweizer et al. 2014; '
                             'default: release_geometry.USE_SLAB_TENSILE_GATE, on).')
    parser.add_argument('--force-explosive-triggers', action=argparse.BooleanOptionalAction,
                        default=None,
                        help='scenarios: also publish a forced-shot set at '
                             '<date>/shots[/T_n] with a trigger at every '
                             '[scenarios] explosive_trigger_points location, no '
                             'initiation gate, propagation physics only (default: '
                             '[scenarios] force_explosive_triggers, off).')
    parser.add_argument('--energy-delta', type=float, default=_ep.ENERGY_DELTA,
                        help=f"delta in the energy-mode G_c (default: {_ep.ENERGY_DELTA}; "
                             "R was fitted at 0). The Eq. 20 delta is unchanged.")
    return parser


def main():
    args = build_parser().parse_args()
    args.snapshot_date = resolve_snapshot_date(args.snapshot_date)
    if args.toml:
        cfg = ProjectConfig.from_toml(args.toml)
    else:
        cfg  = ProjectConfig(project_dir=args.project_dir)
    cfg.ensure_dirs()
    resolve_snapshot_ts(cfg, args)

    # Fill None args from config defaults
    if args.pro_dir is None:
        args.pro_dir = cfg.pro_dir
    if args.zarr_path is None:
        args.zarr_path = cfg.zarr_path
    if args.n_triggers is None:
        args.n_triggers = cfg.n_triggers
    if args.max_clusters is None:
        args.max_clusters = cfg.max_bfs_clusters
    if args.size_factors is None:
        args.size_factors = cfg.size_factors
    if args.depth_pcts is None:
        args.depth_pcts = cfg.depth_percentiles
    if args.mu is None:
        args.mu = cfg.mu
    if args.xi is None:
        args.xi = cfg.xi
    if args.stauchwall_deg is None:
        args.stauchwall_deg = cfg.stauchwall_deg

    if args.step in ('zarr_build', 'all'):
        step_zarr_build(cfg, args)

    if args.step in ('analyze', 'all'):
        step_analyze(cfg, args)

    if args.step in ('scenarios', 'all'):
        step_scenarios(cfg, args)

    if args.step == 'forecast':
        step_forecast(cfg, args)


if __name__ == '__main__':
    main()
    
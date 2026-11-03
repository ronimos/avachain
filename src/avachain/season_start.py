"""
season_start.py — start a slope's season from bare ground, before any survey.

    python season_start.py --toml T bootstrap [--until YYYY-MM-DDTHH:MM]

Needs: the cluster map (cluster_carry.py, or last season's), [snowpack]
season_start and start_from_bare_ground = true in the TOML, the station
cache filled from season_start (aws_ingest.py --since <season_start>, with
[weather] source = "cache"), and an empty SMET dir and pro_dir. It:

  1. writes station-only hourly HS grids from season_start (HS 0 on every
     cell) to the last station hour: spatial_model.gap_fill_station_only, the
     same rule smet_append uses between surveys. Every cluster gets the
     station's HS until the first survey says otherwise;
  2. writes the cluster SMETs from them (forcing_pipeline.step_smet);
  3. runs SNOWPACK from bare ground to the first profile step and saves that
     state as the survey anchor. A survey cycle re-runs from the survey
     anchor, so the season's first survey re-runs the whole early season
     from bare ground, on the forcing that gap_fill's bare-ground segment
     rebuilds from that survey (forcing_pipeline._gap_fill_bare_ground);
  4. runs on to the last observation, saves the daily anchor and builds the
     Zarr store. run_daily_<slope>.sh carries on from there.

Refuses to run over an existing season's state: archive the old pro_dir,
SMET dir and Zarr first.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import ProjectConfig, replay_as_of
from forcing_pipeline import BOOTSTRAP_GRID_GLOB, bare_ground_start_ts
from snowpack_cycle import (OUTPUT_STEP, TS_FMT, CycleError, _require_coherent,
                            _save_as, _sno_files, _write_survey_anchor,
                            on_output_grid, run_snowpack, SURVEY_ANCHOR_META)


def _refuse_existing_state(cfg: ProjectConfig) -> None:
    left = []
    if any(cfg.smet_dir.glob("cluster_*.smet")):
        left.append(f"SMET files in {cfg.smet_dir}")
    if cfg.pro_dir.exists() and (any(cfg.pro_dir.glob("cluster_*.sno"))
                                 or (cfg.pro_dir / SURVEY_ANCHOR_META).exists()):
        left.append(f"restart state in {cfg.pro_dir}")
    if cfg.zarr_path.exists():
        left.append(f"Zarr store {cfg.zarr_path}")
    if any(cfg.grids_dir.glob("hourly_*.npz")):
        left.append(f"hourly grids in {cfg.grids_dir}")
    if left:
        raise CycleError("bootstrap starts a season from nothing, but found "
                         + "; ".join(left) + ". Archive last season's tree first.")


def write_bootstrap_grids(cfg: ProjectConfig, wx: pd.DataFrame,
                          until: pd.Timestamp) -> Path:
    """Station-only HS grids from season_start (HS 0) to `until`."""
    from pipeline_io import load_dem
    from spatial_model import gap_fill_station_only
    t0 = bare_ground_start_ts(cfg)
    stn = wx.loc[t0:until, "HS"] / 100.0               # cm -> m
    if len(stn) < 2 or stn.index[0] > t0 + pd.Timedelta(hours=1):
        raise CycleError(f"station HS covers {wx.index.min()} .. {wx.index.max()}, "
                         f"not season_start {t0}; fill the cache with "
                         f"`aws_ingest.py --since {t0.date()}`")
    if stn.iloc[0] > 0.05:
        print(f"  WARNING: station HS is {100 * stn.iloc[0]:.0f} cm at season_start "
              f"{t0}; the season is meant to start on bare ground")
    dem, _, _ = load_dem(cfg)
    valid = ~np.isnan(dem)
    hs0 = np.where(valid, 0.0, np.nan).astype(np.float32)
    grids = gap_fill_station_only(hs0, stn, valid)
    cfg.grids_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.grids_dir / BOOTSTRAP_GRID_GLOB.replace("*", t0.date().isoformat())
    ts = list(grids)
    np.savez_compressed(str(out), grids=np.stack([grids[t] for t in ts]),
                        timestamps=np.array([str(t) for t in ts]))
    print(f"  Bootstrap grids: {len(ts)} hours, {ts[0]} -> {ts[-1]}, station HS "
          f"{100 * stn.iloc[0]:.0f} -> {100 * stn.iloc[-1]:.0f} cm -> {out.name}")
    return out


def _build_zarr(cfg: ProjectConfig) -> None:
    from snowpack_io import build_zarr_cache
    build_zarr_cache(cfg.pro_dir, cfg.zarr_path, rebuild=True)


def bootstrap(cfg: ProjectConfig, until: Optional[pd.Timestamp] = None, *,
              smet_fn: Optional[Callable] = None,
              snowpack: Callable = run_snowpack,
              zarr_fn: Callable = _build_zarr,
              wx: Optional[pd.DataFrame] = None) -> dict:
    """Steps 1-4 of the module docstring. Returns the anchor times."""
    if not cfg.start_from_bare_ground:
        raise CycleError("[snowpack] start_from_bare_ground is not true in the TOML: "
                         "without it the first survey's gap_fill does not rebuild the "
                         "pre-survey forcing, and the season would restart at that "
                         "survey instead of from bare ground")
    if not cfg.cluster_map_path.exists():
        raise CycleError(f"no cluster map at {cfg.cluster_map_path}; carry last "
                         f"season's onto this season's DEM first (cluster_carry.py)")
    _refuse_existing_state(cfg)
    if smet_fn is None:
        from forcing_pipeline import step_smet as smet_fn
    if wx is None:
        from pipeline_io import load_weather
        wx = load_weather(cfg)

    t0 = bare_ground_start_ts(cfg)
    if until is None:
        as_of = replay_as_of()
        until = wx.index.max() if as_of is None else min(
            wx.index.max(), pd.Timestamp(as_of).tz_convert(None)
            if pd.Timestamp(as_of).tzinfo else pd.Timestamp(as_of))
    obs_end = on_output_grid(pd.Timestamp(until))
    survey_anchor = on_output_grid(t0) + OUTPUT_STEP
    if obs_end <= survey_anchor:
        raise CycleError(f"station data ends {until}, not past the first profile "
                         f"step {survey_anchor:{TS_FMT}}")
    print(f"  Season bootstrap: bare ground {t0:{TS_FMT}} -> survey anchor "
          f"{survey_anchor:{TS_FMT}} -> daily anchor {obs_end:{TS_FMT}}", flush=True)

    write_bootstrap_grids(cfg, wx, obs_end)
    smet_fn(cfg)
    if not any(cfg.smet_dir.glob("cluster_*.smet")):
        raise CycleError(f"the SMET step wrote nothing to {cfg.smet_dir}")

    # Fresh clusters start from template.sno at the first SMET row (t0).
    snowpack(cfg, survey_anchor, "res")
    n = _require_coherent(cfg, "res", survey_anchor)
    _save_as(cfg, "survey")
    _write_survey_anchor(cfg, survey_anchor)
    snowpack(cfg, obs_end, "res")
    _require_coherent(cfg, "res", obs_end, n)
    _save_as(cfg, "anchor")
    zarr_fn(cfg)
    print(f"  Season started for {n} clusters: survey anchor {survey_anchor:{TS_FMT}}, "
          f"daily anchor {obs_end:{TS_FMT}}", flush=True)
    return {"clusters": n, "survey_anchor": survey_anchor, "daily_anchor": obs_end}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--toml", required=True)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bootstrap")
    b.add_argument("--until", default=None,
                   help="last observed hour to run to (default: the station "
                        "cache's end, or $AVACHAIN_AS_OF)")
    args = ap.parse_args()
    cfg = ProjectConfig.from_toml(Path(args.toml))
    try:
        bootstrap(cfg, pd.Timestamp(args.until) if args.until else None)
    except CycleError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

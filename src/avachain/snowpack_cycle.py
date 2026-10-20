"""
snowpack_cycle.py — SNOWPACK restart orchestration for the operational chain.

Two cycles, each re-running SNOWPACK from a saved restart point and replacing
everything the store held after that point:

  survey   a new UAS survey has landed.
             restart  the previous survey's state (<cid>_survey.sno)
             run      S_prev -> each detected avalanche event (stop, analyze,
                      scour the release clusters, continue) -> the new survey
                      (save as the new survey anchor) -> T_obs (save as the
                      daily anchor) -> optional NWP forecast to +48 h
             Zarr     main store replaced after S_prev; forecast store rebuilt

  daily    between surveys, after the day's station data.
             restart  the daily anchor (<cid>_anchor.sno, yesterday's T_obs)
             run      A -> T_obs (save as the new daily anchor) -> forecast
             Zarr     main store replaced after A; forecast store rebuilt

T_obs is T_stable when NWP is on (the SMETs are cut there), else the last
observed SMET row.

Why a survey re-runs from the PREVIOUS survey: the window since then is now
bracketed by two surveys, so gap_fill rebuilds its forcing from both, and
SNOWPACK must re-simulate it for the correction to reach the snowpack. The
old flow restarted from the latest state, so the correction only ever applied
going forward.

Why every segment runs the whole domain: reinit used to re-run only the
scoured clusters in pass 2, which strands each scoured group at its own event
date. That is how the live tree ended up with restart files at 18 different
dates. Here every segment ends with every cluster at the same time, and
_require_coherent() checks it.

Usage (called from run_<slope>.sh / run_daily_<slope>.sh):
    python snowpack_cycle.py --toml T survey --survey-date 2026-02-10 \\
        [--stable-ts 2026-02-10T18:00 --nwp-edate 2026-02-12T18:00]
    python snowpack_cycle.py --toml T daily \\
        [--stable-ts ... --nwp-edate ...]
    python snowpack_cycle.py --toml T init-anchors   # mark the current state
                                                     # as survey + daily anchor
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ProjectConfig                                  # noqa: E402

SURVEY_ANCHOR_META = "survey_anchor.json"
TS_FMT = "%Y-%m-%dT%H:%M"
# SNOWPACK writes a profile every PROF_DAYS_BETWEEN = 0.25 d (master_config.ini).
# Restart points must sit on that grid, or the store, which only has those
# steps, cannot end at its own restart point.
OUTPUT_STEP = pd.Timedelta(hours=6)


class CycleError(RuntimeError):
    """The restart state is not what the cycle needs; stop rather than guess."""


@dataclass
class Event:
    ts: pd.Timestamp          # event time (naive UTC)
    pair_id: str              # "<date_before>__<date_after>"
    release_geojson: Optional[str]

    @property
    def date(self) -> str:
        return self.ts.strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Restart files
# ---------------------------------------------------------------------------

def _ts(x) -> pd.Timestamp:
    t = pd.Timestamp(x)
    return t.tz_convert(None) if t.tzinfo else t


def _sno_files(cfg: ProjectConfig, kind: str) -> dict[str, Path]:
    """cid -> restart file, for kind res | anchor | survey."""
    pat = {"res": "cluster_*_cluster_*.sno", "anchor": "cluster_*_anchor.sno",
           "survey": "cluster_*_survey.sno"}[kind]
    out = {}
    for p in cfg.pro_dir.glob(pat):
        out[p.name.split("_cluster_")[0] if kind == "res"
            else p.name.rsplit("_", 1)[0]] = p
    return out


def _profile_date(p: Path) -> pd.Timestamp:
    with open(p) as f:
        for line in f:
            if line.startswith("ProfileDate"):
                return _ts(line.split("=", 1)[1].strip())
    raise CycleError(f"{p} has no ProfileDate")


def _require_coherent(cfg: ProjectConfig, kind: str, expect: pd.Timestamp,
                      n_expected: Optional[int] = None) -> int:
    """Every restart file of `kind` must be dated `expect`. Returns the count."""
    files = _sno_files(cfg, kind)
    if not files:
        raise CycleError(f"no {kind} restart files in {cfg.pro_dir}")
    off = {c: _profile_date(p) for c, p in files.items()}
    off = {c: t for c, t in off.items() if t != expect}
    if off:
        sample = ", ".join(f"{c}@{t:%Y-%m-%dT%H:%M}" for c, t in list(off.items())[:5])
        raise CycleError(f"{len(off)}/{len(files)} {kind} restart files are not at "
                         f"{expect:%Y-%m-%dT%H:%M}: {sample}")
    if n_expected is not None and len(files) != n_expected:
        raise CycleError(f"{len(files)} {kind} restart files, expected {n_expected}")
    return len(files)


def _save_as(cfg: ProjectConfig, kind: str) -> int:
    """Copy every <cid>_<cid>.sno to <cid>_<kind>.sno."""
    n = 0
    for cid, p in _sno_files(cfg, "res").items():
        shutil.copy2(p, cfg.pro_dir / f"{cid}_{kind}.sno")
        n += 1
    return n


def read_survey_anchor(cfg: ProjectConfig) -> pd.Timestamp:
    meta = cfg.pro_dir / SURVEY_ANCHOR_META
    if not meta.exists():
        raise CycleError(f"{meta} missing: no survey anchor to re-run from. "
                         f"Run `snowpack_cycle.py init-anchors` on a state that "
                         f"sits exactly at a survey time.")
    return _ts(json.loads(meta.read_text())["survey_ts"])


def _write_survey_anchor(cfg: ProjectConfig, ts: pd.Timestamp) -> None:
    (cfg.pro_dir / SURVEY_ANCHOR_META).write_text(json.dumps(
        {"survey_ts": ts.strftime(TS_FMT)}, indent=2) + "\n")


def on_output_grid(t: pd.Timestamp) -> pd.Timestamp:
    """Latest SNOWPACK profile step at or before t."""
    return t.floor(OUTPUT_STEP)


def survey_ts(cfg: ProjectConfig, survey_date: str) -> pd.Timestamp:
    """A survey is imposed in the forcing at date + flight_hour_utc (gap_fill)."""
    return _ts(survey_date) + pd.Timedelta(hours=cfg.flight_hour_utc)


def last_observed_ts(cfg: ProjectConfig) -> pd.Timestamp:
    """Earliest last-observed row over all cluster SMETs (NWP rows excluded)."""
    from nwp_ingest import NWP_FLAG
    last = None
    for p in sorted(cfg.smet_dir.glob("cluster_*.smet")):
        t_file = None
        in_data = False
        with open(p) as f:
            for line in f:
                s = line.strip()
                if not in_data:
                    in_data = s == "[DATA]"
                    continue
                if NWP_FLAG in s:
                    break
                if s and not s.startswith("#"):
                    t_file = s.split()[0]
        if t_file is not None:
            t = _ts(t_file)
            last = t if last is None or t < last else last
    if last is None:
        raise CycleError(f"no observed SMET rows in {cfg.smet_dir}")
    return last


def events_between(cfg: ProjectConfig, t0: pd.Timestamp,
                   t1: pd.Timestamp) -> list[Event]:
    """Detected events needing reinit with t0 < event time <= t1, in order."""
    p = cfg.analysis_dir / "avalanche_events.json"
    if not p.exists():
        return []
    out = []
    for pair_id, ev in json.loads(p.read_text()).items():
        if not ev.get("reinit_needed", ev.get("corrected", False)):
            continue
        ts = ev.get("event_timestamp")
        if not ts:
            continue
        t = _ts(ts)
        if t0 < t <= t1:
            gj = cfg.event_mask_for(t)
            out.append(Event(t, pair_id, str(gj) if gj else None))
    return sorted(out, key=lambda e: e.ts)


# ---------------------------------------------------------------------------
# Actions (replaceable in tests)
# ---------------------------------------------------------------------------

def run_snowpack(cfg: ProjectConfig, end: pd.Timestamp, restart_from: str) -> None:
    script = cfg.slopes_config_dir / "run_snowpack.sh"
    env = dict(os.environ, RESTART_FROM=restart_from, SKIP_ZARR="1",
               SLOPE_DIR=str(cfg.slope_dir), SMET_DIR=str(cfg.smet_dir))
    print(f"    SNOWPACK {restart_from} -> {end:{TS_FMT}}", flush=True)
    r = subprocess.run(["bash", str(script), "", end.strftime(TS_FMT)], env=env)
    if r.returncode != 0:
        raise CycleError(f"run_snowpack.sh exited {r.returncode} "
                         f"(RESTART_FROM={restart_from}, end={end:{TS_FMT}})")


def _py(cfg: ProjectConfig, module: str, *args: str) -> None:
    # --toml is mandatory here: without it a subprocess silently falls back to
    # the default ProjectConfig and reads or writes another slope's files.
    if cfg.toml_path is None:
        raise CycleError("config has no toml_path; load it with from_toml")
    cmd = [sys.executable, str(Path(__file__).resolve().parent / module),
           "--toml", str(cfg.toml_path), *args]
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise CycleError(f"{' '.join(cmd[1:])} exited {r.returncode}")


def analyze(cfg: ProjectConfig, snapshot_date: str) -> None:
    _py(cfg, "analysis_pipeline.py", "analyze", "--snapshot-date", snapshot_date,
        "--force")


def reinit(cfg: ProjectConfig, ev: Event) -> None:
    d_before, d_after = ev.pair_id.split("__")
    args = ["reinit", "--event-date", ev.date, "--event-time",
            ev.ts.strftime("%H:%M:%S"), "--date-before", d_before,
            "--date-after", d_after, "--snapshot-date", ev.date]
    if ev.release_geojson:
        args += ["--release-geojson", ev.release_geojson]
    _py(cfg, "forcing_pipeline.py", *args)


def update_zarr(cfg: ProjectConfig, since: pd.Timestamp) -> None:
    from snowpack_io import update_zarr_since
    update_zarr_since(cfg.pro_dir, cfg.zarr_path, since)


def build_forecast(cfg: ProjectConfig, since: pd.Timestamp) -> None:
    from snowpack_io import build_zarr_cache
    build_zarr_cache(cfg.pro_dir, cfg.forecast_zarr_path, rebuild=True, since=since)


# ---------------------------------------------------------------------------
# Cycles
# ---------------------------------------------------------------------------

def survey_cycle(cfg: ProjectConfig, new_survey: pd.Timestamp,
                 obs_end: Optional[pd.Timestamp] = None,
                 nwp_edate: Optional[pd.Timestamp] = None,
                 *, snowpack: Callable = run_snowpack, analyze_fn: Callable = analyze,
                 reinit_fn: Callable = reinit, zarr_fn: Callable = update_zarr,
                 forecast_fn: Callable = build_forecast) -> dict:
    s_prev = read_survey_anchor(cfg)
    if new_survey <= s_prev:
        raise CycleError(f"new survey {new_survey:{TS_FMT}} is not after the "
                         f"survey anchor {s_prev:{TS_FMT}}")
    n = _require_coherent(cfg, "survey", s_prev)
    obs_end = on_output_grid(obs_end or last_observed_ts(cfg))
    if obs_end < new_survey:
        raise CycleError(f"forcing ends {obs_end:{TS_FMT}}, before the new survey "
                         f"{new_survey:{TS_FMT}}")
    events = events_between(cfg, s_prev, new_survey)
    print(f"  Survey cycle: {s_prev:{TS_FMT}} -> {new_survey:{TS_FMT}} -> "
          f"{obs_end:{TS_FMT}}; {len(events)} event(s) to reinit", flush=True)

    restart = "survey"
    for ev in events:
        snowpack(cfg, ev.ts, restart)
        restart = "res"
        _require_coherent(cfg, "res", ev.ts, n)
        zarr_fn(cfg, s_prev)                 # analyze needs the event-date state
        analyze_fn(cfg, ev.date)
        reinit_fn(cfg, ev)                   # scours res .sno, dated ev.ts
        _require_coherent(cfg, "res", ev.ts, n)

    snowpack(cfg, new_survey, restart)
    _require_coherent(cfg, "res", new_survey, n)
    _save_as(cfg, "survey")
    _write_survey_anchor(cfg, new_survey)

    if obs_end > new_survey:
        snowpack(cfg, obs_end, "res")
        _require_coherent(cfg, "res", obs_end, n)
    _save_as(cfg, "anchor")
    zarr_fn(cfg, s_prev)                     # observation-driven only

    if nwp_edate is not None:
        snowpack(cfg, nwp_edate, "anchor")
        _require_coherent(cfg, "res", nwp_edate, n)   # a stalled forecast is not silent
        forecast_fn(cfg, obs_end)
    return {"survey_prev": s_prev, "survey_new": new_survey, "obs_end": obs_end,
            "events": [e.ts for e in events], "clusters": n}


def daily_cycle(cfg: ProjectConfig, obs_end: Optional[pd.Timestamp] = None,
                nwp_edate: Optional[pd.Timestamp] = None,
                *, snowpack: Callable = run_snowpack, zarr_fn: Callable = update_zarr,
                forecast_fn: Callable = build_forecast) -> dict:
    files = _sno_files(cfg, "anchor")
    if not files:
        raise CycleError(f"no daily anchors in {cfg.pro_dir}")
    a = _profile_date(next(iter(files.values())))
    n = _require_coherent(cfg, "anchor", a)
    obs_end = on_output_grid(obs_end or last_observed_ts(cfg))
    print(f"  Daily cycle: {a:{TS_FMT}} -> {obs_end:{TS_FMT}}", flush=True)
    if obs_end > a:
        snowpack(cfg, obs_end, "anchor")
        _require_coherent(cfg, "res", obs_end, n)
        _save_as(cfg, "anchor")
        zarr_fn(cfg, a)
    else:
        print("    No new observations since the anchor; state unchanged")
    if nwp_edate is not None:
        snowpack(cfg, nwp_edate, "anchor")
        _require_coherent(cfg, "res", nwp_edate, n)   # a stalled forecast is not silent
        forecast_fn(cfg, max(obs_end, a))
    return {"anchor_prev": a, "obs_end": obs_end, "clusters": n}


def init_anchors(cfg: ProjectConfig, survey_date: str) -> int:
    """Mark the current restart state as both anchors. It must sit at the survey."""
    t = survey_ts(cfg, survey_date)
    n = _require_coherent(cfg, "res", t)
    _save_as(cfg, "survey")
    _save_as(cfg, "anchor")
    _write_survey_anchor(cfg, t)
    print(f"  Anchors initialised for {n} clusters at {t:{TS_FMT}}")
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--toml", required=True)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("survey")
    s.add_argument("--survey-date", required=True)
    d = sub.add_parser("daily")
    for p in (s, d):
        p.add_argument("--stable-ts", default=None)
        p.add_argument("--nwp-edate", default=None)
    i = sub.add_parser("init-anchors")
    i.add_argument("--survey-date", required=True)
    args = ap.parse_args()

    cfg = ProjectConfig.from_toml(Path(args.toml))
    try:
        if args.cmd == "init-anchors":
            init_anchors(cfg, args.survey_date)
            return
        obs = _ts(args.stable_ts) if args.stable_ts else None
        fc = _ts(args.nwp_edate) if args.nwp_edate else None
        if args.cmd == "survey":
            survey_cycle(cfg, survey_ts(cfg, args.survey_date), obs, fc)
        else:
            daily_cycle(cfg, obs, fc)
    except CycleError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()

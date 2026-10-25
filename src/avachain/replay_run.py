"""
replay_run.py — Replay the operational chain over a past window, tick by tick.

Spec: docs/operational_replay_testing.md. Each simulated day sets the replay
clock ($AVACHAIN_AS_OF) and runs exactly what operations would run that day:

  survey day   a survey dated D is visible and newer than the survey anchor:
               AS_OF = D T20:00 (after the 18:00 flight), run_<slope>.sh
  other days   AS_OF = D T14:00 (the 08:00 local morning run),
               run_daily_<slope>.sh

After each tick the driver asserts the Tier 3 invariants and stops at the
first failure. The tick's state is left in place for inspection.
  - every daily anchor is at the same time A, A <= the clock, and A moved
    forward (or a survey tick put the survey anchor at D T18:00);
  - the main Zarr ends exactly at A: no forecast step, nothing stale;
  - the forecast Zarr, when NWP ran, starts after A;
  - the tick wrote features for its snapshot date.

It archives what a later survey re-run will overwrite, because the
pre-correction state is the Tier 2 measurement: HS for every cluster at A,
plus the tick's feature CSVs and scenario summary.

    python src/avachain/replay_run.py --toml slopes/little_prof_replay/slope_config.toml \\
        setup --live-toml slopes/little_prof/slope_config.toml --t-start 2026-02-05
    python src/avachain/replay_run.py --toml slopes/little_prof_replay/slope_config.toml \\
        run --start 2026-02-06 --end 2026-03-08 --scenarios-on surveys
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ProjectConfig                                  # noqa: E402
import snowpack_cycle as sc                                       # noqa: E402
from pipeline_io import discover_surveys                          # noqa: E402

SURVEY_HOUR = 20      # tick clock on a survey day (flight at 18:00 UTC)
DAILY_HOUR = 14       # tick clock otherwise (08:00 MDT)
FROZEN_ANALYSIS = ("cluster_map.npy", "cluster_map.tif", "domain_mask.npy")
FROZEN_RESAMPLED = ("dem_1m.tif",)


class Tier3Failure(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------

def plan(survey_dates: list[date], start: date, end: date, anchor_survey: date,
         scenarios_on: set) -> list[dict]:
    """One entry per day: which script, which clock, whether scenarios run."""
    surveys = {d for d in survey_dates if d > anchor_survey}
    out, d = [], start
    while d <= end:
        kind = "survey" if d in surveys else "daily"
        hour = SURVEY_HOUR if kind == "survey" else DAILY_HOUR
        scen = d.isoformat() in scenarios_on or (kind == "survey" and "surveys" in scenarios_on)
        out.append({"date": d.isoformat(), "kind": kind,
                    "as_of": f"{d.isoformat()}T{hour:02d}:00", "scenarios": scen})
        d += timedelta(days=1)
    return out


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def setup(cfg: ProjectConfig, live: ProjectConfig, t_start: date) -> None:
    cfg.analysis_dir.mkdir(parents=True, exist_ok=True)
    for name in FROZEN_ANALYSIS:
        src, dst = live.analysis_dir / name, cfg.analysis_dir / name
        if not dst.exists():
            shutil.copy2(src, dst)
            print(f"  copied frozen {name}")
    # The 1 m DEM is static terrain, so copying it leaks nothing. analyze needs
    # it on every tick, but only `resample` (survey ticks) writes it.
    cfg.resampled_dir.mkdir(parents=True, exist_ok=True)
    for name in FROZEN_RESAMPLED:
        src, dst = live.resampled_dir / name, cfg.resampled_dir / name
        if not dst.exists():
            shutil.copy2(src, dst)
            print(f"  copied static {name}")
    known = json.loads(live.avalanche_events_path.read_text()) \
        if live.avalanche_events_path.exists() else []
    seed = [e for e in known if e["period"].split("__")[1] <= t_start.isoformat()]
    cfg.avalanche_events_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.avalanche_events_path.write_text(json.dumps(seed, indent=2) + "\n")
    print(f"  known events: {len(seed)} of {len(known)} (pair ended by {t_start})")
    s = sc.read_survey_anchor(cfg)
    want = sc.survey_ts(cfg, t_start.isoformat())
    if s != want:
        raise Tier3Failure(f"survey anchor {s} != {want}")
    n = sc._require_coherent(cfg, "survey", s)
    sc._require_coherent(cfg, "anchor", s, n)
    print(f"  anchors: {n} clusters at {s:%Y-%m-%dT%H:%M}")


# ---------------------------------------------------------------------------
# Tier 3 + archive
# ---------------------------------------------------------------------------

def _zarr_times(path: Path):
    import xarray as xr
    if not path.exists():
        return None
    with xr.open_zarr(str(path)) as z:
        return z["time"].values


def check_tick(cfg: ProjectConfig, tick: dict, prev_anchor: pd.Timestamp,
               t0: float, nwp: bool) -> dict:
    as_of = pd.Timestamp(tick["as_of"])
    files = sc._sno_files(cfg, "anchor")
    a = sc._profile_date(next(iter(files.values())))
    rec = {"anchor": a.strftime(sc.TS_FMT), "clusters": sc._require_coherent(cfg, "anchor", a)}
    if a > as_of:
        raise Tier3Failure(f"daily anchor {a} is after the clock {as_of}")
    if a <= prev_anchor:
        raise Tier3Failure(f"daily anchor did not advance: {a} <= {prev_anchor}")
    if tick["kind"] == "survey":
        s = sc.read_survey_anchor(cfg)
        want = sc.survey_ts(cfg, tick["date"])
        if s != want:
            raise Tier3Failure(f"survey anchor {s} != {want}")
        sc._require_coherent(cfg, "survey", s, rec["clusters"])
        rec["survey_anchor"] = s.strftime(sc.TS_FMT)
    main_t = _zarr_times(cfg.zarr_path)
    last = pd.Timestamp(main_t[-1])
    rec["zarr_last"] = last.strftime(sc.TS_FMT)
    if last != a:
        raise Tier3Failure(f"main Zarr ends {last}, daily anchor is {a}")
    if nwp:
        fc_t = _zarr_times(cfg.forecast_zarr_path)
        if fc_t is None or len(fc_t) == 0:
            raise Tier3Failure("NWP on but the forecast store is missing or empty")
        if pd.Timestamp(fc_t[0]) <= a:
            raise Tier3Failure(f"forecast store starts {fc_t[0]}, not after {a}")
        rec["forecast"] = [str(fc_t[0])[:16], str(fc_t[-1])[:16]]
    feat = cfg.analysis_dir / f"all_start_zone_features_{tick['date']}.csv"
    if not feat.exists() or feat.stat().st_mtime < t0:
        raise Tier3Failure(f"{feat.name} was not written by this tick")
    return rec


def archive_tick(cfg: ProjectConfig, tick: dict, rec: dict, out: Path) -> None:
    import xarray as xr
    out.mkdir(parents=True, exist_ok=True)
    d = tick["date"]
    for name in (f"all_start_zone_features_{d}.csv", f"meloche_features_all_{d}.csv"):
        if (cfg.analysis_dir / name).exists():
            shutil.copy2(cfg.analysis_dir / name, out / name)
    scen = cfg.output_dir / "scenarios" / d
    for name in ("summary.csv", "trigger_locations.geojson", "metadata.json"):
        if (scen / name).exists():
            shutil.copy2(scen / name, out / name)
    with xr.open_zarr(str(cfg.zarr_path)) as z:
        hs = z["HS"].sel(time=np.datetime64(pd.Timestamp(rec["anchor"]))).load()
        np.save(out / "hs_at_anchor.npy", hs.values.astype("float32"))
        locs = out.parent / "locations.json"
        if not locs.exists():
            locs.write_text(json.dumps([str(v) for v in z["location"].values]))
    (out / "tier3.json").write_text(json.dumps({**tick, **rec}, indent=2) + "\n")


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run(cfg: ProjectConfig, start: date, end: date, scenarios_on: set,
        archive: Path, nwp: bool, dry_run: bool) -> None:
    repo = cfg.project_dir
    slope = cfg.slope_name
    surveys = [d for d, _ in discover_surveys(cfg, as_of=f"{end.isoformat()}T23:59")]
    anchor_survey = sc.read_survey_anchor(cfg).date()
    ticks = plan(surveys, start, end, anchor_survey, scenarios_on)
    progress = archive / "progress.json"
    done = json.loads(progress.read_text()) if progress.exists() else []
    print(f"Replay {start} -> {end}: {len(ticks)} ticks, "
          f"{sum(t['kind'] == 'survey' for t in ticks)} survey, "
          f"{sum(t['scenarios'] for t in ticks)} with scenarios; {len(done)} already done")
    if dry_run:
        for t in ticks:
            print(f"  {t['date']}  {t['kind']:6s}  as_of={t['as_of']}"
                  f"{'  +scenarios' if t['scenarios'] else ''}")
        return

    for t in ticks:
        if t["date"] in done:
            continue
        prev = sc._profile_date(next(iter(sc._sno_files(cfg, "anchor").values())))
        script = repo / (f"run_{slope}.sh" if t["kind"] == "survey" else f"run_daily_{slope}.sh")
        args = ["--snapshot", t["date"]]
        if t["kind"] == "survey":
            args.append("--no-cluster-update")
        if not t["scenarios"]:
            args.append("--no-scenarios")
        if not nwp:
            args.append("--no-nwp")
        env = dict(os.environ, AVACHAIN_AS_OF=t["as_of"], SLOPE_DIR=str(cfg.slope_dir))
        out = archive / t["date"]
        out.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {t['date']} {t['kind']} as_of={t['as_of']} ===", flush=True)
        t0 = time.time()
        with open(out / "tick.log", "w") as log:
            r = subprocess.run(["bash", str(script), *args], env=env, cwd=repo,
                               stdout=log, stderr=subprocess.STDOUT)
        t["seconds"] = round(time.time() - t0)
        if r.returncode != 0:
            raise Tier3Failure(f"{script.name} exited {r.returncode} on {t['date']}; "
                               f"see {out / 'tick.log'}")
        rec = check_tick(cfg, t, prev, t0, nwp)
        archive_tick(cfg, t, rec, out)
        done.append(t["date"])
        progress.write_text(json.dumps(done, indent=2) + "\n")
        print(f"    ok in {t['seconds']}s: anchor {rec['anchor']}, zarr {rec['zarr_last']}"
              + (f", survey {rec['survey_anchor']}" if "survey_anchor" in rec else ""),
              flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--toml", required=True)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("setup")
    s.add_argument("--live-toml", required=True)
    s.add_argument("--t-start", required=True)
    r = sub.add_parser("run")
    r.add_argument("--start", required=True)
    r.add_argument("--end", required=True)
    r.add_argument("--scenarios-on", default="surveys")
    r.add_argument("--archive", default=None)
    r.add_argument("--no-nwp", action="store_true")
    r.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    cfg = ProjectConfig.from_toml(Path(a.toml))
    try:
        if a.cmd == "setup":
            setup(cfg, ProjectConfig.from_toml(Path(a.live_toml)),
                  date.fromisoformat(a.t_start))
        else:
            run(cfg, date.fromisoformat(a.start), date.fromisoformat(a.end),
                set(a.scenarios_on.split(",")),
                Path(a.archive) if a.archive else cfg.output_dir / "ticks",
                nwp=not a.no_nwp, dry_run=a.dry_run)
    except (Tier3Failure, sc.CycleError) as exc:
        print(f"\nREPLAY STOPPED: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()

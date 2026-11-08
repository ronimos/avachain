#!/usr/bin/env python3
"""Season analytics: append-only records of what only the live season knows.

docs/analytics_design.md is the design (Ron, 2026-11-08). Four JSONL files
per season, under <metrics_dir>/<season>/:

    runs.jsonl          one line per daily or survey run (timings, exit, inputs)
    surveys.jsonl       one line per survey (model HS before the anchor vs survey)
    cycles.jsonl        one line per published release set (trigger funnel)
    observations.jsonl  ground truth: avalanches, control shots, forecaster notes

and <issued_dir>/<season>/<cycle_date>/ for what forecasters were shown and a
rerun would otherwise overwrite (forecast-store snapshots, superseded release
sets and reports).

The one rule: **a recorder never changes a run's outcome.** Every public
function catches everything, prints one WARNING line and returns None. The
records hold raw numbers, not scores, so a metric redefined next summer
needs no rerun.

CLI (called by the generated run scripts and by hand):

    metrics.py --toml T run --kind daily --exit-code 0 --log L --steps "aws_ingest=12 ..."
    metrics.py --toml T survey --date 2026-11-26
    metrics.py --toml T observe --type control_shot --time "2026-12-03 07:40" \\
        --source "CDOT log" --field shot_point=PW-3 --field result=no_result
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import shutil
import sys
import traceback
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import ProjectConfig  # noqa: E402

SCHEMA = 1
FILES = {"run": "runs.jsonl", "survey": "surveys.jsonl",
         "cycle": "cycles.jsonl", "observation": "observations.jsonl"}
OBSERVATION_TYPES = ("avalanche", "control_shot", "forecaster_note")
REPORT_DEADLINE_LOCAL = dt.time(3, 0)     # Ron, 2026-11-04: report by 03:00 local


# One stamp per run: the run scripts export AVACHAIN_RUN_ID once, so every
# subprocess's records and superseded copies share it. Outside a run script
# each process gets its own.
_PROCESS_STAMP = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def run_id() -> str:
    import os
    return os.environ.get("AVACHAIN_RUN_ID") or _PROCESS_STAMP


def _warn(what: str, exc: BaseException) -> None:
    print(f"WARNING: metrics: {what} not recorded: "
          f"{type(exc).__name__}: {exc}", flush=True)


def _now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def season_dir(cfg: ProjectConfig, root: Path) -> Path:
    return Path(root) / cfg.season_label


def issued_cycle_dir(cfg: ProjectConfig, cycle_date: str) -> Path:
    return season_dir(cfg, cfg.issued_dir) / str(cycle_date)


def _common(cfg: ProjectConfig, kind: str) -> dict:
    import handoff
    toml = cfg.toml_path
    return {"schema": SCHEMA, "kind": kind, "run_id": run_id(), "slope": cfg.slope_name,
            "written_utc": _now_utc(), "commit": handoff._commit(),
            "toml_sha256": handoff._sha256(toml) if toml and Path(toml).exists() else None,
            "calibration_version": calibration_version(toml)}


def calibration_version(toml) -> Optional[str]:
    """[calibration] version from the slope TOML, or None (uncalibrated).
    CLAUDE.md "Calibration": every record says which version produced it."""
    try:
        import tomllib
        with open(toml, "rb") as f:
            v = tomllib.load(f).get("calibration", {}).get("version")
        return str(v) if v is not None else None
    except Exception:                            # noqa: BLE001
        return None


def record(cfg: ProjectConfig, kind: str, fields: dict) -> Optional[Path]:
    """Append one record. Never raises."""
    try:
        out = season_dir(cfg, cfg.metrics_dir) / FILES[kind]
        out.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps({**_common(cfg, kind), **fields}, default=_jsonable,
                          sort_keys=False)
        with open(out, "a") as f:
            f.write(line + "\n")
        return out
    except Exception as e:                       # noqa: BLE001 - see module rule
        _warn(f"{kind} record", e)
        return None


def _jsonable(v):
    try:
        import numpy as np
        if isinstance(v, np.generic):
            return v.item()
    except ImportError:
        pass
    if isinstance(v, (dt.date, dt.datetime)):
        return v.isoformat()
    if isinstance(v, Path):
        return str(v)
    return str(v)


# ---------------------------------------------------------------------------
# Run record: the run script's own timings + what its log says
# ---------------------------------------------------------------------------

_RE = {
    "station_rows": re.compile(r"^\s*\[(\w+)\] (\d+) rows, (\S+ \S+) -> (\S+ \S+)"),
    "station_cached": re.compile(r"^\s*\[(\w+)\] cached (\d+) rows"),
    "smet": re.compile(r"SMET append: (\d+)/(\d+) files updated, (\d+) total rows"),
    "t_stable": re.compile(r"^\s*T_stable\s+=\s+(\S+)"),
    "nwp_edate": re.compile(r"^\s*NWP EDATE = (\S+)"),
    "wrf_unavailable": re.compile(r"WRF not available"),
    "sp_anchor": re.compile(r"SNOWPACK anchor -> (\S+)(.*)"),
    "sp_started": re.compile(r"^\s*Started:\s+(.+ UTC \d{4})"),
    "sp_done": re.compile(r"=== Run complete: (.+ UTC \d{4}) ==="),
    "sp_ok": re.compile(r"^\s*OK: (\d+)\s+Failed: (\d+)"),
    "sp_clamped": re.compile(r"Slope clamped to [\d.]+ deg for (\d+) cluster"),
    "zarr_update": re.compile(r"update_zarr_since: .* in (\d+)s"),
    "zarr_done": re.compile(r"^Done in ([\d.]+)min → (\S+)"),
    "cycle": re.compile(r"=== step_forecast\s+cycle=(\S+)\s+days=(\d+)"),
    "day_failed": re.compile(r"^\s*(T_\d+) (\S+): FAILED: (.*)"),
    "day_skipped": re.compile(r"^\s*(T_\d+) (\S+): skipped \((.*)\)"),
    "published": re.compile(r"forecast cycle (\S+): (\d+) of (\d+) days published"),
    "no_snow": re.compile(r"No snow cover: start-zone max HS ([\d.]+) m"),
    "runout_exit": re.compile(r"^Runout \((\w+)\) exit (\d+)"),
    "alert": re.compile(r"^ALERT sent to [^:]+: (.*)"),
    "runout_lock": re.compile(r"^\s*lock taken after (\d+) s"),
    "survey": re.compile(r"^\s*Survey:\s+(\S+)"),
}


def _utc_from_date_cmd(s: str) -> Optional[str]:
    """'Sun Nov  8 07:01:20 AM UTC 2026' (date(1) in run_snowpack.sh) -> ISO."""
    try:
        t = dt.datetime.strptime(re.sub(r"\s+", " ", s.strip()),
                                 "%a %b %d %I:%M:%S %p UTC %Y")
        return t.strftime("%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return None


def parse_run_log(text: str) -> dict:
    """What a daily or survey run log says, as raw fields. Missing = absent."""
    out: dict = {"stations": {}, "snowpack_passes": [], "forecast_days": {},
                 "runout_exit": {}, "alerts": [], "n_warning_lines": 0}
    sp_kind = None
    cur = None
    for line in text.splitlines():
        if line.startswith("WARNING") or line.lstrip().startswith("WARNING:"):
            out["n_warning_lines"] += 1
        if m := _RE["station_rows"].search(line):
            out["stations"].setdefault(m[1], {}).update(
                rows=int(m[2]), first_utc=m[3], last_utc=m[4])
        elif m := _RE["station_cached"].search(line):
            out["stations"].setdefault(m[1], {})["cached"] = int(m[2])
        elif m := _RE["smet"].search(line):
            out["smet_append"] = {"files_updated": int(m[1]), "files": int(m[2]),
                                  "rows": int(m[3])}
        elif m := _RE["t_stable"].search(line):
            out["t_stable"] = m[1]
        elif m := _RE["nwp_edate"].search(line):
            out["nwp_edate"] = m[1]
            out["nwp_clamped"] = "clamped" in line and "T_stable +" not in line.split("(")[0]
        elif _RE["wrf_unavailable"].search(line):
            out["wrf_available"] = False
        elif m := _RE["sp_anchor"].search(line):
            sp_kind = "forecast" if "forecast" in m[2] else "observed"
        elif m := _RE["sp_started"].search(line):
            cur = {"pass": sp_kind or "observed", "started_utc": _utc_from_date_cmd(m[1])}
            out["snowpack_passes"].append(cur)
        elif (m := _RE["sp_done"].search(line)) and cur is not None:
            cur["finished_utc"] = _utc_from_date_cmd(m[1])
            if cur.get("started_utc") and cur["finished_utc"]:
                a = dt.datetime.fromisoformat(cur["started_utc"].rstrip("Z"))
                b = dt.datetime.fromisoformat(cur["finished_utc"].rstrip("Z"))
                cur["seconds"] = int((b - a).total_seconds())
        elif (m := _RE["sp_ok"].search(line)) and cur is not None:
            cur["ok"], cur["failed"] = int(m[1]), int(m[2])
        elif (m := _RE["sp_clamped"].search(line)) and cur is not None:
            cur["slope_clamped"] = int(m[1])
        elif m := _RE["zarr_update"].search(line):
            out["zarr_update_s"] = int(m[1])
        elif m := _RE["zarr_done"].search(line):
            key = "forecast_zarr_s" if "forecast" in m[2] else "zarr_build_s"
            out[key] = round(float(m[1]) * 60)
        elif m := _RE["cycle"].search(line):
            out["cycle_date"], out["n_forecast_days"] = m[1], int(m[2])
        elif m := _RE["day_failed"].search(line):
            out["forecast_days"][m[1]] = {"date": m[2], "status": "failed", "error": m[3]}
        elif m := _RE["day_skipped"].search(line):
            out["forecast_days"][m[1]] = {"date": m[2], "status": "skipped", "reason": m[3]}
        elif m := _RE["published"].search(line):
            out["days_published"] = int(m[2])
        elif m := _RE["no_snow"].search(line):
            out["no_snow_max_hs_m"] = float(m[1])
        elif m := _RE["runout_exit"].search(line):
            out["runout_exit"][m[1]] = int(m[2])
        elif m := _RE["runout_lock"].search(line):
            out["runout_lock_wait_s"] = out.get("runout_lock_wait_s", 0) + int(m[1])
        elif m := _RE["alert"].search(line):
            out["alerts"].append(m[1])        # subject only: no addresses
        elif m := _RE["survey"].search(line):
            out["survey"] = m[1]
    return out


def _parse_steps(s: str) -> dict:
    steps = {}
    for tok in (s or "").split():
        name, _, val = tok.partition("=")
        if name and val.strip().lstrip("-").isdigit():
            steps[name] = int(val)
    return steps


def _report_info(cfg: ProjectConfig, cycle_date: Optional[str]) -> dict:
    """mtime of the published report for the cycle, and the margin to 03:00."""
    import alerts
    s = alerts.settings(cfg.toml_path)
    if not (cycle_date and s.get("publish_root") and s.get("path_profile")):
        return {}
    day = cycle_date.replace("-", "")
    rep = Path(s["publish_root"]) / "forecasts" / s["path_profile"] / day / "report.html"
    if not rep.exists():
        return {"report_published": False}
    t = dt.datetime.fromtimestamp(rep.stat().st_mtime)          # host local time
    deadline = dt.datetime.combine(dt.date.fromisoformat(cycle_date), REPORT_DEADLINE_LOCAL)
    return {"report_published": True, "report_mtime_local": t.isoformat(timespec="seconds"),
            "deadline_margin_min": round((deadline - t).total_seconds() / 60)}


def run_record(cfg: ProjectConfig, kind: str, exit_code: int, log: Optional[Path],
               steps: str = "", total_s: Optional[int] = None,
               runout_status: str = "", failed_step: str = "",
               snapshot: str = "") -> Optional[Path]:
    try:
        text = Path(log).read_text(errors="replace") if log and Path(log).exists() else ""
        parsed = parse_run_log(text)
        cycle = parsed.get("cycle_date") or snapshot or None
        if failed_step == "" and exit_code != 0:
            hits = [l[4:] for l in text.splitlines() if l.startswith(">>> Step")]
            failed_step = hits[-1] if hits else "start"
        try:
            disk = shutil.disk_usage(cfg.output_dir if cfg.output_dir.exists()
                                     else cfg.project_dir).free / 1e9
        except OSError:
            disk = None
        fields = {
            "run_kind": kind, "exit_code": int(exit_code),
            "failed_step": failed_step or None,
            "as_of": __import__("os").environ.get("AVACHAIN_AS_OF") or None,
            "log": str(log) if log else None,
            "total_s": total_s, "steps_s": _parse_steps(steps),
            "runout_status": runout_status or None,
            "cycle_date": cycle,
            **parsed,
            **(_report_info(cfg, cycle) if exit_code == 0 else {}),
            "disk_free_gb": round(disk, 1) if disk is not None else None,
        }
        return record(cfg, "run", fields)
    except Exception as e:                       # noqa: BLE001
        _warn("run record", e)
        return None


# ---------------------------------------------------------------------------
# Survey record: model HS before the anchor against the survey, by terrain
# ---------------------------------------------------------------------------

def _pcts(vals) -> Optional[dict]:
    import numpy as np
    v = np.asarray([x for x in vals if x is not None], dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None
    p = np.percentile(v, [1, 10, 50, 90, 99])
    return {"n": int(v.size), "p1": round(float(p[0]), 3), "p10": round(float(p[1]), 3),
            "p50": round(float(p[2]), 3), "p90": round(float(p[3]), 3),
            "p99": round(float(p[4]), 3), "mean": round(float(v.mean()), 3)}


def _smet_terrain(cfg: ProjectConfig) -> dict:
    """{cid: (altitude_m, slope_azi_deg)} from the SMET headers SNOWPACK used."""
    out = {}
    for f in sorted(Path(cfg.smet_dir).glob("cluster_*.smet")):
        alt = azi = None
        with open(f) as fh:
            for line in fh:
                if line.startswith("[DATA]"):
                    break
                k, _, v = line.partition("=")
                k = k.strip()
                if k == "altitude":
                    alt = float(v)
                elif k == "slope_azi":
                    azi = float(v)
        try:
            out[int(f.stem.split("_")[-1])] = (alt, azi)
        except ValueError:
            continue
    return out


def _start_zone_ids(cfg: ProjectConfig) -> set:
    import numpy as np
    import rasterio
    from snowpack_analysis import kml_to_mask
    with rasterio.open(cfg.analysis_dir / "cluster_map.tif") as ds:
        cmap = ds.read(1)
        mask = kml_to_mask(cfg.start_zone_kml, cmap.shape, ds.transform)
    return set(int(c) for c in np.unique(cmap[mask]) if c > 0)


OCTANTS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")


def _octant(azi: Optional[float]) -> Optional[str]:
    # SMET slope_azi is a compass bearing (0 = N, clockwise), the map frame.
    if azi is None:
        return None
    return OCTANTS[int(((azi % 360) + 22.5) // 45) % 8]


def survey_summary(cfg: ProjectConfig, date: str) -> dict:
    a = cfg.analysis_dir
    out: dict = {"survey_date": date}
    chk = a / f"survey_check_{date}.json"
    if chk.exists():
        c = json.loads(chk.read_text())
        out["frame_check"] = {k: c.get(k) for k in ("verdict", "p10_m", "p50_m", "cover")}
    for name, key in ((f"anchor_counts_{date}.json", "anchor_counts"),
                      (f"anchor_check_{date}.json", "after_anchor")):
        if (a / name).exists():
            out[key] = json.loads((a / name).read_text())
    st = a / f"anchor_stats_{date}.json"
    if st.exists():
        stats = json.loads(st.read_text())
        diff = {s["cid"]: s["hs_model_m"] - s["hs_survey_m"] for s in stats
                if s.get("hs_model_m") is not None and s.get("hs_survey_m") is not None}
        out["prior_minus_survey_m"] = {"all": _pcts(diff.values())}
        try:
            sz = _start_zone_ids(cfg)
            out["prior_minus_survey_m"]["start_zone"] = _pcts(v for c, v in diff.items() if c in sz)
            out["prior_minus_survey_m"]["outside_start_zone"] = _pcts(
                v for c, v in diff.items() if c not in sz)
        except Exception as e:                   # noqa: BLE001
            _warn("start-zone split", e)
        try:
            terr = _smet_terrain(cfg)
            by_elev, by_asp = {}, {}
            for c, v in diff.items():
                alt, azi = terr.get(c, (None, None))
                if alt is not None:
                    band = f"{int(alt // 100) * 100}"
                    by_elev.setdefault(band, []).append(v)
                if (o := _octant(azi)) is not None:
                    by_asp.setdefault(o, []).append(v)
            out["prior_minus_survey_m"]["by_elevation_100m"] = {
                k: _pcts(v) for k, v in sorted(by_elev.items())}
            out["prior_minus_survey_m"]["by_aspect"] = {
                k: _pcts(by_asp[k]) for k in OCTANTS if k in by_asp}
        except Exception as e:                   # noqa: BLE001
            _warn("terrain split", e)
    rs = a / f"reinit_stats_{date}.json"
    if rs.exists():
        r = json.loads(rs.read_text())
        rows = r if isinstance(r, list) else r.get("clusters", [])
        out["reinit"] = {"n_clusters": len(rows),
                         "scour_depth_m": _pcts(x.get("scour_depth_m") for x in rows)}
    try:
        from pipeline_io import discover_surveys
        for d, p in discover_surveys(cfg):
            if d.isoformat() == date:
                out["tif"] = str(p)
                out["tif_mtime_utc"] = dt.datetime.fromtimestamp(
                    Path(p).stat().st_mtime, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except Exception as e:                       # noqa: BLE001
        _warn("survey TIF lookup", e)
    try:
        ev = cfg.avalanche_events_file
        if ev and Path(ev).exists():
            events = json.loads(Path(ev).read_text())
            for period, info in (events.items() if isinstance(events, dict) else []):
                if str(period).endswith(date.replace("-", "")) or str(period).endswith(date):
                    out["period"] = {"name": period, **(info if isinstance(info, dict) else {})}
    except Exception as e:                       # noqa: BLE001
        _warn("events lookup", e)
    return out


def survey_record(cfg: ProjectConfig, date: str, exit_code: Optional[int] = None) -> Optional[Path]:
    try:
        return record(cfg, "survey", {**survey_summary(cfg, date), "run_exit_code": exit_code})
    except Exception as e:                       # noqa: BLE001
        _warn("survey record", e)
        return None


# ---------------------------------------------------------------------------
# Cycle record: one published release set
# ---------------------------------------------------------------------------

def cycle_record(cfg: ProjectConfig, final: Path, funnel: Optional[dict] = None,
                 near_miss: Optional[list] = None) -> Optional[Path]:
    """Called by analysis_pipeline.step_scenarios after a set is published."""
    try:
        final = Path(final)
        man = json.loads((final / "manifest.json").read_text())
        meta = json.loads((final / "metadata.json").read_text()) \
            if (final / "metadata.json").exists() else {}
        fields = {k: man.get(k) for k in ("snapshot_date", "cycle_date", "horizon",
                                           "snapshot_ts", "store", "clamped", "status",
                                           "n_scenarios", "trigger_source")}
        fields["set"] = str(final)
        fields["shots"] = final.parent.name == "shots" or final.name == "shots"
        for k in ("n_trigger_locations", "n_below_depth_floor", "max_hs_m", "p_no_release"):
            if k in meta:
                fields[k] = meta[k]
        summ = final / "summary.csv"
        if summ.exists():
            import pandas as pd
            df = pd.read_csv(summ)
            fields["release_area_m2"] = _pcts(df["release_area_m2"]) if "release_area_m2" in df else None
            fields["mean_depth_m"] = _pcts(df["mean_depth_m"]) if "mean_depth_m" in df else None
            fields["trigger_clusters"] = sorted(int(c) for c in df["trigger_cluster"].unique()) \
                if "trigger_cluster" in df else None
        if funnel is not None:
            fields["funnel"] = funnel
        if near_miss is not None:
            fields["near_miss"] = near_miss
        return record(cfg, "cycle", fields)
    except Exception as e:                       # noqa: BLE001
        _warn("cycle record", e)
        return None


# ---------------------------------------------------------------------------
# Keep what a rerun or the next cycle would overwrite
# ---------------------------------------------------------------------------

def keep_forecast_store(cfg: ProjectConfig, cycle_date: str, timestamps) -> Optional[Path]:
    """Copy the forecast store's snapshot timesteps (the T_n read times) to
    issued/<cycle>/forecast_store.zarr. The store itself is rebuilt with
    mode='w' every cycle, so nothing else keeps a T+1/T+2 profile."""
    try:
        import pandas as pd
        import xarray as xr
        ts = sorted({pd.Timestamp(t) for t in timestamps})
        if not ts or not cfg.forecast_zarr_path.exists():
            return None
        ds = xr.open_zarr(cfg.forecast_zarr_path)
        sub = ds.sel(time=ts, method="nearest")
        sub = sub.isel(time=~pd.Index(sub.time.values).duplicated())
        out = issued_cycle_dir(cfg, cycle_date) / "forecast_store.zarr"
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + ".tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        for v in sub.variables.values():
            v.encoding.pop("chunks", None)
            v.encoding.pop("preferred_chunks", None)
        sub.chunk({"time": 1}).to_zarr(tmp, mode="w")
        if out.exists():
            # A rerun of the cycle: keep the snapshot it replaces.
            old = superseded_dir(cfg, cycle_date) / out.name
            old.parent.mkdir(parents=True, exist_ok=True)
            shutil.rmtree(old, ignore_errors=True)
            shutil.move(str(out), str(old))
        tmp.rename(out)
        print(f"  metrics: kept forecast store at {', '.join(f'{t:%m-%dT%H}' for t in ts)} "
              f"-> {out}", flush=True)
        return out
    except Exception as e:                       # noqa: BLE001
        _warn("forecast-store snapshot", e)
        return None


def superseded_dir(cfg: ProjectConfig, cycle_date: str) -> Path:
    """Where one run puts everything it replaced for a cycle."""
    return issued_cycle_dir(cfg, cycle_date) / f"superseded_{run_id()}"


def keep_published_reports(cfg: ProjectConfig, cycle_date: str) -> Optional[Path]:
    """Before runout republishes a date: copy the web folders already there
    (model and shots) into issued/<cycle>/superseded_<utc>/www/. A copy, so
    the page forecasters have open stays up while the rerun runs."""
    try:
        import alerts
        s = alerts.settings(cfg.toml_path)
        if not (s.get("publish_root") and s.get("path_profile")):
            return None
        day = str(cycle_date).replace("-", "")
        root = Path(s["publish_root"])
        srcs = [(sub, root / sub / "forecasts" / s["path_profile"] / day)
                for sub in ("", "shots")]
        srcs = [(sub, p) for sub, p in srcs if p.is_dir()]
        if not srcs:
            return None
        dest = superseded_dir(cfg, cycle_date) / "www"
        for sub, p in srcs:
            shutil.copytree(p, dest / (sub or "model"))
        print(f"  metrics: kept the published {cycle_date} reports -> {dest}", flush=True)
        return dest
    except Exception as e:                       # noqa: BLE001
        _warn("superseded reports", e)
        return None


# ---------------------------------------------------------------------------
# Observations: ground truth, entered by hand (Ron emails them; Claude enters)
# ---------------------------------------------------------------------------

def observe(cfg: ProjectConfig, obs_type: str, time_local: str, source: str,
            observer: Optional[str] = None, fields: Optional[dict] = None,
            note: Optional[str] = None) -> Path:
    """Append one observation. Unlike the automatic records this one raises:
    it runs by hand, and a typo should be seen, not swallowed."""
    if obs_type not in OBSERVATION_TYPES:
        raise ValueError(f"type must be one of {OBSERVATION_TYPES}, not {obs_type!r}")
    t = dt.datetime.fromisoformat(time_local)
    rec = {"type": obs_type, "time_local": t.isoformat(timespec="minutes"),
           "source": source, "observer": observer,
           "linked_cycle": t.date().isoformat(), **(fields or {})}
    if note:
        rec["note"] = note
    out = season_dir(cfg, cfg.metrics_dir) / FILES["observation"]
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a") as f:
        f.write(json.dumps({**_common(cfg, "observation"), **rec}, default=_jsonable) + "\n")
    return out


def _kv(items) -> dict:
    out = {}
    for it in items or []:
        k, sep, v = it.partition("=")
        if not sep:
            raise ValueError(f"--field wants key=value, got {it!r}")
        try:
            out[k] = json.loads(v)
        except ValueError:
            out[k] = v
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--toml", required=True, type=Path)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--kind", required=True, choices=("daily", "survey"))
    r.add_argument("--exit-code", type=int, required=True)
    r.add_argument("--log", type=Path)
    r.add_argument("--steps", default="")
    r.add_argument("--total-s", type=int)
    r.add_argument("--runout-status", default="")
    r.add_argument("--failed-step", default="")
    r.add_argument("--snapshot", default="")
    s = sub.add_parser("survey")
    s.add_argument("--date", required=True)
    s.add_argument("--exit-code", type=int)
    o = sub.add_parser("observe")
    o.add_argument("--type", required=True, choices=OBSERVATION_TYPES)
    o.add_argument("--time", required=True, help="local time, 'YYYY-MM-DD HH:MM'")
    o.add_argument("--source", required=True)
    o.add_argument("--observer")
    o.add_argument("--field", action="append", help="key=value (JSON values allowed)")
    o.add_argument("--note")
    a = ap.parse_args(argv)

    if a.cmd == "observe":
        cfg = ProjectConfig.from_toml(a.toml)
        out = observe(cfg, a.type, a.time, a.source, a.observer, _kv(a.field), a.note)
        print(f"observation appended: {out}")
        return 0
    # Automatic records: never fail the caller.
    try:
        cfg = ProjectConfig.from_toml(a.toml)
        if a.cmd == "run":
            run_record(cfg, a.kind, a.exit_code, a.log, a.steps, a.total_s,
                       a.runout_status, a.failed_step, a.snapshot)
        elif a.cmd == "survey":
            survey_record(cfg, a.date, a.exit_code)
    except Exception as e:                       # noqa: BLE001
        _warn(a.cmd, e)
        traceback.print_exc(limit=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())

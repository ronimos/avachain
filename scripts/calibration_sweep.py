#!/usr/bin/env python3
"""Calibration sensitivity sweep of the release-area model (CLAUDE.md "Calibration").

Ron, 2026-11-08: calibrate to improve the Jan 18 IoU, and treat the no-event
days as "keep their expected release area from growing". One knob at a time:

    python scripts/calibration_sweep.py --knob c_fit --values 0.03 0.045 0.06 --jobs 8
    python scripts/calibration_sweep.py --knob size_scale --values 0.8 1.0 1.2 --jobs 8
    python scripts/calibration_sweep.py --summarise outputs/little_prof/calibration/<run>

For every value it writes a copy of the slope TOML with a [calibration] block,
and for every date runs analyze (c_fit only: it moves A_ca_brittle; size_scale
shares one default analyze per date), scenarios, then event_scoring. Each job
appends one JSON line to results.jsonl, so an interrupted sweep keeps what
finished and a rerun skips it.

Dates: the Jan 18 event and the 10 no-event survey days of the 2026-10-28
scoring (outputs/little_prof/no_event/run_no_event_survey.py): survey periods
labelled wind_scour, 2026-01-29 .. 03-26, including 03-07 (Ron: no avalanche).
On no-event days the score is the expected release area (event_scoring); the
false-alarm count cannot move with these knobs, because the trigger gates
decide it before any polygon exists.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY = str(REPO / ".venv" / "bin" / "python")
TOML = REPO / "slopes" / "little_prof" / "slope_config.toml"
EVENT = "2026-01-18"
NO_EVENT = ["2026-01-29", "2026-02-05", "2026-02-12", "2026-02-19", "2026-02-26",
            "2026-03-05", "2026-03-07", "2026-03-12", "2026-03-19", "2026-03-26"]
KNOBS = ("c_fit", "size_scale")


def _vtag(v: float) -> str:
    return f"{v:g}".replace(".", "p")


def write_toml(out: Path, knob: str, value: float, base: Path = TOML) -> Path:
    text = base.read_text()
    if "\n[calibration]" in text:
        sys.exit(f"{base} already has [calibration]: sweep a slope without one")
    d = out / f"{knob}_{_vtag(value)}"
    d.mkdir(parents=True, exist_ok=True)
    t = d / "slope_config.toml"
    t.write_text(text + f'\n[calibration]\nversion = "sweep-{knob}-{value:g}"\n'
                        f"{knob} = {value}\n")
    return t


def _run(cmd, log: Path) -> int:
    with open(log, "w") as f:
        return subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT).returncode


def analyze(toml: Path, date: str, feat: Path, log: Path) -> int:
    feat.mkdir(parents=True, exist_ok=True)
    return _run([PY, "src/avachain/analysis_pipeline.py", "analyze", "--toml", str(toml),
                 "--snapshot-date", date, "--features-dir", str(feat), "--force"], log)


def job(out: Path, knob: str, value: float, date: str, feat: Path, toml: Path) -> dict:
    tag = f"cal_{out.name}_{knob}_{_vtag(value)}"
    logs = out / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    row = {"knob": knob, "value": value, "date": date, "event": date == EVENT, "tag": tag}
    t0 = time.time()
    if knob == "c_fit":
        rc = analyze(toml, date, feat, logs / f"analyze_{date}_{tag}.log")
        if rc:
            return {**row, "status": f"analyze failed rc={rc}"}
    slog = logs / f"scenarios_{date}_{tag}.log"
    rc = _run([PY, "src/avachain/analysis_pipeline.py", "scenarios", "--toml", str(toml),
               "--snapshot-date", date, "--features-dir", str(feat),
               "--forecast-horizon", tag], slog)
    if rc:
        return {**row, "status": f"scenarios failed rc={rc}", "log": str(slog)}
    text = slog.read_text(errors="replace")
    import re
    row["n_triggers"] = len(re.findall(r"^\s*Trigger \d", text, re.M)) or None
    m = re.search(r"Trigger clusters \((\w+)\): \[([^\]]*)\]", text)
    if m:
        row["trigger_source"] = m[1]
        row["triggers"] = [int(x) for x in m[2].split(",") if x.strip()]
    scen = REPO / "outputs" / "little_prof" / "scenarios" / date / tag
    s = subprocess.run([PY, "src/avachain/event_scoring.py", "--toml", str(toml),
                        "--scen-dir", str(scen), "--date", date],
                       cwd=REPO, capture_output=True, text=True)
    try:
        row.update(json.loads(s.stdout))
        row["status"] = "ok"
    except ValueError:
        row["status"] = f"scoring failed: {s.stderr[-300:]}"
    row["seconds"] = round(time.time() - t0)
    return row


def done_keys(res: Path) -> set:
    if not res.exists():
        return set()
    return {(r["knob"], r["value"], r["date"]) for r in map(json.loads, res.read_text().splitlines())
            if r.get("status") == "ok"}


def sweep(knob: str, values: list, dates: list, jobs: int, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    res = out / "results.jsonl"
    done = done_keys(res)
    tomls = {v: write_toml(out, knob, v) for v in values}
    if knob == "size_scale":
        # One default analyze per date, shared by every size_scale value.
        base = out / "analyze_default"
        with cf.ThreadPoolExecutor(jobs) as ex:
            futs = {ex.submit(analyze, TOML, d, base / d, out / "logs" / f"analyze_{d}_default.log"): d
                    for d in dates if not (base / d / f"all_start_zone_features_{d}.csv").exists()}
            for f in cf.as_completed(futs):
                if f.result():
                    sys.exit(f"analyze failed for {futs[f]}")
        feat = {(v, d): base / d for v in values for d in dates}
    else:
        feat = {(v, d): out / f"{knob}_{_vtag(v)}" / "features" / d for v in values for d in dates}
    todo = [(v, d) for v in values for d in dates if (knob, v, d) not in done]
    print(f"{knob}: {len(values)} values x {len(dates)} dates, {len(todo)} to run, "
          f"{jobs} at a time -> {out}", flush=True)
    with cf.ThreadPoolExecutor(jobs) as ex:
        futs = {ex.submit(job, out, knob, v, d, feat[(v, d)], tomls[v]): (v, d) for v, d in todo}
        for f in cf.as_completed(futs):
            r = f.result()
            with open(res, "a") as fh:
                fh.write(json.dumps(r) + "\n")
            v, d = futs[f]
            print(f"  {knob}={v:g} {d}: {r['status']} "
                  f"E[IoU]={r.get('expected_iou', '-')} E[area]={r.get('expected_area_m2', '-')}",
                  flush=True)
    summarise(out)


def summarise(out: Path) -> None:
    import numpy as np
    rows = [r for r in map(json.loads, (out / "results.jsonl").read_text().splitlines())
            if r.get("status") == "ok"]
    by = {}
    for r in rows:
        by.setdefault((r["knob"], r["value"]), []).append(r)
    lines = ["| knob | value | Jan 18 E[IoU] | Jan 18 E[area] m² | no-event median E[area] m² "
             "| no-event max | n no-event |", "|---|---|---|---|---|---|---|"]
    table = []
    for (k, v), rs in sorted(by.items()):
        ev = [r for r in rs if r["event"]]
        ne = [r["expected_area_m2"] for r in rs if not r["event"] and "expected_area_m2" in r]
        e = ev[0] if ev else {}
        rec = {"knob": k, "value": v, "jan18_expected_iou": e.get("expected_iou"),
               "jan18_expected_area_m2": e.get("expected_area_m2"),
               "jan18_auc": e.get("auc"), "jan18_bss": e.get("bss"),
               "no_event_median_area_m2": float(np.median(ne)) if ne else None,
               "no_event_max_area_m2": float(np.max(ne)) if ne else None,
               "no_event_n": len(ne),
               "no_event_area_by_date": {r["date"]: r.get("expected_area_m2")
                                         for r in rs if not r["event"]}}
        table.append(rec)
        f = lambda x, n=0: "-" if x is None else f"{x:.{n}f}"
        lines.append(f"| {k} | {v:g} | {f(rec['jan18_expected_iou'], 3)} | "
                     f"{f(rec['jan18_expected_area_m2'])} | {f(rec['no_event_median_area_m2'])} | "
                     f"{f(rec['no_event_max_area_m2'])} | {len(ne)} |")
    (out / "summary.json").write_text(json.dumps(table, indent=1))
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--knob", choices=KNOBS)
    ap.add_argument("--values", type=float, nargs="+")
    ap.add_argument("--dates", nargs="+", default=[EVENT] + NO_EVENT)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--summarise", type=Path)
    a = ap.parse_args(argv)
    if a.summarise:
        summarise(a.summarise)
        return 0
    if not (a.knob and a.values):
        ap.error("--knob and --values are required")
    out = a.out or REPO / "outputs" / "little_prof" / "calibration" / f"{time.strftime('%Y%m%d')}_{a.knob}"
    sweep(a.knob, sorted(a.values), a.dates, a.jobs, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

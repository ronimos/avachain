"""
survey_markers.py — survey-arrival markers shared with the survey producer.

Each survey raster arrives as YYMMDD_<path>_snowHeight.tif with a sibling
marker YYMMDD_<path>_snowHeight.<STATE>. The producer only ever writes READY;
this pipeline moves it on:

    READY    new survey, not started (the producer wrote it after the TIF)
    RUNNING  claimed by a run; the rename is the claim, so two overlapping
             cron runs cannot take the same survey
    DONE     the run finished
    FAILED   the run failed; rename it back to READY to retry

The state lives in the file name. The body is a JSON record of the run that
holds it: who, when, which commit and TOML, where the log is, and on FAILED
the step and the log tail. A retry keeps earlier records under "history",
and anything the producer wrote into READY is kept under "ready".

Only RUNNING and DONE surveys are visible to the pipeline
(pipeline_io.discover_surveys, when [surveys] markers = true). A bare TIF may
still be copying, READY has not been taken in yet (so a run never sees a
newer survey than the one it claimed), and FAILED never completed.

CLI (called by run_<slope>.sh):
    python survey_markers.py --toml X claim --pid $$ --log L   # prints the date, or nothing
    python survey_markers.py --toml X finish --survey-date D --status DONE|FAILED \\
        --exit-code N [--step S] [--log L] [--output DIR ...]
    python survey_markers.py --toml X status
"""

from __future__ import annotations

import datetime as _dt
import getpass
import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Optional

from config import ProjectConfig

STATES = ("READY", "RUNNING", "DONE", "FAILED")
VISIBLE = ("RUNNING", "DONE")
LOG_TAIL_LINES = 20

_REPO = Path(__file__).resolve().parent.parent.parent


class OutOfOrder(RuntimeError):
    """A READY survey is older than one already DONE."""


def marker_path(tif: Path, state: str) -> Path:
    return tif.with_suffix(f".{state}")


def state_of(tif: Path) -> Optional[str]:
    """The marker state of a survey TIF, or None for a bare TIF.

    More than one marker for a TIF is an inconsistency nobody should have to
    guess about, so it is an error rather than a precedence rule.
    """
    found = [s for s in STATES if marker_path(tif, s).exists()]
    if len(found) > 1:
        raise RuntimeError(f"{tif.name}: several markers {found}; keep one")
    return found[0] if found else None


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _commit() -> Optional[str]:
    try:
        return subprocess.run(["git", "-C", str(_REPO), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _read(path: Path) -> dict:
    """The JSON record in a marker; a non-JSON body is the producer's, kept verbatim."""
    text = path.read_text() if path.exists() else ""
    if not text.strip():
        return {}
    try:
        rec = json.loads(text)
        return rec if isinstance(rec, dict) else {"ready": rec}
    except json.JSONDecodeError:
        return {"ready": text}


def _write(path: Path, rec: dict) -> None:
    """Replace a marker's body atomically: nothing ever reads half a record."""
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(rec, indent=2) + "\n")
    os.replace(tmp, path)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:          # alive, owned by another user
        return True
    return True


def _log_tail(log: Optional[str]) -> Optional[list]:
    if not log or not Path(log).exists():
        return None
    return Path(log).read_text(errors="replace").splitlines()[-LOG_TAIL_LINES:]


def survey_tifs(cfg: ProjectConfig) -> list:
    """(date, tif) for every survey TIF, sorted by date, regardless of marker."""
    from pipeline_io import parse_survey_date
    return sorted(((parse_survey_date(f.name), f)
                   for f in cfg.survey_dir.glob(cfg.survey_glob)), key=lambda x: x[0])


def _move(tif: Path, src: str, dst: str, rec: dict) -> None:
    """Rename src → dst (the state change), then write the record into dst."""
    os.rename(marker_path(tif, src), marker_path(tif, dst))
    _write(marker_path(tif, dst), rec)


def claim(cfg: ProjectConfig, pid: int, log: Optional[str] = None) -> Optional[_dt.date]:
    """Claim the oldest READY survey: READY → RUNNING. None if there is nothing
    to do or another live run holds a RUNNING marker.

    A RUNNING marker whose process is gone on this host is a crash the run's
    own trap never saw (kill -9, reboot); it is recorded as FAILED, so a
    retry is the same READY rename as any other failure.
    """
    host = socket.gethostname()
    tifs = survey_tifs(cfg)
    for d, tif in tifs:
        if state_of(tif) != "RUNNING":
            continue
        rec = _read(marker_path(tif, "RUNNING"))
        if rec.get("host") == host and rec.get("pid") and not _pid_alive(int(rec["pid"])):
            rec.update(status="FAILED", finished_utc=_now(), exit_code=None,
                       failed_step=rec.get("step"),
                       reason=f"stale: pid {rec['pid']} gone, found by pid {pid}",
                       log_tail=_log_tail(rec.get("log")))
            _move(tif, "RUNNING", "FAILED", rec)
            print(f"{tif.name}: stale RUNNING (pid {rec['pid']} gone) → FAILED",
                  file=sys.stderr)
            continue
        print(f"{tif.name} is RUNNING (host {rec.get('host')}, pid {rec.get('pid')}); "
              f"not claiming another survey", file=sys.stderr)
        return None

    done = [d for d, tif in tifs if state_of(tif) == "DONE"]
    for d, tif in tifs:
        if state_of(tif) != "READY":
            continue
        if done and d < max(done):
            # The survey cycle restarts from the previous survey's anchor; a
            # survey older than one already done cannot be slotted in behind it.
            raise OutOfOrder(f"{tif.name} is READY but {max(done)} is already DONE; "
                             f"surveys must run in date order (rebuild, or mark it DONE)")
        ready = marker_path(tif, "READY")
        prev = _read(ready)
        try:
            os.rename(ready, marker_path(tif, "RUNNING"))
        except FileNotFoundError:    # another run took it between the check and here
            return None
        history = prev.pop("history", [])
        if "status" in prev:         # a FAILED/DONE record flipped back by hand
            history.append(prev)
            prev = {}
        rec = {"survey": tif.name, "survey_date": d.isoformat(), "status": "RUNNING",
               "started_utc": _now(), "host": host, "user": getpass.getuser(),
               "pid": pid, "commit": _commit(),
               "toml": str(cfg.toml_path) if cfg.toml_path else None, "log": log}
        if prev:
            rec["ready"] = prev.get("ready", prev)
        if history:
            rec["history"] = history
        _write(marker_path(tif, "RUNNING"), rec)
        return d
    return None


def finish(cfg: ProjectConfig, survey_date: _dt.date, status: str, exit_code: int,
           step: Optional[str] = None, log: Optional[str] = None,
           outputs: Optional[list] = None) -> Path:
    """RUNNING → DONE or FAILED, with the outcome added to the record."""
    if status not in ("DONE", "FAILED"):
        raise ValueError(f"status must be DONE or FAILED, not {status!r}")
    tif = next((f for d, f in survey_tifs(cfg) if d == survey_date), None)
    if tif is None or state_of(tif) != "RUNNING":
        raise RuntimeError(f"no RUNNING survey for {survey_date}")
    rec = _read(marker_path(tif, "RUNNING"))
    started = rec.get("started_utc")
    rec.update(status=status, finished_utc=_now(), exit_code=exit_code)
    if started:
        t0 = _dt.datetime.strptime(started, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=_dt.timezone.utc)
        rec["duration_s"] = int((_dt.datetime.now(_dt.timezone.utc) - t0).total_seconds())
    if log:
        rec["log"] = log
    if outputs:
        rec["outputs"] = outputs
    if status == "FAILED":
        rec["failed_step"] = step
        rec["log_tail"] = _log_tail(rec.get("log"))
    _move(tif, "RUNNING", status, rec)
    return marker_path(tif, status)


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--toml", required=True)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("claim")
    c.add_argument("--pid", type=int, required=True, help="the run's shell pid ($$)")
    c.add_argument("--log")
    f = sub.add_parser("finish")
    f.add_argument("--survey-date", required=True, type=_dt.date.fromisoformat)
    f.add_argument("--status", required=True, choices=("DONE", "FAILED"))
    f.add_argument("--exit-code", type=int, required=True)
    f.add_argument("--step")
    f.add_argument("--log")
    f.add_argument("--output", action="append")
    sub.add_parser("status")
    args = ap.parse_args(argv)
    cfg = ProjectConfig.from_toml(args.toml)

    if args.cmd == "claim":
        try:
            d = claim(cfg, args.pid, args.log)
        except OutOfOrder as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 2
        if d:
            print(d.isoformat())
    elif args.cmd == "finish":
        print(finish(cfg, args.survey_date, args.status, args.exit_code,
                     args.step, args.log, args.output))
    else:
        for d, tif in survey_tifs(cfg):
            print(f"{d.isoformat()}  {state_of(tif) or '-':8s} {tif.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

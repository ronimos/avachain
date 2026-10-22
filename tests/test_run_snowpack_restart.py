"""Behaviour of the generated run_snowpack.sh at restart, against a stub binary.

Until 2026-10-20 every restart was given -b = BDATE, and every operational
caller passed an empty BDATE, so -b fell back to the first SMET row while
input/snow held the restart state. SNOWPACK then simulated the whole season
again on top of that state. On cluster 2859, restarted from 2026-02-05, HS went
1.82 -> 2.03 m and 91 -> 107 elements, and the output ProfileDate still looked
correct.

The stub stands in for SNOWPACK. It records the -b/-e it was given and writes
a restart .sno dated -e, the way SNOWPACK does, so these tests check the
script's choice of restart file and begin date without running the model.
"""
import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "slopes" / "little_prof_replay" / "run_snowpack.sh"

STUB = r"""#!/bin/bash
# Stub SNOWPACK: record -b/-e, write <out>/<exp>_<exp>.sno dated -e.
while [[ $# -gt 0 ]]; do
  case "$1" in
    -c) ini="$2"; shift 2 ;;
    -b) b="$2"; shift 2 ;;
    -e) e="$2"; shift 2 ;;
    *) shift ;;
  esac
done
exp=$(awk -F'= *' '/^EXPERIMENT/{print $2}' "$ini")
out=$(awk -F'= *' '/^\[Output\]/{o=1} o && /^SNOWPATH/{print $2; exit}' "$ini")
snin=$(awk -F'= *' '/^\[Input\]/{i=1} i && /^SNOWPATH/{print $2; exit}' "$ini")
echo "BEGIN=$b END=$e"
echo "INPUT_PROFILEDATE=$(awk -F'= *' '/^ProfileDate/{print $2}' "$snin/$exp.sno")"
step=$(awk -F'= *' '/^CALCULATION_STEP_LENGTH/{print $2}' "$ini")
echo "STEP=${step:-master}"
if [[ -n "${STUB_CRASH:-}" && ( -z "$step" || "$STUB_CRASH" == always ) ]]; then
  # SNOWPACK's surface-node failure: erases its output .sno, writes no new one.
  echo "[E] [Snowpack.cc:1043] Temperature out of bound at beginning of iteration!"
  rm -f "$out/${exp}_${exp}.sno"
  exit 1
elif [[ -n "${STUB_STALL:-}" ]]; then
  # SNOWPACK with ENFORCE_MEASURED_SNOW_HEIGHTS and HS=-999 in the forcing:
  # logs "missing { HS }", leaves the restart file where it was, and still
  # prints FINISHED.
  echo "[E] $exp missing { HS }"
  [[ -f "$out/${exp}_${exp}.sno" ]] || printf 'ProfileDate      = %s\n' "$b" > "$out/${exp}_${exp}.sno"
else
  printf 'ProfileDate      = %s\n' "$e" > "$out/${exp}_${exp}.sno"
fi
echo "FINISHED running stub"
"""

SMET = """SMET 1.1 ASCII
[HEADER]
station_id = {cid}
latitude = 39.6
longitude = -105.9
altitude = 3600
slope_angle = 31.0
slope_azi = 120.0
fields = timestamp TA RH VW HS
[DATA]
2025-11-26T18:00:00 -5.0 60 2.0 50
2026-02-05T12:00:00 -6.0 46 3.9 180
2026-02-06T12:00:00 -7.0 50 2.0 182
"""


@pytest.fixture
def slope(tmp_path):
    d = tmp_path / "slope"
    for sub in ("input/smet", "input/snow", "output"):
        (d / sub).mkdir(parents=True)
    tmpl = REPO / "snowpack" / "little_prof_replay" / "input" / "snow" / "template.sno"
    (d / "input/snow/template.sno").write_text(
        tmpl.read_text() if tmpl.exists() else "ProfileDate = WRFDATE\n")
    stub = tmp_path / "stub_snowpack"
    stub.write_text(STUB)
    stub.chmod(0o755)
    return d, stub


def _add_cluster(d, cid):
    (d / "input/smet" / f"{cid}.smet").write_text(SMET.format(cid=cid))


def _sno(d, name, profile_date):
    (d / "output" / name).write_text(f"ProfileDate      = {profile_date}\n")


def _run(d, stub, *args, **env):
    e = dict(os.environ, SLOPE_DIR=str(d), SMET_DIR=str(d / "input/smet"),
             SNOWPACK_BIN=str(stub), SKIP_ZARR="1", **env)
    return subprocess.run(["bash", str(SCRIPT), *args], env=e,
                          capture_output=True, text=True, timeout=60)


def _log(d, cid):
    return (d / "output" / f"{cid}.log").read_text()


def test_fresh_cluster_starts_at_first_smet_row(slope):
    d, stub = slope
    _add_cluster(d, "cluster_0001")
    r = _run(d, stub, "", "2026-02-05T12:00")
    assert r.returncode == 0, r.stderr
    assert "BEGIN=2025-11-26T18:00:00 " in _log(d, "cluster_0001")


def test_restart_begins_at_its_own_profiledate_not_bdate(slope):
    """The bug: empty BDATE + restart -> -b was the first SMET row."""
    d, stub = slope
    _add_cluster(d, "cluster_0001")
    _sno(d, "cluster_0001_cluster_0001.sno", "2026-02-05T12:00:00")
    r = _run(d, stub, "", "2026-02-06T12:00")
    assert r.returncode == 0, r.stderr
    assert "BEGIN=2026-02-05T12:00:00 " in _log(d, "cluster_0001")


def test_each_cluster_restarts_at_its_own_date(slope):
    """Reinit leaves scoured clusters at the event time and the rest later."""
    d, stub = slope
    for cid, pdate in (("cluster_0001", "2026-02-05T12:00:00"),
                       ("cluster_0002", "2026-02-03T12:00:00")):
        _add_cluster(d, cid)
        _sno(d, f"{cid}_{cid}.sno", pdate)
    _run(d, stub, "", "2026-02-06T12:00")
    assert "BEGIN=2026-02-05T12:00:00 " in _log(d, "cluster_0001")
    assert "BEGIN=2026-02-03T12:00:00 " in _log(d, "cluster_0002")


def test_explicit_bdate_does_not_override_a_restart(slope):
    d, stub = slope
    _add_cluster(d, "cluster_0001")
    _sno(d, "cluster_0001_cluster_0001.sno", "2026-02-05T12:00:00")
    _run(d, stub, "2025-11-26T18:00", "2026-02-06T12:00")
    assert "BEGIN=2026-02-05T12:00:00 " in _log(d, "cluster_0001")


@pytest.mark.parametrize("mode,fname", [("anchor", "cluster_0001_anchor.sno"),
                                        ("survey", "cluster_0001_survey.sno")])
def test_named_anchor_wins_over_latest_state(slope, mode, fname):
    """The latest .sno may hold a +48 h forecast state; the anchor must win."""
    d, stub = slope
    _add_cluster(d, "cluster_0001")
    _sno(d, "cluster_0001_cluster_0001.sno", "2026-02-07T12:00:00")  # forecast end
    _sno(d, fname, "2026-02-05T12:00:00")
    r = _run(d, stub, "", "2026-02-06T12:00", RESTART_FROM=mode)
    assert r.returncode == 0, r.stderr
    log = _log(d, "cluster_0001")
    assert "BEGIN=2026-02-05T12:00:00 " in log
    assert "INPUT_PROFILEDATE=2026-02-05T12:00:00" in log


def test_missing_named_anchor_fails_instead_of_falling_back(slope):
    d, stub = slope
    _add_cluster(d, "cluster_0001")
    _sno(d, "cluster_0001_cluster_0001.sno", "2026-02-07T12:00:00")
    r = _run(d, stub, "", "2026-02-06T12:00", RESTART_FROM="survey")
    assert r.returncode == 2
    assert "cluster_0001" in r.stderr
    assert not (d / "output" / "cluster_0001.log").exists()


def test_unknown_restart_mode_is_rejected(slope):
    d, stub = slope
    _add_cluster(d, "cluster_0001")
    r = _run(d, stub, "", "2026-02-06T12:00", RESTART_FROM="yesterday")
    assert r.returncode == 1


def test_a_run_that_does_not_reach_the_end_date_is_a_failure(slope):
    """FINISHED + a non-empty .sno is not success: the forecast pass stalled on
    HS=-999 at its first step, printed FINISHED, and was counted OK."""
    d, stub = slope
    _add_cluster(d, "cluster_0001")
    _sno(d, "cluster_0001_cluster_0001.sno", "2026-02-06T12:00:00")
    r = _run(d, stub, "", "2026-02-08T12:00", STUB_STALL="1")
    assert "OK: 0   Failed: 1" in r.stdout
    assert "did not reach 2026-02-08T12:00" in r.stdout


def test_a_run_that_reaches_the_end_date_is_ok(slope):
    d, stub = slope
    _add_cluster(d, "cluster_0001")
    _sno(d, "cluster_0001_cluster_0001.sno", "2026-02-06T12:00:00")
    r = _run(d, stub, "", "2026-02-08T12:00")
    assert "OK: 1   Failed: 0" in r.stdout


def test_a_temperature_bound_crash_is_retried_at_a_shorter_step(slope):
    """Replay tick 2026-03-06: 4 clusters crashed at 15 min, all ran at 5."""
    d, stub = slope
    _add_cluster(d, "cluster_0001")
    _sno(d, "cluster_0001_anchor.sno", "2026-02-05T12:00:00")
    r = _run(d, stub, "", "2026-02-06T12:00", RESTART_FROM="anchor", STUB_CRASH="1")
    assert "OK: 1   Failed: 0" in r.stdout, r.stdout
    assert "Re-run at 5 min" in r.stdout
    first = (d / "output" / "cluster_0001.log.first").read_text()
    assert "STEP=master" in first and "out of bound" in first
    log = _log(d, "cluster_0001")
    assert "STEP=5" in log
    # Same restart and begin date as the first attempt
    assert "BEGIN=2026-02-05T12:00:00 " in log
    assert "INPUT_PROFILEDATE=2026-02-05T12:00:00" in log


def test_a_crash_that_persists_at_the_shorter_step_still_fails(slope):
    d, stub = slope
    _add_cluster(d, "cluster_0001")
    _sno(d, "cluster_0001_anchor.sno", "2026-02-05T12:00:00")
    r = _run(d, stub, "", "2026-02-06T12:00", RESTART_FROM="anchor",
             STUB_CRASH="always")
    assert "OK: 0   Failed: 1" in r.stdout


def test_retry_can_be_disabled(slope):
    d, stub = slope
    _add_cluster(d, "cluster_0001")
    _sno(d, "cluster_0001_anchor.sno", "2026-02-05T12:00:00")
    r = _run(d, stub, "", "2026-02-06T12:00", RESTART_FROM="anchor",
             STUB_CRASH="1", RETRY_STEP_MIN="0")
    assert "OK: 0   Failed: 1" in r.stdout
    assert not (d / "output" / "cluster_0001.log.first").exists()


def test_only_crashed_clusters_are_retried(slope):
    """A stalled forecast (HS=-999) fails the same way at any step: no retry."""
    d, stub = slope
    _add_cluster(d, "cluster_0001")
    _sno(d, "cluster_0001_cluster_0001.sno", "2026-02-06T12:00:00")
    r = _run(d, stub, "", "2026-02-08T12:00", STUB_STALL="1")
    assert "OK: 0   Failed: 1" in r.stdout
    assert "Re-run" not in r.stdout
    assert not (d / "output" / "cluster_0001.log.first").exists()

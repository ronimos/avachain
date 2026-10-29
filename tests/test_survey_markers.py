"""
tests/test_survey_markers.py — survey-arrival markers (READY → RUNNING → DONE/FAILED).

The producer writes YYMMDD_<path>_snowHeight.tif and then a .READY marker;
the pipeline claims by renaming. These pin what the deployment relies on:
which surveys the pipeline can see, that a claim is exclusive, that a crash
cannot leave a survey RUNNING forever, and that the operational script's
trap actually records the outcome.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src" / "avachain"))

import survey_markers as sm                     # noqa: E402
from config import ProjectConfig                 # noqa: E402
from pipeline_io import discover_surveys         # noqa: E402

D1, D2, D3 = dt.date(2026, 1, 14), dt.date(2026, 1, 20), dt.date(2026, 1, 27)


def _tif(d: dt.date) -> str:
    return f"{d:%y%m%d}_Professor_PTC_snowHeight.tif"


def _toml(tmp_path: Path, markers: bool = True) -> Path:
    p = tmp_path / "slope_config.toml"
    p.write_text(f'[slope]\nname = "Professor_PTC"\n\n'
                 f'[paths]\nproject_dir = "{tmp_path}"\nsurvey_dir = "surveys"\n'
                 f'output_dir = "out"\nslope_dir = "work"\n\n'
                 f'[surveys]\nmarkers = {"true" if markers else "false"}\n')
    return p


def _survey(tmp_path: Path, d: dt.date, state: str | None, body: str = "") -> Path:
    sdir = tmp_path / "surveys"
    sdir.mkdir(exist_ok=True)
    tif = sdir / _tif(d)
    tif.write_bytes(b"")
    if state:
        sm.marker_path(tif, state).write_text(body)
    return tif


def _dead_pid() -> int:
    p = subprocess.Popen(["true"])
    p.wait()
    return p.pid


@pytest.fixture
def cfg(tmp_path):
    return ProjectConfig.from_toml(_toml(tmp_path))


class TestVisibility:
    def test_only_running_and_done_count(self, tmp_path, cfg):
        _survey(tmp_path, dt.date(2026, 1, 1), None)        # still copying
        _survey(tmp_path, dt.date(2026, 1, 2), "DONE")
        _survey(tmp_path, dt.date(2026, 1, 3), "FAILED")
        _survey(tmp_path, dt.date(2026, 1, 4), "RUNNING")
        _survey(tmp_path, dt.date(2026, 1, 5), "READY")     # newer than the claimed one
        assert [d.day for d, _ in discover_surveys(cfg)] == [2, 4]

    def test_off_sees_every_tif(self, tmp_path):
        _survey(tmp_path, D1, None)
        _survey(tmp_path, D2, "READY")
        cfg = ProjectConfig.from_toml(_toml(tmp_path, markers=False))
        assert [d for d, _ in discover_surveys(cfg)] == [D1, D2]

    def test_two_markers_is_an_error(self, tmp_path):
        tif = _survey(tmp_path, D1, "READY")
        sm.marker_path(tif, "DONE").write_text("")
        with pytest.raises(RuntimeError, match="several markers"):
            sm.state_of(tif)


class TestClaim:
    def test_claims_the_oldest_ready_and_records_the_run(self, tmp_path, cfg):
        _survey(tmp_path, D1, "DONE")
        _survey(tmp_path, D3, "READY")
        tif = _survey(tmp_path, D2, "READY", body="from val\n")
        assert sm.claim(cfg, pid=os.getpid(), log="/x.log") == D2
        assert sm.state_of(tif) == "RUNNING"
        rec = json.loads(sm.marker_path(tif, "RUNNING").read_text())
        assert rec["survey_date"] == "2026-01-20" and rec["status"] == "RUNNING"
        assert rec["pid"] == os.getpid() and rec["log"] == "/x.log"
        assert rec["ready"] == "from val\n"            # the producer's body is kept

    def test_nothing_ready(self, tmp_path, cfg):
        _survey(tmp_path, D1, "DONE")
        assert sm.claim(cfg, pid=os.getpid()) is None

    def test_a_live_run_blocks_another_claim(self, tmp_path, cfg):
        _survey(tmp_path, D1, "READY")
        assert sm.claim(cfg, pid=os.getpid()) == D1
        _survey(tmp_path, D2, "READY")
        assert sm.claim(cfg, pid=os.getpid()) is None
        assert sm.state_of(tmp_path / "surveys" / _tif(D2)) == "READY"

    def test_a_stale_run_is_failed_and_the_next_claimed(self, tmp_path, cfg):
        _survey(tmp_path, D1, "READY")
        sm.claim(cfg, pid=_dead_pid())
        _survey(tmp_path, D2, "READY")
        assert sm.claim(cfg, pid=os.getpid()) == D2
        old = tmp_path / "surveys" / _tif(D1)
        assert sm.state_of(old) == "FAILED"
        assert "stale" in json.loads(sm.marker_path(old, "FAILED").read_text())["reason"]

    def test_out_of_order_is_refused_and_nothing_moves(self, tmp_path, cfg):
        old = _survey(tmp_path, D1, "READY")
        _survey(tmp_path, D2, "DONE")
        with pytest.raises(sm.OutOfOrder):
            sm.claim(cfg, pid=os.getpid())
        assert sm.state_of(old) == "READY"


class TestFinish:
    def test_done(self, tmp_path, cfg):
        tif = _survey(tmp_path, D1, "READY")
        sm.claim(cfg, pid=os.getpid())
        sm.finish(cfg, D1, "DONE", 0, outputs=["/h/release"])
        rec = json.loads(sm.marker_path(tif, "DONE").read_text())
        assert sm.state_of(tif) == "DONE"
        assert rec["exit_code"] == 0 and rec["outputs"] == ["/h/release"]
        assert "duration_s" in rec and "log_tail" not in rec

    def test_failed_keeps_step_and_log_tail_and_a_retry_keeps_history(self, tmp_path, cfg):
        log = tmp_path / "run.log"
        log.write_text("\n".join(f"line {i}" for i in range(50)))
        tif = _survey(tmp_path, D1, "READY")
        sm.claim(cfg, pid=os.getpid(), log=str(log))
        sm.finish(cfg, D1, "FAILED", 3, step="Step 3: SNOWPACK")
        rec = json.loads(sm.marker_path(tif, "FAILED").read_text())
        assert rec["failed_step"] == "Step 3: SNOWPACK" and rec["exit_code"] == 3
        assert rec["log_tail"] == [f"line {i}" for i in range(30, 50)]

        # Retry the way Val described it: flip FAILED back to READY.
        os.rename(sm.marker_path(tif, "FAILED"), sm.marker_path(tif, "READY"))
        sm.claim(cfg, pid=os.getpid())
        rec = json.loads(sm.marker_path(tif, "RUNNING").read_text())
        assert [h["status"] for h in rec["history"]] == ["FAILED"]
        assert "ready" not in rec

    def test_finish_needs_a_running_survey(self, tmp_path, cfg):
        _survey(tmp_path, D1, "READY")
        with pytest.raises(RuntimeError, match="no RUNNING"):
            sm.finish(cfg, D1, "DONE", 0)


class TestOperationalScript:
    """The rendered run_<slope>.sh header: claim, then the EXIT trap records
    the outcome. Cut at the point the pipeline would start and replaced by a
    step that succeeds or fails."""

    def _script(self, tmp_path, tail: str) -> Path:
        spec = __import__("importlib.util").util.spec_from_file_location(
            "install_mod_markers", REPO / "install.py")
        mod = __import__("importlib.util").util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        with open(REPO / "slopes" / "little_prof" / "slope_config.toml", "rb") as fh:
            sh = mod._render_run_operational(tomllib.load(fh))
        head = sh[:sh.index("exec > >(tee")]
        assert "trap _finish_marker EXIT" in head
        head = head.replace('PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
                            f'PROJECT_DIR="{REPO}"')
        head = "\n".join(f'TOML={_toml(tmp_path)}' if l.startswith("TOML=") else l
                         for l in head.splitlines())
        p = tmp_path / "run.sh"
        p.write_text(head + "\n" + tail + "\n")
        return p

    def _run(self, script: Path) -> subprocess.CompletedProcess:
        return subprocess.run(["bash", str(script)], capture_output=True, text=True)

    def test_failure_is_recorded_as_failed_with_the_step(self, tmp_path):
        tif = _survey(tmp_path, D1, "READY")
        r = self._run(self._script(tmp_path, 'CURRENT_STEP="Step 3: SNOWPACK"\nfalse'))
        assert r.returncode == 1
        assert sm.state_of(tif) == "FAILED", r.stderr
        rec = json.loads(sm.marker_path(tif, "FAILED").read_text())
        assert rec["failed_step"] == "Step 3: SNOWPACK" and rec["exit_code"] == 1

    def test_success_is_recorded_as_done(self, tmp_path):
        tif = _survey(tmp_path, D1, "READY")
        r = self._run(self._script(tmp_path, 'echo "claimed $CLAIMED"'))
        assert r.returncode == 0, r.stderr
        assert "claimed 2026-01-14" in r.stdout
        assert sm.state_of(tif) == "DONE"

    def test_nothing_ready_exits_quietly_without_a_log(self, tmp_path):
        _survey(tmp_path, D1, "DONE")
        r = self._run(self._script(tmp_path, 'echo SHOULD-NOT-RUN'))
        assert r.returncode == 0 and "SHOULD-NOT-RUN" not in r.stdout

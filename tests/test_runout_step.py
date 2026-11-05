"""runout_step: runout chained after `scenarios` (Ron, 2026-11-04).

Pins: the set is read only when READY and its manifest checks out; runout's
tree (config, T_0 link) is built under [runout] work_dir; one global lock
serialises runouts on the host (the 4-sim cap must hold across slopes); the
run scripts keep a run successful when runout fails, and a slope without
[runout] skips the step.
"""
import json
import os
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src" / "avachain"))
import handoff  # noqa: E402
import runout_step as rs  # noqa: E402


def _published_set(tmp_path, files=None):
    stage = tmp_path / ".stage"
    stage.mkdir()
    for name, text in (files or {"scour_depth.tif": "raster",
                                  "scenarios/scenario_001/release.geojson": "{}"}).items():
        p = stage / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    final = tmp_path / "release" / "2026-12-05"
    final.parent.mkdir(parents=True)
    handoff.publish(stage, final, "release", path="P", snapshot_date="2026-12-05",
                    status="ok", n_scenarios=1)
    return final


def _toml(tmp_path, runout=True, scen_dir=None):
    t = tmp_path / "slope.toml"
    body = textwrap.dedent(f"""\
        [slope]
        name = "P"
        [paths]
        project_dir = "{tmp_path}"
        scenarios_dir = "{scen_dir or tmp_path / 'release'}"
        """)
    if runout:
        body += textwrap.dedent(f"""\
            [runout]
            path_profile = "P_profile"
            work_dir = "{tmp_path / 'work' / 'runout' / 'P'}"
            publish_root = "{tmp_path / 'www'}"
            """)
    t.write_text(body)
    return t


class TestVerify:

    def test_a_published_set_verifies(self, tmp_path):
        assert handoff.verify(_published_set(tmp_path)) == []

    def test_no_ready(self, tmp_path):
        s = _published_set(tmp_path)
        (s / handoff.READY).unlink()
        assert handoff.verify(s) and "READY" in handoff.verify(s)[0]

    def test_a_changed_file(self, tmp_path):
        s = _published_set(tmp_path)
        (s / "scour_depth.tif").write_text("RASTER")         # same size, other bytes
        assert handoff.verify(s) == ["changed since publish: scour_depth.tif"]

    def test_a_missing_file(self, tmp_path):
        s = _published_set(tmp_path)
        (s / "scenarios" / "scenario_001" / "release.geojson").unlink()
        assert handoff.verify(s) == ["missing scenarios/scenario_001/release.geojson"]


class TestSettings:

    def test_no_runout_table_is_none(self, tmp_path):
        assert rs.runout_settings(_toml(tmp_path, runout=False)) is None

    def test_defaults(self, tmp_path):
        rc = rs.runout_settings(_toml(tmp_path))
        assert rc["profile"] == "forecast"
        assert rc["lock_file"] == tmp_path / "work" / "runout" / ".runout.lock"

    def test_operational_toml(self):
        rc = rs.runout_settings(REPO / "slopes" / "Professor_PTC" / "slope_config.toml")
        assert rc["path_profile"] == "Professor_PTC"
        assert (REPO / "runout" / "config" / "paths" / "Professor_PTC.yaml").exists()
        assert rc["publish_root"] == "/home/www/html/ron/runout_forecast"
        # one lock for every slope on the host
        assert rc["lock_file"] == Path("/home/avachain/work/runout/.runout.lock")


def test_tree_links_the_set_and_the_repo_config(tmp_path):
    s = _published_set(tmp_path)
    rc = rs.runout_settings(_toml(tmp_path))
    for _ in range(2):                                  # a rerun replaces the links
        cfg_dir = rs.prepare_tree(rc, {"T_0": s}, "2026-12-05")
    work = tmp_path / "work" / "runout" / "P"
    assert (work / "scen" / "P_profile" / "20261205" / "T_0").resolve() == s.resolve()
    local = (cfg_dir / "local.yaml").read_text()
    assert f"avachain_root: {work}" in local and f"publish_root: {tmp_path / 'www'}" in local
    for name in ("paths", "physics", "profiles.yaml"):
        assert (cfg_dir / name).resolve() == (rs.RUNOUT / "config" / name).resolve()


def test_command(tmp_path):
    s = _published_set(tmp_path)
    rc = rs.runout_settings(_toml(tmp_path))
    cmd = rs.runout_command(rc, tmp_path / "cfg", "2026-12-05", "P")
    assert cmd[1:5] == ["-m", "scripts", "run", "--profile"]
    assert cmd[cmd.index("--path") + 1] == "P_profile"
    assert cmd[cmd.index("--date") + 1] == "20261205"
    assert cmd[cmd.index("--snow-height") + 1] == "set"   # each day its own set's raster


def test_no_runout_table_skips(tmp_path, capsys):
    assert rs.main(["--toml", str(_toml(tmp_path, runout=False))]) == 0
    assert "skipped" in capsys.readouterr().out


def test_an_unreadable_set_is_refused(tmp_path, capsys):
    s = _published_set(tmp_path)
    (s / handoff.READY).unlink()
    t = _toml(tmp_path)
    assert rs.main(["--toml", str(t), "--snapshot-date", "2026-12-05"]) == 2
    err = capsys.readouterr().err
    assert "no readable release set" in err


def test_one_runout_at_a_time_on_the_host(tmp_path, monkeypatch):
    """A second slope's runout waits for the global lock instead of adding
    its 4 sims to the first one's."""
    s = _published_set(tmp_path)
    fake = tmp_path / "fake_runout"
    (fake / ".venv" / "bin").mkdir(parents=True)
    (fake / "config").mkdir()
    for name in ("paths", "physics"):
        (fake / "config" / name).mkdir()
    (fake / "config" / "profiles.yaml").write_text("")
    log = tmp_path / "ran.log"
    py = fake / ".venv" / "bin" / "python"
    py.write_text(f"#!/bin/bash\necho start $(date +%s.%N) >> {log}\nsleep 1\n"
                  f"echo end $(date +%s.%N) >> {log}\n")
    py.chmod(0o755)
    monkeypatch.setattr(rs, "RUNOUT", fake)
    # two slopes: their own work dirs, one lock file for the host
    lock = tmp_path / "work" / "runout" / ".runout.lock"
    tomls = []
    for name in ("A", "B"):
        t = tmp_path / f"{name}.toml"
        t.write_text(textwrap.dedent(f"""\
            [slope]
            name = "{name}"
            [paths]
            project_dir = "{tmp_path}"
            scenarios_dir = "{tmp_path / 'release'}"
            [runout]
            path_profile = "{name}"
            work_dir = "{tmp_path / 'work' / 'runout' / name}"
            lock_file = "{lock}"
            """))
        tomls.append(t)
    rcs = []
    threads = [threading.Thread(target=lambda t=t: rcs.append(
        rs.main(["--toml", str(t), "--snapshot-date", "2026-12-05"]))) for t in tomls]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert rcs == [0, 0]
    ev = [l.split() for l in log.read_text().splitlines()]
    assert [e[0] for e in ev] == ["start", "end", "start", "end"]     # never overlapping


class TestRunScripts:
    """The generated scripts' runout step: a failure keeps the run successful."""

    @pytest.fixture
    def install(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("install_mod", REPO / "install.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m

    def _block(self, sh, step):
        a = sh.index(f'echo ">>> Step {step}: Runout')
        return sh[a:sh.index("total_elapsed=$SECONDS", a)]

    def _run(self, block, rc, no_runout=0):
        script = ("set -euo pipefail\nPROJECT_DIR=/x; TOML=/x/t.toml; SNAPSHOT_ARG=\n"
                  "LOGFILE=/x/run.log\n"
                  "alert_stub() { echo ALERTED \"$@\"; }\nALERT=alert_stub\n"
                  f"NO_SCENARIOS=0; NO_RUNOUT={no_runout}\n"
                  f"python() {{ echo called \"$@\"; return {rc}; }}\n"
                  + block + 'echo "STATUS=$RUNOUT_STATUS"\n')
        return subprocess.run(["bash", "-c", script], capture_output=True, text=True)

    @pytest.mark.parametrize("which", ["survey", "daily"])
    def test_runout_failure_is_a_warning(self, install, which):
        import tomllib
        cfg = tomllib.loads((REPO / "slopes" / "Professor_PTC" / "slope_config.toml").read_text())
        sh, step = ((install._render_run_operational(cfg), 5) if which == "survey"
                    else (install._render_run_daily(cfg), 6))
        block = self._block(sh, step)
        ok = self._run(block, 0)
        assert ok.returncode == 0 and "STATUS=ok" in ok.stdout
        assert "runout_step.py --toml /x/t.toml" in ok.stdout
        bad = self._run(block, 3)
        assert bad.returncode == 0, bad.stderr               # the run is not failed
        assert "STATUS=FAILED (exit 3)" in bad.stdout
        assert "Rerun runout alone" in bad.stdout
        assert "ALERTED --subject runout FAILED (exit 3)" in bad.stdout   # someone is told
        assert "ALERTED" not in ok.stdout
        skip = self._run(block, 0, no_runout=1)
        assert "STATUS=skipped" in skip.stdout and "called" not in skip.stdout


def _forecast_sets(tmp_path, labels):
    """Publish one set per label under release/2026-12-05/<label>/."""
    out = {}
    for lab in labels:
        stage = tmp_path / f".stage_{lab}"
        stage.mkdir()
        (stage / "scour_depth.tif").write_text(lab)
        final = tmp_path / "release" / "2026-12-05" / lab
        final.parent.mkdir(parents=True, exist_ok=True)
        handoff.publish(stage, final, "release", horizon=lab)
        out[lab] = final
    return out


def test_every_forecast_day_is_found_in_order(tmp_path):
    sets = _forecast_sets(tmp_path, ["T_0", "T_2", "T_10", "T_1"])
    from config import ProjectConfig
    cfg = ProjectConfig.from_toml(_toml(tmp_path))
    assert list(rs.release_sets(cfg, "2026-12-05")) == ["T_0", "T_1", "T_2", "T_10"]


def test_a_single_day_set_runs_as_t0(tmp_path):
    s = _published_set(tmp_path)
    from config import ProjectConfig
    cfg = ProjectConfig.from_toml(_toml(tmp_path))
    assert rs.release_sets(cfg, "2026-12-05") == {"T_0": s}


def test_stale_links_of_an_earlier_run_are_cleared(tmp_path):
    sets = _forecast_sets(tmp_path, ["T_0", "T_1"])
    rc = rs.runout_settings(_toml(tmp_path))
    rs.prepare_tree(rc, sets, "2026-12-05")
    rs.prepare_tree(rc, {"T_0": sets["T_0"]}, "2026-12-05")
    day = tmp_path / "work" / "runout" / "P" / "scen" / "P_profile" / "20261205"
    assert sorted(p.name for p in day.iterdir()) == ["T_0"]


def test_a_damaged_day_is_left_out_and_the_rest_runs(tmp_path, capsys):
    sets = _forecast_sets(tmp_path, ["T_0", "T_1"])
    (sets["T_1"] / "scour_depth.tif").write_text("XX")       # edited after publish
    fake = tmp_path / "fake_runout"
    (fake / ".venv" / "bin").mkdir(parents=True)
    (fake / ".venv" / "bin" / "python").write_text("")
    import pytest as _pt
    mp = _pt.MonkeyPatch()
    mp.setattr(rs, "RUNOUT", fake)
    try:
        assert rs.main(["--toml", str(_toml(tmp_path)), "--snapshot-date", "2026-12-05",
                        "--dry-run"]) == 0
    finally:
        mp.undo()
    out, err = capsys.readouterr()
    assert "T_1 left out" in err and "T_0=" in out and "T_1=" not in out

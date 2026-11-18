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


class TestForcedShots:
    """The forced-shot sets (<date>/shots[/T_n]) get a second runout run with
    its own tree and web folder, under the same lock, after the model run
    (Ron, 2026-11-06)."""

    def _sets(self, tmp_path, horizons=None):
        model = _published_set(tmp_path)
        out = {}
        for h in horizons or [None]:
            stage = tmp_path / f".shots{h}"
            stage.mkdir()
            (stage / "scour_depth.tif").write_text("raster")
            final = model / "shots" / h if h else model / "shots"
            final.parent.mkdir(parents=True, exist_ok=True)
            handoff.publish(stage, final, "release", status="ok", n_scenarios=1,
                            trigger_source="explosive_forced")
            out[h or "T_0"] = final
        return model, out

    def test_release_sets_keep_shots_apart(self, tmp_path):
        model, shots = self._sets(tmp_path)
        from config import ProjectConfig
        cfg = ProjectConfig.from_toml(_toml(tmp_path))
        assert rs.release_sets(cfg, "2026-12-05") == {"T_0": model}
        assert rs.release_sets(cfg, "2026-12-05", shots=True) == {"T_0": shots["T_0"]}

    def test_forecast_horizons(self, tmp_path):
        model, shots = self._sets(tmp_path, ["T_0", "T_1"])
        from config import ProjectConfig
        cfg = ProjectConfig.from_toml(_toml(tmp_path))
        assert list(rs.release_sets(cfg, "2026-12-05", shots=True)) == ["T_0", "T_1"]

    def test_own_tree_web_folder_and_label(self, tmp_path):
        rc = rs.runout_settings(_toml(tmp_path))
        rc["path_profile"] = "Professor_PTC"           # a real path profile to relabel
        sr = rs.shot_settings(rc)
        assert sr["work_dir"] == rc["work_dir"] / "shots"
        assert sr["publish_root"] == str(tmp_path / "www" / "shots")
        assert sr["lock_file"] == rc["lock_file"]      # one lock for the host
        model, shots = self._sets(tmp_path)
        cfg_dir = rs.prepare_tree(sr, shots, "2026-12-05")
        cfg_dir = rs.prepare_tree(sr, shots, "2026-12-05")      # rerun
        local = (cfg_dir / "local.yaml").read_text()
        assert f"avachain_root: {sr['work_dir']}" in local
        assert f"publish_root: {tmp_path / 'www' / 'shots'}" in local
        # the runout's own loader reads the relabelled profile
        sys.path.insert(0, str(rs.RUNOUT))
        from scripts import config as runout_cfg
        prof = runout_cfg.load_path_profile("Professor_PTC", sr["work_dir"], cfg_dir)
        assert prof["display"]["project_name"] == "Little Professor — explosive shots"
        assert prof["display"]["caveats"][0] == rs.SHOT_CAVEAT
        assert prof["terrain"]["dem"].startswith("/home/avachain/data/static")
        assert "shots_20261205" in " ".join(rs.runout_command(sr, cfg_dir, "2026-12-05", "P"))
        # the model run's tree still links the repo's paths
        mdir = rs.prepare_tree(rc, {"T_0": model}, "2026-12-05")
        assert (mdir / "paths").resolve() == (rs.RUNOUT / "config" / "paths").resolve()

    def test_both_runs_one_lock_model_first(self, tmp_path, monkeypatch):
        self._sets(tmp_path)
        fake = tmp_path / "fake_runout"
        (fake / ".venv" / "bin").mkdir(parents=True)
        for name in ("paths", "physics"):
            (fake / "config" / name).mkdir(parents=True)
        (fake / "config" / "paths" / "P_profile.yaml").write_text("slug: P_profile\n")
        (fake / "config" / "profiles.yaml").write_text("")
        log = tmp_path / "ran.log"
        py = fake / ".venv" / "bin" / "python"
        py.write_text(f'#!/bin/bash\necho "$@" >> {log}\n')
        py.chmod(0o755)
        monkeypatch.setattr(rs, "RUNOUT", fake)
        assert rs.main(["--toml", str(_toml(tmp_path)), "--snapshot-date", "2026-12-05"]) == 0
        runs = log.read_text().splitlines()
        assert len(runs) == 2
        assert "--run-name P_20261205" in runs[0] and "--run-name P_shots_20261205" in runs[1]

    def test_a_failed_shot_run_fails_the_step(self, tmp_path, monkeypatch):
        self._sets(tmp_path)
        fake = tmp_path / "fake_runout"
        (fake / ".venv" / "bin").mkdir(parents=True)
        for name in ("paths", "physics"):
            (fake / "config" / name).mkdir(parents=True)
        (fake / "config" / "paths" / "P_profile.yaml").write_text("slug: P_profile\n")
        (fake / "config" / "profiles.yaml").write_text("")
        py = fake / ".venv" / "bin" / "python"
        py.write_text('#!/bin/bash\n[[ "$*" == *shots* ]] && exit 3\nexit 0\n')
        py.chmod(0o755)
        monkeypatch.setattr(rs, "RUNOUT", fake)
        assert rs.main(["--toml", str(_toml(tmp_path)), "--snapshot-date", "2026-12-05"]) == 3


# --- identical scenarios run once (Ron, 2026-11-18) ---------------------------

def _scenario(stage, sid, *, depth="d1", poly=((0, 0), (10, 0), (10, 10)), w=0.1,
              sf=1.0, mu=0.155, density=None, total=None):
    d = stage / "scenarios" / sid
    d.mkdir(parents=True)
    (d / "depth.tif").write_text(depth)
    (d / "density.json").write_text(json.dumps(density or {"mean_kgm3": 300.0, "std_kgm3": 20.0}))
    params = {"mu": mu, "xi": 1500.0, "scour_depth_m": 0.0, "size_factor": sf,
              "trigger_cluster": 7, "scenario_probability": w}
    if total is not None:
        params["ensemble_weight_total"] = total
    (d / "params.json").write_text(json.dumps(params))
    (d / "release.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": [{
        "type": "Feature",
        "properties": {"scenario_id": sid, "size_factor": sf, "weight": w, "trigger_cluster": 7},
        "geometry": {"type": "Polygon", "coordinates": [[list(p) for p in poly + poly[:1]]]}}]}))


def _set(tmp_path, specs, name="2026-12-05"):
    """Publish a set from {sid: kwargs for _scenario}."""
    stage = tmp_path / f".stage_{name}"
    (stage / "scenarios").mkdir(parents=True)
    (stage / "scour_depth.tif").write_text("scour")
    for sid, kw in specs.items():
        _scenario(stage, sid, **kw)
    final = tmp_path / "release" / name
    final.parent.mkdir(parents=True, exist_ok=True)
    handoff.publish(stage, final, "release", snapshot_date="2026-12-05",
                    status="ok", n_scenarios=len(specs))
    return final


def _params(view, sid):
    return json.loads((view / "scenarios" / sid / "params.json").read_text())


# five size factors of one trigger and depth, as the distance caps off makes them
SIZE_FACTORS = (0.7, 0.85, 1.0, 1.15, 1.3)


class TestMergeIdentical:
    def test_size_factor_variants_run_once_with_summed_probability(self, tmp_path):
        specs = {}
        weights = {}
        n = 0
        for depth, w in (("p10", 0.0039), ("p50", 0.0116), ("p90", 0.0039)):
            for sf in SIZE_FACTORS:
                n += 1
                sid = f"scenario_{n:03d}"
                specs[sid] = dict(depth=depth, w=w * sf, sf=sf)
                weights.setdefault(depth, []).append((sid, w * sf))
        s = _set(tmp_path, specs)
        view = tmp_path / "view"
        rs.merge_view(s, view)
        runs = sorted(p.name for p in (view / "scenarios").iterdir())
        assert runs == ["scenario_001", "scenario_006", "scenario_011"]
        for rep, (depth, members) in zip(runs, weights.items()):
            p = _params(view, rep)
            assert p["merged_scenarios"] == [m for m, _ in members]
            assert p["scenario_probability"] == pytest.approx(sum(w for _, w in members), rel=1e-15)
        # the probability mass is all still there
        assert sum(_params(view, r)["scenario_probability"] for r in runs) == pytest.approx(
            sum(kw["w"] for kw in specs.values()), rel=1e-15)
        # inputs are the published files themselves, and the set is untouched
        rep = view / "scenarios" / "scenario_006"
        assert (rep / "depth.tif").resolve() == s / "scenarios" / "scenario_006" / "depth.tif"
        assert (view / "scour_depth.tif").resolve() == s / "scour_depth.tif"
        assert handoff.verify(s) == []
        assert json.loads((view / "merged_scenarios.json").read_text())["scenario_011"] == [
            f"scenario_{i:03d}" for i in range(11, 16)]

    @pytest.mark.parametrize("change", [
        dict(depth="other raster"),
        dict(mu=0.2),
        dict(poly=((0, 0), (10, 0), (10, 11))),
        dict(density={"mean_kgm3": 300.0, "std_kgm3": 21.0}),
    ])
    def test_any_input_difference_keeps_them_apart(self, tmp_path, change):
        s = _set(tmp_path, {"scenario_001": dict(sf=0.7), "scenario_002": dict(sf=1.0, **change)})
        view = tmp_path / "view"
        rs.merge_view(s, view)
        assert view.is_symlink() and view.resolve() == s.resolve()

    def test_an_unknown_params_field_keeps_them_apart(self, tmp_path):
        stage_specs = {"scenario_001": dict(sf=0.7), "scenario_002": dict(sf=1.0)}
        s = _set(tmp_path, stage_specs)
        groups = rs.merge_groups(s)
        assert [len(g) for g in groups] == [2]
        p = s / "scenarios" / "scenario_002" / "params.json"
        d = json.loads(p.read_text())
        d["new_runout_knob"] = 1
        p.write_text(json.dumps(d))
        assert [len(g) for g in rs.merge_groups(s)] == [1, 1]

    def test_absolute_shot_weights_keep_their_total(self, tmp_path):
        s = _set(tmp_path, {f"scenario_{i:03d}": dict(sf=sf, w=1 / 3, total=1.0)
                            for i, sf in enumerate(SIZE_FACTORS[:3], 1)}
                 | {"scenario_004": dict(depth="other", w=1 / 3, total=1.0)})
        view = tmp_path / "view"
        rs.merge_view(s, view)
        p = _params(view, "scenario_001")
        assert p["scenario_probability"] == pytest.approx(1.0)
        assert p["ensemble_weight_total"] == 1.0
        assert _params(view, "scenario_004")["ensemble_weight_total"] == 1.0

    def test_rerun_replaces_a_merged_view(self, tmp_path):
        (tmp_path / "b").mkdir()
        s = _set(tmp_path, {"scenario_001": dict(sf=0.7), "scenario_002": dict(sf=1.0)})
        rc = rs.runout_settings(_toml(tmp_path))
        day = tmp_path / "work" / "runout" / "P" / "scen" / "P_profile" / "20261205"
        rs.prepare_tree(rc, {"T_0": s}, "2026-12-05")
        assert (day / "T_0").is_dir() and not (day / "T_0").is_symlink()
        rs.prepare_tree(rc, {"T_0": s}, "2026-12-05")
        assert sorted(p.name for p in (day / "T_0" / "scenarios").iterdir()) == ["scenario_001"]
        other = _published_set(tmp_path / "b")                # no duplicates
        rs.prepare_tree(rc, {"T_0": other}, "2026-12-05")
        assert (day / "T_0").is_symlink()

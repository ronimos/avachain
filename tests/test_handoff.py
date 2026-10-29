"""
tests/test_handoff.py — the release → runout handoff set.

Runout starts on READY, so these pin what READY promises: the set is
complete, carries a manifest that matches its files, a rerun never exposes a
mix of old and new, a failed build leaves the published set alone, and
step_scenarios publishes through this path.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src" / "avachain"))

import handoff                       # noqa: E402


def _build(stage: Path, files: dict) -> None:
    for rel, text in files.items():
        (stage / rel).parent.mkdir(parents=True, exist_ok=True)
        (stage / rel).write_text(text)


def test_publish_moves_the_set_into_place_with_manifest_and_ready(tmp_path):
    final = tmp_path / "handoff" / "P" / "release" / "2026-01-18"
    stage = handoff.staging_dir(final)
    assert stage.parent == final.parent and stage.name.startswith(".")
    _build(stage, {"summary.csv": "a,b\n", "scenarios/S1/depth.asc": "1 2\n"})
    toml = tmp_path / "slope.toml"
    toml.write_text("x = 1\n")

    handoff.publish(stage, final, "release", toml=toml, snapshot_date="2026-01-18")

    assert not stage.exists() and handoff.is_ready(final)
    m = json.loads((final / "manifest.json").read_text())
    assert m["producer"] == "release" and m["snapshot_date"] == "2026-01-18"
    assert len(m["commit"]) == 40 and isinstance(m["commit_dirty"], bool)
    assert m["toml_sha256"] == hashlib.sha256(b"x = 1\n").hexdigest()
    listed = {f["path"]: f for f in m["files"]}
    assert set(listed) == {"summary.csv", "scenarios/S1/depth.asc"}
    assert listed["summary.csv"]["sha256"] == hashlib.sha256(b"a,b\n").hexdigest()


def test_republish_replaces_the_whole_set(tmp_path):
    final = tmp_path / "2026-01-18"
    s1 = handoff.staging_dir(final)
    _build(s1, {"old_only.csv": "x", "summary.csv": "old"})
    handoff.publish(s1, final, "release")
    s2 = handoff.staging_dir(final)
    _build(s2, {"summary.csv": "new"})
    handoff.publish(s2, final, "release")

    assert (final / "summary.csv").read_text() == "new"
    assert not (final / "old_only.csv").exists()          # no mix of runs
    assert handoff.is_ready(final)
    assert [p.name for p in tmp_path.iterdir()] == ["2026-01-18"]   # no leftovers


def test_republish_keeps_nested_horizon_sets(tmp_path):
    day = tmp_path / "2026-01-18"
    s = handoff.staging_dir(day / "T_1")
    _build(s, {"summary.csv": "t1"})
    handoff.publish(s, day / "T_1", "release")
    s = handoff.staging_dir(day)
    _build(s, {"summary.csv": "analysis"})
    handoff.publish(s, day, "release")
    assert (day / "T_1" / "summary.csv").read_text() == "t1"
    assert handoff.is_ready(day / "T_1") and handoff.is_ready(day)


class TestStepScenarios:
    """step_scenarios publishes through handoff; a failed build publishes nothing."""

    def _cfg_args(self, tmp_path):
        cfg = SimpleNamespace(scenarios_dir=tmp_path / "release", toml_path=None,
                              slope_name="Professor_PTC")
        args = SimpleNamespace(snapshot_date="2026-01-18", forecast_horizon=None)
        return cfg, args

    def test_success_publishes(self, tmp_path, monkeypatch):
        import analysis_pipeline as ap

        def fake_build(cfg, args, out_dir):
            (out_dir / "summary.csv").write_text("s\n")
            (out_dir / "metadata.json").write_text(json.dumps({"n_scenarios": 75}))
        monkeypatch.setattr(ap, "_build_scenarios", fake_build)
        cfg, args = self._cfg_args(tmp_path)
        final = ap.step_scenarios(cfg, args)
        assert final == tmp_path / "release" / "2026-01-18" and handoff.is_ready(final)
        m = json.loads((final / "manifest.json").read_text())
        assert m["status"] == "ok" and m["n_scenarios"] == 75 and m["path"] == "Professor_PTC"

    def test_failure_keeps_the_published_set_and_leaves_no_staging(self, tmp_path, monkeypatch):
        import analysis_pipeline as ap
        cfg, args = self._cfg_args(tmp_path)
        final = tmp_path / "release" / "2026-01-18"
        s = handoff.staging_dir(final)
        _build(s, {"summary.csv": "published"})
        handoff.publish(s, final, "release")

        def broken(cfg, args, out_dir):
            (out_dir / "summary.csv").write_text("half")
            raise RuntimeError("boom")
        monkeypatch.setattr(ap, "_build_scenarios", broken)
        with pytest.raises(RuntimeError):
            ap.step_scenarios(cfg, args)
        assert (final / "summary.csv").read_text() == "published" and handoff.is_ready(final)
        assert [p.name for p in final.parent.iterdir()] == ["2026-01-18"]


def test_republish_keeps_foreign_subdirectories(tmp_path):
    """The dev date dirs hold experiment runs (gate_off_20261026/, energy_check/,
    ...) that a rerun must not delete."""
    day = tmp_path / "2026-01-18"
    s = handoff.staging_dir(day)
    _build(s, {"summary.csv": "old", "scenarios/S1/a.asc": "old"})
    handoff.publish(s, day, "release")
    _build(day, {"gate_off_20261026/summary.csv": "experiment"})
    s = handoff.staging_dir(day)
    _build(s, {"summary.csv": "new", "scenarios/S2/a.asc": "new"})
    handoff.publish(s, day, "release")
    assert (day / "gate_off_20261026" / "summary.csv").read_text() == "experiment"
    assert not (day / "scenarios" / "S1").exists()     # the set's own dir is replaced
    m = json.loads((day / "manifest.json").read_text())
    assert "gate_off_20261026/summary.csv" not in {f["path"] for f in m["files"]}

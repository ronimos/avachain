"""
tests/test_forcing_pipeline.py — Tests for forcing_pipeline.step_reinit argument routing.

Run with:  pytest tests/test_forcing_pipeline.py -v

Key risk: step_reinit reads cfg._reinit_args and calls _reinit_single / run_reinit
with a set of keyword arguments.  If an attribute name changes (e.g. 'event_date'
renamed to 'event_dt'), getattr(..., default) silently passes the default instead
of erroring.  These tests verify the call signature under both single-event and
multi-event paths.
"""

from __future__ import annotations

import json
import types
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "avachain"))

from config import ProjectConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_args(**overrides) -> types.SimpleNamespace:
    defaults = dict(
        all_events=False,
        event_date="2026-01-18",
        event_time="12:00:00",
        date_before="2026-01-14",
        date_after="2026-01-20",
        snapshot_date="2026-01-18",
        release_geojson=None,
        kernel_size_reinit=7,
        threshold_sigma_reinit=1.2,
        reinit_dry_run=False,
        reinit_no_backup=False,
    )
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


def _make_cfg(tmp_path: Path, args: types.SimpleNamespace) -> ProjectConfig:
    cfg = ProjectConfig(project_dir=tmp_path)
    cfg._reinit_args = args
    # boundaries_dir depends on boundary_kml; point it to tmp_path
    cfg.boundary_kml = tmp_path / "dummy.kml"
    return cfg


# ---------------------------------------------------------------------------
# Single-event path
# ---------------------------------------------------------------------------

class TestStepReinitSingleEvent:
    def test_run_reinit_called_once(self, tmp_path):
        args = _make_args()
        cfg = _make_cfg(tmp_path, args)
        with patch("forcing_pipeline._reinit_single") as mock_single:
            from forcing_pipeline import step_reinit
            step_reinit(cfg)
        mock_single.assert_called_once()

    def test_run_reinit_receives_event_date(self, tmp_path):
        args = _make_args(event_date="2026-01-18")
        cfg = _make_cfg(tmp_path, args)
        with patch("forcing_pipeline._reinit_single") as mock_single:
            from forcing_pipeline import step_reinit
            step_reinit(cfg)
        _, kwargs = mock_single.call_args
        assert kwargs["event_date"] == "2026-01-18"

    def test_run_reinit_receives_date_before_and_after(self, tmp_path):
        args = _make_args(date_before="2026-01-14", date_after="2026-01-20")
        cfg = _make_cfg(tmp_path, args)
        with patch("forcing_pipeline._reinit_single") as mock_single:
            from forcing_pipeline import step_reinit
            step_reinit(cfg)
        _, kwargs = mock_single.call_args
        assert kwargs["date_before"] == "2026-01-14"
        assert kwargs["date_after"] == "2026-01-20"

    def test_run_reinit_receives_snapshot_date(self, tmp_path):
        args = _make_args(snapshot_date="2026-01-18")
        cfg = _make_cfg(tmp_path, args)
        with patch("forcing_pipeline._reinit_single") as mock_single:
            from forcing_pipeline import step_reinit
            step_reinit(cfg)
        _, kwargs = mock_single.call_args
        assert kwargs["snapshot_date"] == "2026-01-18"

    def test_release_geojson_none_by_default(self, tmp_path):
        args = _make_args(release_geojson=None)
        cfg = _make_cfg(tmp_path, args)
        with patch("forcing_pipeline._reinit_single") as mock_single:
            from forcing_pipeline import step_reinit
            step_reinit(cfg)
        _, kwargs = mock_single.call_args
        assert kwargs["release_geojson"] is None

    def test_release_geojson_passed_when_set(self, tmp_path):
        gj = str(tmp_path / "release.geojson")
        args = _make_args(release_geojson=gj)
        cfg = _make_cfg(tmp_path, args)
        with patch("forcing_pipeline._reinit_single") as mock_single:
            from forcing_pipeline import step_reinit
            step_reinit(cfg)
        _, kwargs = mock_single.call_args
        assert kwargs["release_geojson"] == gj


# ---------------------------------------------------------------------------
# _reinit_single — argument forwarding to run_reinit
# ---------------------------------------------------------------------------

class TestReinitSingleArgForwarding:
    # _reinit_single does `from reinitialize_snowpack import run_reinit` locally,
    # so the patch target is the source module, not forcing_pipeline.
    def test_run_reinit_called_with_event_date(self, tmp_path):
        cfg = _make_cfg(tmp_path, _make_args())
        args = _make_args()
        with patch("reinitialize_snowpack.run_reinit") as mock_rr:
            from forcing_pipeline import _reinit_single
            _reinit_single(cfg, args,
                           event_date="2026-01-18",
                           event_time="14:00:00",
                           date_before="2026-01-14",
                           date_after="2026-01-20",
                           snapshot_date="2026-01-18",
                           release_geojson=None)
        mock_rr.assert_called_once()
        _, kwargs = mock_rr.call_args
        assert kwargs["event_date"] == "2026-01-18"
        assert kwargs["event_time"] == "14:00:00"
        assert kwargs["date_before"] == "2026-01-14"
        assert kwargs["date_after"] == "2026-01-20"

    def test_kernel_size_default_forwarded(self, tmp_path):
        cfg = _make_cfg(tmp_path, _make_args())
        args = types.SimpleNamespace()  # no kernel_size_reinit → getattr fallback
        with patch("reinitialize_snowpack.run_reinit") as mock_rr:
            from forcing_pipeline import _reinit_single
            _reinit_single(cfg, args,
                           event_date="2026-01-18", event_time="12:00:00",
                           date_before="2026-01-14", date_after="2026-01-20",
                           snapshot_date="2026-01-18", release_geojson=None)
        _, kwargs = mock_rr.call_args
        assert kwargs["kernel_size"] == 7

    def test_threshold_sigma_default_forwarded(self, tmp_path):
        cfg = _make_cfg(tmp_path, _make_args())
        args = types.SimpleNamespace()
        with patch("reinitialize_snowpack.run_reinit") as mock_rr:
            from forcing_pipeline import _reinit_single
            _reinit_single(cfg, args,
                           event_date="2026-01-18", event_time="12:00:00",
                           date_before="2026-01-14", date_after="2026-01-20",
                           snapshot_date="2026-01-18", release_geojson=None)
        _, kwargs = mock_rr.call_args
        assert kwargs["threshold_sigma"] == pytest.approx(1.2)

    def test_dry_run_default_is_false(self, tmp_path):
        cfg = _make_cfg(tmp_path, _make_args())
        args = types.SimpleNamespace()
        with patch("reinitialize_snowpack.run_reinit") as mock_rr:
            from forcing_pipeline import _reinit_single
            _reinit_single(cfg, args,
                           event_date="2026-01-18", event_time="12:00:00",
                           date_before="2026-01-14", date_after="2026-01-20",
                           snapshot_date="2026-01-18", release_geojson=None)
        _, kwargs = mock_rr.call_args
        assert kwargs["dry_run"] is False

    def test_dry_run_true_forwarded(self, tmp_path):
        cfg = _make_cfg(tmp_path, _make_args())
        args = types.SimpleNamespace(reinit_dry_run=True,
                                     reinit_no_backup=False,
                                     kernel_size_reinit=7,
                                     threshold_sigma_reinit=1.2)
        with patch("reinitialize_snowpack.run_reinit") as mock_rr:
            from forcing_pipeline import _reinit_single
            _reinit_single(cfg, args,
                           event_date="2026-01-18", event_time="12:00:00",
                           date_before="2026-01-14", date_after="2026-01-20",
                           snapshot_date="2026-01-18", release_geojson=None)
        _, kwargs = mock_rr.call_args
        assert kwargs["dry_run"] is True


# ---------------------------------------------------------------------------
# Multi-event path
# ---------------------------------------------------------------------------

class TestStepReinitMultiEvent:
    def _setup_cfg(self, tmp_path: Path) -> ProjectConfig:
        """Build a cfg whose analysis_dir and boundaries_dir both sit inside tmp_path."""
        args = _make_args(all_events=True)
        cfg = _make_cfg(tmp_path, args)
        # Point output_dir so analysis_dir == tmp_path/analysis
        cfg.output_dir = tmp_path
        # boundaries_dir == boundary_kml.parent == tmp_path (set by _make_cfg)
        return cfg

    def _write_events_for(self, cfg: ProjectConfig, events: dict):
        cfg.analysis_dir.mkdir(parents=True, exist_ok=True)
        (cfg.analysis_dir / "avalanche_events.json").write_text(json.dumps(events))

    def test_multi_event_calls_reinit_per_event(self, tmp_path):
        cfg = self._setup_cfg(tmp_path)
        events = {
            "2026-01-14__2026-01-18": {
                "reinit_needed": True,
                "event_timestamp": "2026-01-18T14:00:00",
            },
            "2026-01-18__2026-01-22": {
                "reinit_needed": True,
                "event_timestamp": "2026-01-22T10:00:00",
            },
        }
        self._write_events_for(cfg, events)

        with patch("forcing_pipeline._reinit_single") as mock_single:
            from forcing_pipeline import step_reinit
            step_reinit(cfg)

        assert mock_single.call_count == 2

    def test_multi_event_no_reinit_needed_skips_all(self, tmp_path):
        cfg = self._setup_cfg(tmp_path)
        events = {
            "2026-01-14__2026-01-18": {
                "reinit_needed": False,
                "event_timestamp": "2026-01-18T14:00:00",
            },
        }
        self._write_events_for(cfg, events)

        with patch("forcing_pipeline._reinit_single") as mock_single:
            from forcing_pipeline import step_reinit
            step_reinit(cfg)

        mock_single.assert_not_called()

    def test_multi_event_missing_json_returns_without_crash(self, tmp_path):
        args = _make_args(all_events=True)
        cfg = _make_cfg(tmp_path, args)
        # Do NOT create avalanche_events.json

        with patch("forcing_pipeline._reinit_single") as mock_single:
            from forcing_pipeline import step_reinit
            step_reinit(cfg)   # should not raise

        mock_single.assert_not_called()

    def test_geojson_path_passed_when_file_exists(self, tmp_path):
        cfg = self._setup_cfg(tmp_path)
        events = {
            "2026-01-14__2026-01-18": {
                "reinit_needed": True,
                "event_timestamp": "2026-01-18T14:00:00",
            },
        }
        self._write_events_for(cfg, events)
        # boundaries_dir == tmp_path; create the matching geojson there
        (tmp_path / "avalanche_release_area_20260118.geojson").write_text(
            '{"type":"FeatureCollection","features":[]}'
        )

        with patch("forcing_pipeline._reinit_single") as mock_single:
            from forcing_pipeline import step_reinit
            step_reinit(cfg)

        _, kwargs = mock_single.call_args
        assert kwargs["release_geojson"] is not None
        assert "20260118" in kwargs["release_geojson"]

    def test_geojson_none_when_file_missing(self, tmp_path):
        cfg = self._setup_cfg(tmp_path)
        events = {
            "2026-01-14__2026-01-18": {
                "reinit_needed": True,
                "event_timestamp": "2026-01-18T14:00:00",
            },
        }
        self._write_events_for(cfg, events)
        # Do NOT create the geojson file

        with patch("forcing_pipeline._reinit_single") as mock_single:
            from forcing_pipeline import step_reinit
            step_reinit(cfg)

        _, kwargs = mock_single.call_args
        assert kwargs["release_geojson"] is None

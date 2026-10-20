"""Where avalanche-event artifacts live, and that a replay cannot share them.

The avalanche step reads its known-events file as manual labels AND appends
its own auto-detections to it. It also exported an auto mask per event into
data/boundaries/, next to the observed polygons (the Jan 18 crown). A replay
slope sharing those paths would have written its detections into the live
slope's inputs, and read the live season's later auto events back as hand
labels from its own past.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src" / "avachain"))

from config import ProjectConfig                                # noqa: E402


def _cfg(tmp_path, **paths):
    toml = tmp_path / "slope_config.toml"
    lines = ["[slope]", 'name = "s"', "[paths]", f'project_dir = "{tmp_path}"',
             'boundary_kml = "data/boundaries/b.kml"', 'output_dir = "outputs/s"']
    lines += [f'{k} = "{v}"' for k, v in paths.items()]
    toml.write_text("\n".join(lines) + "\n")
    return ProjectConfig.from_toml(toml)


def test_known_events_default_is_unchanged_for_live(tmp_path):
    cfg = _cfg(tmp_path)
    assert cfg.avalanche_events_path == tmp_path / "data/boundaries/avalanche_events.json"


def test_known_events_path_is_configurable_per_slope(tmp_path):
    cfg = _cfg(tmp_path, avalanche_events="outputs/s/known_avalanche_events.json")
    assert cfg.avalanche_events_path == tmp_path / "outputs/s/known_avalanche_events.json"


def test_auto_masks_are_slope_namespaced(tmp_path):
    cfg = _cfg(tmp_path)
    assert cfg.event_masks_dir == tmp_path / "outputs/s/event_masks"
    assert cfg.event_masks_dir != cfg.boundaries_dir


def test_observed_polygon_wins_over_auto_mask(tmp_path):
    cfg = _cfg(tmp_path)
    t = pd.Timestamp("2026-01-18T17:00")
    auto = cfg.event_masks_dir / "avalanche_release_area_20260118.geojson"
    auto.parent.mkdir(parents=True)
    auto.write_text("{}")
    assert cfg.event_mask_for(t) == auto
    obs = cfg.boundaries_dir / "avalanche_release_area_20260118.geojson"
    obs.parent.mkdir(parents=True)
    obs.write_text("{}")
    assert cfg.event_mask_for(t) == obs


def test_no_mask_is_none(tmp_path):
    assert _cfg(tmp_path).event_mask_for(pd.Timestamp("2026-02-20")) is None


def test_the_replay_slope_does_not_share_the_live_known_events(tmp_path):
    live = ProjectConfig.from_toml(REPO / "slopes/little_prof/slope_config.toml")
    replay = ProjectConfig.from_toml(REPO / "slopes/little_prof_replay/slope_config.toml")
    assert replay.avalanche_events_path != live.avalanche_events_path
    assert replay.event_masks_dir != live.event_masks_dir

"""The January replay must not see the observed Jan 18 crown polygon.

data/boundaries/avalanche_release_area_20260118.geojson is the validation
answer for the n=1 Jan 18 event. Two readers take it from boundaries_dir:
event_mask_for (reinit prefers an observed polygon over its own detection)
and release_geojsons_for_date (analyze). boundaries_dir is the parent of
boundary_kml, so the replay slope points its KMLs at its own copy of the
boundaries without that file (Ron, 2026-10-22). Nothing downstream would
notice a leak: reinit would simply scour the right cells.
"""
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src" / "avachain"))

from config import ProjectConfig  # noqa: E402

JAN = REPO / "slopes" / "little_prof_replay_jan" / "slope_config.toml"
LIVE = REPO / "slopes" / "little_prof" / "slope_config.toml"


def _cfg(p):
    return ProjectConfig.from_toml(p)


def test_jan_replay_has_its_own_boundaries_dir():
    jan, live = _cfg(JAN), _cfg(LIVE)
    assert jan.boundaries_dir.resolve() != live.boundaries_dir.resolve()
    assert jan.boundary_kml.exists() and jan.start_zone_kml.exists()


def test_jan_replay_holds_no_observed_release_polygon():
    jan = _cfg(JAN)
    assert sorted(jan.boundaries_dir.glob("avalanche_release_area_*.geojson")) == []


def test_reinit_and_analyze_cannot_read_the_jan18_crown():
    jan = _cfg(JAN)
    assert jan.event_mask_for(date(2026, 1, 18)) is None or \
        jan.event_mask_for(date(2026, 1, 18)).parent == jan.event_masks_dir
    assert jan.release_geojsons_for_date("2026-03-31") == []


def test_the_live_slope_still_sees_it():
    """Guards the guard: if the crown moved, the tests above prove nothing."""
    live = _cfg(LIVE)
    assert live.event_mask_for(date(2026, 1, 18)) == \
        live.boundaries_dir / "avalanche_release_area_20260118.geojson"


def test_jan_replay_state_is_namespaced():
    jan, live = _cfg(JAN), _cfg(LIVE)
    for attr in ("output_dir", "slope_dir", "avalanche_events_path", "zarr_path"):
        assert getattr(jan, attr) != getattr(live, attr), attr
        assert "little_prof_replay_jan" in str(getattr(jan, attr)), attr

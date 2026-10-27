"""Regression guard: the default (legacy) Jan 18 release polygons must not move.

Reference generated 2026-10-25 on the frozen fixture (tests/jan18_fixture.py)
against the 6 934 m2 observed crown. The areas equal the 2026-10-23 rebuild's
scenarios for the same triggers (code at 2f79a8e), so the energy-mode port left
the legacy path unchanged. Triggers are the top three of that rebuild.

Re-pinned 2026-10-27 for STAUCHWALL_DEG 28 -> 29 (Ron). At 28 the polygons
still reproduce the old references exactly (1772 11309 m2, 1998 10954,
4300 10458), so the angle is the only change. All three are walled in by a
neighbouring cluster whose mean slope is 28-29 deg, the per-cluster downslope
arrest; they are not the current operational triggers.
"""
import pytest

import jan18_fixture as fx
import energy_path as ep
import release_geometry as rg

# trigger_cluster -> (release_area_m2, iou)
REFERENCE = {
    1772: (615.0, 0.05109608184675175),
    1998: (1370.0, 0.02304515046757521),
    4300: (1570.0, 0.09157456180556534),
}


@pytest.fixture(scope='module')
def legacy():
    assert rg.ARREST_MODE == 'legacy' and rg.MASK_MODE == 'gate'
    return {t: fx.polygon_for(t) for t in REFERENCE}


def test_observed_crown_is_the_6934_m2_mapping():
    assert fx.load()['observed'].area == pytest.approx(6933.6, abs=1.0)


@pytest.mark.parametrize('trig', list(REFERENCE))
def test_legacy_area_and_iou(legacy, trig):
    area, iou = REFERENCE[trig]
    assert legacy[trig].area == pytest.approx(area, abs=1.0)
    assert fx.iou(legacy[trig]) == pytest.approx(iou, abs=1e-3)


def test_energy_switches_do_not_touch_legacy(legacy, monkeypatch):
    """ENERGY_DELTA / SIGMA_T_COLUMN / ENERGY_R feed only the energy mode."""
    monkeypatch.setattr(ep, 'ENERGY_DELTA', 1.0)
    monkeypatch.setattr(ep, 'SIGMA_T_COLUMN', 'sigma_t_wl')
    monkeypatch.setattr(ep, 'ENERGY_R', 0.7)
    assert fx.polygon_for(1772).equals(legacy[1772])

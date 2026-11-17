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

Re-pinned 2026-10-27 again for the forcing rebuild (21 false avalanche
corrections removed, reinit at 18:00) and the slab tensile gate on by
default. The fixture CSVs now come from that rebuild and carry
sigma_t_mean__jamieson1990, so the gate can fire here; the old ones lacked
the column and the gate was a silent no-op. Trigger 37 is the rebuild's
second operational trigger. GATE_OFF pins the gate's effect: with it off,
1998 and 37 grow back to 1519 and 3850 m2.

Re-pinned 2026-11-10 for USE_SLAB_CRACK_LINES on by default (Ron): 37 goes
3657 -> 3498 m2 (IoU 0.2670 -> 0.2664) because a crack line from a tensile-gate
failure cuts off part of its region; the other three start no line that cuts
anything. CRACK_OFF pins the old value with the flag off.

Re-pinned 2026-11-13 for USE_SEVERED_SLIDE on by default (Ron): 37 goes
3498 -> 3542 m2 (IoU 0.2664 -> 0.2707) because part of the piece its crack line
cut off lies above atan(0.62) and slides. SLIDE_OFF pins the 11-10 value. The
fixture CSVs are precomputed, so the 11-13 facet-rule and slab-floor defaults
do not reach this test.

Re-pinned 2026-11-16 for SLAB_CRACK_MODE 'opening' by default (Ron): 37 goes
3542 -> 3652 m2 (IoU 0.2707 -> 0.2671). Opening flank lines need diverging fall
lines on both sides, so 37 runs 33 line clusters instead of 62 and severs 2
instead of 19; the result is within 5 m2 of lines off (3657). The other three
are unchanged. SHEAR pins the 11-13 value. These are module defaults; the
calibrated slopes pin 'shear' in [calibration] until the next recalibration.

Re-pinned 2026-11-16 again for USE_OPENING_TOUGHNESS on by default (Ron): 37
goes 3652 -> 3637 m2 (IoU 0.2671 -> 0.2676): 5 opening flank cells cannot be
broken by the opening load (K < K_Ic) and are removed; it is an extra gate, so
the area can only fall. OPENING_NO_TOUGHNESS pins the value before.
"""
import pytest

import jan18_fixture as fx
import energy_path as ep
import release_geometry as rg
import model_params as mp

# trigger_cluster -> (release_area_m2, iou)
REFERENCE = {
    1772: (612.0, 0.05067835270873417),
    1998: (1385.0, 0.029587436796836643),
    4300: (1676.0, 0.10518130657424013),
    37: (3637.0, 0.26762218182322545),
}

# Same trigger with USE_OPENING_TOUGHNESS off (the reference earlier on 11-16).
OPENING_NO_TOUGHNESS = {
    37: (3652.0, 0.2671416509692288),
}

# Same trigger with SLAB_CRACK_MODE 'shear' (the reference 2026-11-13 to 11-16).
SHEAR = {
    37: (3542.0, 0.2707061518182006),
}

# Same trigger with USE_SEVERED_SLIDE off (the reference 2026-11-10 to 11-13).
SLIDE_OFF = {
    37: (3498.0, 0.26644425400377847),
}

# Same triggers with USE_SLAB_CRACK_LINES off (the reference before 2026-11-10).
CRACK_OFF = {
    37: (3657.0, 0.26698185726428897),
}

# Same triggers with USE_SLAB_TENSILE_GATE off.
GATE_OFF = {
    1998: (1519.0, 0.029480750878604123),
    37: (3850.0, 0.26095662636612055),
}


@pytest.fixture(scope='module')
def legacy():
    assert mp.ARREST_MODE == 'legacy' and mp.MASK_MODE == 'gate'
    return {t: fx.polygon_for(t) for t in REFERENCE}


def test_observed_crown_is_the_6934_m2_mapping():
    assert fx.load()['observed'].area == pytest.approx(6933.6, abs=1.0)


@pytest.mark.parametrize('trig', list(REFERENCE))
def test_legacy_area_and_iou(legacy, trig):
    area, iou = REFERENCE[trig]
    assert legacy[trig].area == pytest.approx(area, abs=1.0)
    assert fx.iou(legacy[trig]) == pytest.approx(iou, abs=1e-3)


def test_tensile_gate_is_on_and_fires(legacy, monkeypatch):
    assert mp.USE_SLAB_TENSILE_GATE
    assert 'sigma_t_mean__jamieson1990' in fx.load()['snap_features'].columns
    monkeypatch.setattr(mp, 'USE_SLAB_TENSILE_GATE', False)
    for trig, (area, iou) in GATE_OFF.items():
        poly = fx.polygon_for(trig)
        assert poly.area == pytest.approx(area, abs=1.0)
        assert fx.iou(poly) == pytest.approx(iou, abs=1e-3)
        assert poly.area > legacy[trig].area


def test_energy_switches_do_not_touch_legacy(legacy, monkeypatch):
    """ENERGY_DELTA / SIGMA_T_COLUMN / ENERGY_R feed only the energy mode."""
    monkeypatch.setattr(mp, 'ENERGY_DELTA', 1.0)
    monkeypatch.setattr(mp, 'SIGMA_T_COLUMN', 'sigma_t_wl')
    monkeypatch.setattr(mp, 'ENERGY_R', 0.7)
    assert fx.polygon_for(1772).equals(legacy[1772])


def test_slab_crack_lines_on_and_cut(legacy, monkeypatch):
    assert mp.USE_SLAB_CRACK_LINES
    monkeypatch.setattr(mp, 'USE_SLAB_CRACK_LINES', False)
    for trig, (area, iou) in CRACK_OFF.items():
        poly = fx.polygon_for(trig)
        assert poly.area == pytest.approx(area, abs=1.0)
        assert fx.iou(poly) == pytest.approx(iou, abs=1e-3)
        assert poly.area > legacy[trig].area


def test_severed_slide_is_on_and_moves_37(legacy, monkeypatch):
    """Measured under shear lines (11-13), where 37 severs 19 clusters."""
    assert mp.USE_SEVERED_SLIDE
    monkeypatch.setattr(mp, 'SLAB_CRACK_MODE', 'shear')
    with_slide = {t: fx.polygon_for(t) for t in SLIDE_OFF}
    monkeypatch.setattr(mp, 'USE_SEVERED_SLIDE', False)
    for trig, (area, iou) in SLIDE_OFF.items():
        poly = fx.polygon_for(trig)
        assert poly.area == pytest.approx(area, abs=1.0)
        assert fx.iou(poly) == pytest.approx(iou, abs=1e-3)
        assert poly.area < with_slide[trig].area


def test_crack_mode_is_opening_and_shear_restores_11_13(legacy, monkeypatch):
    assert mp.SLAB_CRACK_MODE == 'opening'
    monkeypatch.setattr(mp, 'SLAB_CRACK_MODE', 'shear')
    for trig, (area, iou) in SHEAR.items():
        poly = fx.polygon_for(trig)
        assert poly.area == pytest.approx(area, abs=1.0)
        assert fx.iou(poly) == pytest.approx(iou, abs=1e-3)
    for trig in set(REFERENCE) - set(SHEAR):
        assert fx.polygon_for(trig).equals(legacy[trig])


def test_opening_toughness_is_on_and_only_removes(legacy, monkeypatch):
    assert mp.USE_OPENING_TOUGHNESS
    monkeypatch.setattr(mp, 'USE_OPENING_TOUGHNESS', False)
    for trig, (area, iou) in OPENING_NO_TOUGHNESS.items():
        poly = fx.polygon_for(trig)
        assert poly.area == pytest.approx(area, abs=1.0)
        assert fx.iou(poly) == pytest.approx(iou, abs=1e-3)
        assert poly.area >= legacy[trig].area


# USE_DOWNSLOPE_SLIDE on (off by default), opening: UNIFIED_SLIDE off / on.
# 37 is the only trigger whose lines sever anything; with UNIFIED_SLIDE the
# severed pieces are decided by the push routing (line clusters as sinks)
# instead of the mean-slope rule.
DOWNSLOPE_37 = {False: (3442.0, 0.2721), True: (3317.0, 0.2764)}


@pytest.mark.parametrize('unified', [False, True])
def test_downslope_and_unified_slide_on_37(monkeypatch, unified):
    monkeypatch.setattr(mp, 'USE_DOWNSLOPE_SLIDE', True)
    monkeypatch.setattr(mp, 'UNIFIED_SLIDE', unified)
    poly = fx.polygon_for(37)
    area, iou = DOWNSLOPE_37[unified]
    assert poly.area == pytest.approx(area, abs=1.0)
    assert fx.iou(poly) == pytest.approx(iou, abs=1e-3)


def test_unified_slide_needs_the_downslope_test(legacy, monkeypatch):
    """UNIFIED_SLIDE alone (slide test off) changes nothing."""
    monkeypatch.setattr(mp, 'UNIFIED_SLIDE', True)
    assert fx.polygon_for(37).equals(legacy[37])

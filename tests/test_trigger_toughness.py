"""Trigger fracture toughness gate and the BFS distance-caps switch (Ron,
2026-11-10). Jan 18 cannot see the toughness gate (its five triggers pass with
or without it), so the behaviour is pinned here."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "avachain"))
import arrest_indices as ai      # noqa: E402
import release_geometry as rg    # noqa: E402
import calibration               # noqa: E402
import snowpack_analysis as sa   # noqa: E402


def test_toughness_limited_strength_formula():
    K, h, st = 1650.0, 1.1, 20e3
    want = K / (1.122 * np.sqrt(np.pi * 0.5 * h))
    assert ai.toughness_limited_strength(st, K, h) == pytest.approx(want)
    # never raises sigma_t
    assert ai.toughness_limited_strength(100.0, K, h) == 100.0
    # missing K or h leaves sigma_t alone
    assert ai.toughness_limited_strength(st, np.nan, h) == st
    assert ai.toughness_limited_strength(st, K, np.nan) == st


def _frames(K):
    cid = 7
    snap = pd.DataFrame({rg.SLAB_TENSILE_SIGMA_COL: [8000.0], 'slab_density': [300.0],
                         'slab_thickness': [1.0], rg.TOUGHNESS_K_COL: [K]}, index=[cid])
    mel = pd.DataFrame({'rc_wl': [0.5]}, index=[cid])
    return cid, mel, snap


def test_trigger_support_uses_toughness(monkeypatch):
    # sigma_t 8 kPa holds (S_t > 1); K 800 Pa m^.5 caps it at ~0.57 kPa (S_t < 1)
    cid, mel, snap = _frames(800.0)
    monkeypatch.setattr(rg, "USE_TRIGGER_TOUGHNESS", False)
    plain = rg.cluster_tensile_support(cid, mel, snap)
    monkeypatch.setattr(rg, "USE_TRIGGER_TOUGHNESS", True)
    tough = rg.cluster_tensile_support(cid, mel, snap)
    assert plain > 1.0 > tough
    s_eff = 800.0 / (1.122 * np.sqrt(np.pi * 0.5))
    assert tough == pytest.approx(plain * s_eff / 8000.0)


def test_trigger_support_without_k_column_is_plain(monkeypatch):
    cid, mel, snap = _frames(np.nan)
    monkeypatch.setattr(rg, "USE_TRIGGER_TOUGHNESS", True)
    a = rg.cluster_tensile_support(cid, mel, snap)
    monkeypatch.setattr(rg, "USE_TRIGGER_TOUGHNESS", False)
    assert rg.cluster_tensile_support(cid, mel, snap) == pytest.approx(a)


def test_defaults_match_the_standalone_contract():
    assert rg.USE_TRIGGER_TOUGHNESS is True
    assert rg.USE_DISTANCE_CAPS is True
    assert rg.TOUGHNESS_K_COL == 'K_Ic_ligament'
    assert (ai.TOUGHNESS_EDGE_F, ai.TOUGHNESS_A_FRAC) == (1.122, 0.5)


class _Cfg:
    def __init__(self, cal):
        self.calibration = cal


def test_calibration_sets_facet_a_and_caps(monkeypatch):
    monkeypatch.setattr(sa, "FACET_STRENGTH_FACTOR", sa.FACET_STRENGTH_FACTOR)
    monkeypatch.setattr(rg, "USE_DISTANCE_CAPS", True)
    got = calibration.apply(_Cfg({"version": "t", "facet_a": 0.75,
                                  "distance_caps": False}), quiet=True)
    assert got == {"facet_a": 0.75, "distance_caps": False}
    assert sa.FACET_STRENGTH_FACTOR == 0.75
    assert rg.USE_DISTANCE_CAPS is False


@pytest.mark.parametrize("bad", [{"facet_a": 0.0}, {"facet_a": 1.5},
                                 {"distance_caps": "no"}])
def test_calibration_rejects_bad_values(bad):
    with pytest.raises(ValueError):
        calibration.apply(_Cfg(bad), quiet=True)


def test_calibration_sets_tau_g_floor(monkeypatch):
    # The floor was a function-local literal until 2026-11-10; it is now a
    # module parameter the BFS reads at call time, so a slope can set it.
    monkeypatch.setattr(rg, "TAU_G_ABS_FLOOR", rg.TAU_G_ABS_FLOOR)
    assert rg.TAU_G_ABS_FLOOR == 350.0
    got = calibration.apply(_Cfg({"tau_g_floor": 420.0}), quiet=True)
    assert got == {"tau_g_floor": 420.0} and rg.TAU_G_ABS_FLOOR == 420.0
    with pytest.raises(ValueError):
        calibration.apply(_Cfg({"tau_g_floor": -1.0}), quiet=True)


def test_no_bfs_literals_left_in_propagate_release():
    import inspect
    src = inspect.getsource(rg.propagate_release)
    for lit in ("350.0", "TAU_G_ABS_FLOOR =", "MIN_PROPAGATION_SLAB  =",
                "LAMBDA_DROP_FACTOR     =", "20.0)", "15.0)", "else 50.0"):
        assert lit not in src, lit

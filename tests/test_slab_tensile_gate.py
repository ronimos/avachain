"""Slab tensile-failure gate (release_geometry.USE_SLAB_TENSILE_GATE).

A candidate is arrested when its slab would break in tension before its
weak-layer crack reaches critical length: S_t = sigma_t / (3 rho g r_c^2 / h)
< 1 (Schweizer et al. 2014, ISSW, Eq. 1). Default OFF; when off the BFS must
be untouched, and when on it must stop at a band of S_t < 1 clusters even
though every other gate passes there.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from affine import Affine

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "avachain"))

import arrest_indices as ai                                     # noqa: E402
import release_geometry as rg                                   # noqa: E402
from release_geometry import (                                  # noqa: E402
    cluster_pixel_index, compute_slope_aspect, propagate_release,
)

T = Affine(1.0, 0.0, 400000.0, 0.0, -1.0, 4000000.0)
BLOCK, N = 4, 81
SIG = rg.SLAB_TENSILE_SIGMA_COL


class TestFormula:
    def test_reproduces_the_papers_worked_value(self):
        """Schweizer et al. (2014): rho 100, h 0.3 m, r_c 0.2 m -> ~400 Pa."""
        assert ai.slab_tensile_stress_cantilever(100.0, 0.3, 0.2) == \
            pytest.approx(3 * 100 * 9.81 * 0.04 / 0.3)
        assert 380 < ai.slab_tensile_stress_cantilever(100.0, 0.3, 0.2) < 420

    def test_support_scales(self):
        base = ai.slab_tensile_support(4000.0, 330.0, 1.0, 0.4)
        assert ai.slab_tensile_support(4000.0, 330.0, 1.0, 0.8) == pytest.approx(base / 4)
        assert ai.slab_tensile_support(4000.0, 330.0, 2.0, 0.4) == pytest.approx(base * 2)
        assert ai.slab_tensile_support(2000.0, 330.0, 1.0, 0.4) == pytest.approx(base / 2)


def test_off_by_default():
    import re
    src = Path(rg.__file__).read_text()
    assert re.search(r'^USE_SLAB_TENSILE_GATE\s*=\s*False\b', src, re.M)


@pytest.fixture
def scene():
    """Uniform 35 deg south-facing plane, 4x4 m clusters, every gate passing,
    S_t ~ 9 everywhere except a full-width cross-slope band 5-6 clusters below
    the trigger, where r_c is raised so S_t ~ 0.6."""
    r, c = np.mgrid[0:N, 0:N]
    cluster_map = ((r // BLOCK) * (N // BLOCK + 1) + (c // BLOCK) + 1).astype(float)
    dem = -np.tan(np.radians(35.0)) * (-r.astype(float))   # falls toward +row
    slope_grid, _ = compute_slope_aspect(dem, 1.0)
    px = cluster_pixel_index(cluster_map)
    cids = sorted(px)
    trigger = int(cluster_map[N // 2, N // 2])
    t_row = px[trigger][0].mean()

    band = [cid for cid in cids if 18 <= px[cid][0].mean() - t_row <= 26]
    rc = pd.Series(0.3, index=cids)
    rc[band] = 1.2
    meloche_df = pd.DataFrame({
        'Pi1_elastic': 50.0, 'Lambda': 2.0, 'Lambda_cross': 2.0,
        'tau_g': 800.0, 'L_t': 30.0, 'A_ca_brittle': 60.0, 'rc_wl': rc,
    }, index=cids)
    snap = pd.DataFrame({
        'slab_thickness': 1.0, 'slab_density': 330.0,
        'wl_shear_strength': 2400.0, 'sigma_t': 4000.0, SIG: 2800.0,
    }, index=cids)
    below = [cid for cid in cids if px[cid][0].mean() - t_row > 26]
    return dict(trigger=trigger, cluster_map=cluster_map, dem=dem,
                slope_grid=slope_grid, meloche_df=meloche_df, snap=snap,
                band=set(band), below=set(below), pixel_index=px)


def _run(scene, snap=None):
    reasons = {}
    def cb(_ring, _prop, arrested):
        for a in arrested:
            reasons[a['cid']] = a['reason']
    _, reached = propagate_release(
        trigger_cluster_id=scene['trigger'], meloche_df=scene['meloche_df'],
        cluster_map=scene['cluster_map'], dem=scene['dem'],
        slope_grid=scene['slope_grid'], transform=T, A_ca=200.0,
        snap_features=scene['snap'] if snap is None else snap,
        max_clusters=20000, wave_callback=cb, pixel_index=scene['pixel_index'])
    return set(reached), reasons


def test_scene_values(scene):
    """The fixture means what the docstring says."""
    in_band = ai.slab_tensile_support(2800.0, 330.0, 1.0, 1.2)
    outside = ai.slab_tensile_support(2800.0, 330.0, 1.0, 0.3)
    assert in_band < 1.0 < outside


def test_off_leaves_the_band_reachable(scene, monkeypatch):
    monkeypatch.setattr(rg, 'USE_SLAB_TENSILE_GATE', False)
    reached, reasons = _run(scene)
    assert scene['band'] <= reached
    assert scene['below'] & reached
    assert 'slab_tensile_failure' not in reasons.values()


def test_on_arrests_at_the_band(scene, monkeypatch):
    monkeypatch.setattr(rg, 'USE_SLAB_TENSILE_GATE', True)
    reached, reasons = _run(scene)
    assert not (scene['band'] & reached)
    assert not (scene['below'] & reached)       # the band spans the slope
    hit = {c for c, why in reasons.items() if why == 'slab_tensile_failure'}
    assert hit and hit <= scene['band']


def test_missing_input_is_not_gated(scene, monkeypatch):
    monkeypatch.setattr(rg, 'USE_SLAB_TENSILE_GATE', True)
    reached, _ = _run(scene, snap=scene['snap'].drop(columns=[SIG]))
    assert scene['band'] <= reached


# --- the trigger itself -----------------------------------------------------

def _polygon(scene, trigger, state):
    return rg.make_release_polygon_2d(
        trigger_cluster_id=trigger, A_ca=200.0, meloche_df=scene['meloche_df'],
        cluster_map=scene['cluster_map'], dem=scene['dem'], transform=T,
        snap_features=scene['snap'], pixel_index=scene['pixel_index'],
        max_clusters=20000, state_out=state)


def test_cluster_support_matches_the_gate(scene):
    cid = next(iter(scene['band']))
    assert rg.cluster_tensile_support(cid, scene['meloche_df'], scene['snap']) == \
        pytest.approx(ai.slab_tensile_support(2800.0, 330.0, 1.0, 1.2))
    assert np.isnan(rg.cluster_tensile_support(
        cid, scene['meloche_df'], scene['snap'].drop(columns=[SIG])))


def test_trigger_in_failing_slab_releases_nothing(scene, monkeypatch):
    """No polygon and, in particular, no rectangle fallback."""
    trigger = next(iter(scene['band']))
    monkeypatch.setattr(rg, 'USE_SLAB_TENSILE_GATE', True)
    state = {}
    assert _polygon(scene, trigger, state) is None
    assert state.get('trigger_tensile_failure') is True


def test_trigger_gate_off_or_sound_slab_still_releases(scene, monkeypatch):
    monkeypatch.setattr(rg, 'USE_SLAB_TENSILE_GATE', False)
    assert _polygon(scene, next(iter(scene['band'])), {}) is not None
    monkeypatch.setattr(rg, 'USE_SLAB_TENSILE_GATE', True)
    state = {}
    assert _polygon(scene, scene['trigger'], state) is not None
    assert 'trigger_tensile_failure' not in state

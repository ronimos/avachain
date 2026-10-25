"""Synthetic-slope tests for the path-energy BFS (rg.ARREST_MODE = 'energy').

Scene: a south-facing slope tiled with 3 x 3 m clusters (the Jan 18 cluster
size), fall line along +row. Each test sets per-cluster snowpack as a function
of the cluster's position, so the expected arrest point can be stated exactly.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'avachain'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import pytest
from affine import Affine

import energy_path as ep
import release_geometry as rg
from arrest_indices import (
    R_FIT,
    slab_modulus,
    slab_tensile_strength,
    weak_layer_stiffness,
)
from energy_path import g_slab, tau_net
from release_geometry import (
    cluster_pixel_index,
    compute_slope_aspect,
    make_release_polygon_2d,
    propagate_release,
)

T = Affine(1.0, 0.0, 400000.0, 0.0, -1.0, 4000000.0)
BLOCK = 3
RHO, H, E, SIG_T, D_WL = 300.0, 1.0, 5e6, 6e3, 0.04
K_WL = weak_layer_stiffness(ep.G_WL, D_WL)
RG_SLAB = R_FIT * g_slab(SIG_T, H, E)            # 1.57 J/m2, saturated capability


def tau_p_for(G_c, delta=ep.ENERGY_DELTA):
    """Weak-layer strength giving fracture energy G_c (tau_r = 0)."""
    return np.sqrt(2.0 * K_WL * G_c / (1.0 + delta))


def scene(nr, nc, trig_px, snow, slope_deg=35.0):
    """Build a scene. snow(y_m, x_m) -> dict overriding the uniform slab, with
    (y_m, x_m) the cluster centroid offset from the trigger, +y downslope, +x east.
    slope_deg is a number or a function of the pixel row (m)."""
    r, c = np.mgrid[0:nr, 0:nc]
    cluster_map = ((r // BLOCK) * (nc // BLOCK) + (c // BLOCK) + 1).astype(float)
    rows = np.arange(nr, dtype=float)
    psi = np.array([slope_deg(y) if callable(slope_deg) else slope_deg for y in rows])
    z = -np.concatenate([[0.0], np.cumsum(np.tan(np.radians(0.5 * (psi[1:] + psi[:-1]))))])
    dem = np.repeat(z[:, None], nc, axis=1)
    slope_grid, aspect_grid = compute_slope_aspect(dem, 1.0)
    pixel_index = cluster_pixel_index(cluster_map)
    trigger = int(cluster_map[trig_px])
    tr, tc = (v.mean() for v in pixel_index[trigger])
    recs = {}
    for cid, (pr, pc) in pixel_index.items():
        y, x = pr.mean() - tr, pc.mean() - tc
        recs[cid] = {'slab_density': RHO, 'slab_thickness': H, 'E_eff': E,
                     'sigma_t_mean': SIG_T, 'wl_thickness': D_WL,
                     'wl_shear_strength': tau_p_for(0.05), **snow(y, x),
                     'y': y, 'x': x}
    feats = pd.DataFrame.from_dict(recs, orient='index')
    return dict(trigger=trigger, cluster_map=cluster_map, dem=dem,
                slope_grid=slope_grid, aspect_grid=aspect_grid,
                pixel_index=pixel_index, feats=feats)


@pytest.fixture(autouse=True)
def energy_mode(monkeypatch):
    monkeypatch.setattr(rg, 'ARREST_MODE', 'energy')


def run(s, stauchwall_deg=rg.STAUCHWALL_DEG, state=None):
    poly, reached = propagate_release(
        trigger_cluster_id=s['trigger'], meloche_df=pd.DataFrame(),
        cluster_map=s['cluster_map'], dem=s['dem'], slope_grid=s['slope_grid'],
        transform=T, snap_features=s['feats'], pixel_index=s['pixel_index'],
        aspect_grid=s['aspect_grid'], stauchwall_deg=stauchwall_deg,
        max_clusters=100000, state_out=state)
    return poly, reached


def test_uniform_slope_reaches_every_cell():
    s = scene(60, 60, (31, 31), lambda y, x: {})
    state = {}
    _, reached = run(s, state=state)
    assert reached == set(s['pixel_index'])
    assert set(state['rejected']) == set()


def test_tension_is_fall_line_integral_not_path_length():
    """N is the detached fall-line extent: tau_net * |y| on a uniform slope,
    however the outward path zig-zags (alternating diagonals must not pump N)."""
    s = scene(90, 30, (1, 16), lambda y, x: {})
    state = {}
    run(s, state=state)
    tn = tau_net(RHO, H, 35.0)
    for cid, c in state['cells'].items():
        y = s['feats'].loc[cid, 'y']
        assert c['N'] <= min(tn * abs(y), SIG_T * H) * 1.01 + 1e-9


def test_strength_step_arrests_exactly_at_step():
    step_y = -15.0                                   # upslope of the trigger
    s = scene(90, 30, (61, 16),
              lambda y, x: {'wl_shear_strength': tau_p_for(2.0)} if y < step_y else {})
    state = {}
    _, reached = run(s, state=state)
    ys = s['feats'].loc[list(reached), 'y']
    assert ys.min() >= step_y
    assert ys.min() < step_y + BLOCK                 # last cluster below the step is reached
    beyond = s['feats'].index[s['feats']['y'] < step_y]
    assert set(state['rejected']) & set(beyond)
    assert {state['rejected'][c] for c in set(state['rejected']) & set(beyond)} == {'energy_deficit'}


def test_flattening_slope_arrests_between_phi_and_drain_limit():
    """35 deg to 20 deg over 40 m. Tension saturates on the steep part, then drains
    once psi < phi; arrest lies past psi = phi and at the analytic drain point."""
    y0, y1 = 45.0, 85.0

    def psi(y):
        return 35.0 if y < y0 else 20.0 if y > y1 else 35.0 - 15.0 * (y - y0) / (y1 - y0)

    trig_row = 16
    G_c_strong = 0.5 * RG_SLAB
    s = scene(180, 30, (trig_row, 16),
              lambda y, x: {} if y + trig_row < y0
              else {'wl_shear_strength': tau_p_for(G_c_strong)},
              slope_deg=psi)
    _, reached = run(s, stauchwall_deg=0.0)
    y_last = s['feats'].loc[list(reached), 'y'].max() + trig_row

    y_phi = y0 + (y1 - y0) * (35.0 - ep.PHI_DEG) / 15.0
    # Drain from N_max to N* = N_max sqrt(G_c / R G_slab) on the continuous profile.
    N_star = SIG_T * H * np.sqrt(G_c_strong / RG_SLAB)
    yy = np.arange(y_phi, 180.0, 0.01)
    drain = np.cumsum(-tau_net(RHO, H, np.array([psi(v) for v in yy]))) * 0.01
    y_star = yy[np.argmax(drain >= SIG_T * H - N_star)]
    assert y_last >= y_phi
    assert abs(y_last - y_star) <= 2 * BLOCK


def test_thin_slab_band_arrests_at_band():
    band = (15.0, 21.0)       # two clusters: edge clusters' k-NN links span 6 m
    rho_b, h_b = 250.0, 0.4
    E_b, st_b = float(slab_modulus(rho_b)), float(slab_tensile_strength(rho_b))
    assert R_FIT * g_slab(st_b, h_b, E_b) < 0.3

    def snow(y, x):
        if band[0] <= y < band[1]:
            return {'slab_density': rho_b, 'slab_thickness': h_b,
                    'E_eff': E_b, 'sigma_t_mean': st_b,
                    'wl_shear_strength': tau_p_for(0.3)}
        if abs(y) <= 9.0:
            return {'wl_shear_strength': tau_p_for(0.01)}   # weak nucleation zone
        return {'wl_shear_strength': tau_p_for(0.3)}

    s = scene(90, 30, (31, 16), snow)
    state = {}
    _, reached = run(s, state=state)
    ys = s['feats'].loc[list(reached), 'y']
    assert ys.max() < band[0]
    assert ys.max() >= band[0] - BLOCK
    assert ys.min() <= -27.0                         # upslope runs on to the edge


def test_early_arrest_gives_trigger_only_polygon():
    s = scene(30, 30, (16, 16), lambda y, x: {'wl_shear_strength': tau_p_for(2.0)})
    poly, reached = run(s)
    assert reached == {s['trigger']}
    assert poly is not None and poly.area == pytest.approx(BLOCK * BLOCK)


def test_early_arrest_survives_make_release_polygon_without_fallback():
    s = scene(30, 30, (16, 16), lambda y, x: {'wl_shear_strength': tau_p_for(2.0)})
    poly = make_release_polygon_2d(
        trigger_cluster_id=s['trigger'], A_ca=50.0, meloche_df=pd.DataFrame(),
        cluster_map=s['cluster_map'], dem=s['dem'], transform=T,
        snap_features=s['feats'], pixel_index=s['pixel_index'])
    assert poly is not None and poly.area == pytest.approx(BLOCK * BLOCK)


def test_no_drive_below_friction_angle():
    s = scene(30, 30, (16, 16), lambda y, x: {}, slope_deg=25.0)
    state = {}
    _, reached = run(s, stauchwall_deg=0.0, state=state)
    assert reached == {s['trigger']}
    assert set(state['rejected'].values()) == {'no_drive'}


def test_lateral_growth_stops_at_missing_weak_layer_stripe():
    """Step 3: a cross-slope stripe with no weak layer stops the east flank only.
    Two clusters wide: k-NN links at the domain border span 6 m and jump one."""
    s = scene(30, 60, (16, 31),
              lambda y, x: {'wl_shear_strength': np.nan} if 6.0 <= x < 12.0 else {})
    state = {}
    _, reached = run(s, state=state)
    xs = s['feats'].loc[list(reached), 'x']
    assert xs.max() < 6.0
    assert xs.min() == s['feats']['x'].min()         # west flank reaches the edge
    stripe = s['feats'].index[(s['feats']['x'] >= 6.0) & (s['feats']['x'] < 12.0)]
    assert {state['rejected'][c] for c in stripe if c in state['rejected']} == {'no_weak_layer'}


def test_laterally_reached_cell_starts_at_zero_tension():
    s = scene(30, 60, (16, 31), lambda y, x: {})
    state = {}
    run(s, state=state)
    cross = [c for c in state['cells'].values() if c['step'] == 'cross']
    assert cross and all(c['N'] == 0.0 for c in cross)


def _polygon(s, mask):
    return make_release_polygon_2d(
        trigger_cluster_id=s['trigger'], A_ca=50.0, meloche_df=pd.DataFrame(),
        cluster_map=s['cluster_map'], dem=s['dem'], transform=T,
        start_zone_mask=mask, snap_features=s['feats'],
        pixel_index=s['pixel_index'], max_clusters=100000)


def test_mask_gate_versus_diagnostic(monkeypatch):
    """Step 4: 'gate' stops at the mask and clips; 'diagnostic' ignores it."""
    from analysis_pipeline import frac_outside_mask
    s = scene(30, 60, (16, 31), lambda y, x: {})
    mask = np.zeros(s['cluster_map'].shape, bool)
    mask[:, :45] = True                              # east quarter outside
    gated = _polygon(s, mask)
    assert frac_outside_mask(gated, mask, T) == 0.0
    monkeypatch.setattr(rg, 'MASK_MODE', 'diagnostic')
    diag = _polygon(s, mask)
    assert diag.area == pytest.approx(30 * 60)
    assert frac_outside_mask(diag, mask, T) == pytest.approx(0.25)
    assert diag.area > gated.area


def test_unknown_mask_mode_raises(monkeypatch):
    monkeypatch.setattr(rg, 'MASK_MODE', 'nonsense')
    s = scene(30, 30, (16, 16), lambda y, x: {})
    with pytest.raises(ValueError):
        _polygon(s, None)


def test_unknown_mode_raises(monkeypatch):
    monkeypatch.setattr(rg, 'ARREST_MODE', 'nonsense')
    s = scene(30, 30, (16, 16), lambda y, x: {})
    with pytest.raises(ValueError):
        run(s)

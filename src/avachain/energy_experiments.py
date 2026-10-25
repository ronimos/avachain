"""Path-energy brief, Steps 6-9: fixed-trigger runs, trigger-sensitivity map,
parameter sensitivity, and the comparison metrics they share.

Inputs: snowpack features for the whole cluster map from slopes/little_prof_energy
(analyze --feature-buffer-m), used by 'energy' and the baselines; legacy in 'gate'
mode reads outputs/little_prof, the operational start-zone features, so it
reproduces the operational polygons. Nothing here ranks triggers.

    python src/avachain/energy_experiments.py fixed   [--n-sampled 20]
    python src/avachain/energy_experiments.py envelope [--n 300]
    python src/avachain/energy_experiments.py params
    python src/avachain/energy_experiments.py check

Writes outputs/little_prof_energy/experiments/. Hypothesis tooling; nothing
here is validated.
"""
from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import energy_path as ep
import release_geometry as rg
from config import ProjectConfig

SNAPSHOT = '2026-01-18'
MODES = ('legacy', 'energy', 'slope_only', 'stability_connected')
MASK_MODES = ('gate', 'diagnostic')
SHIPPED_TRIGGERS = (2859, 5656, 6191)      # ~/avalanche-release-areas default triggers
SEED = 20260118
N_CROSS = 5                                # crown offset sample positions
SEGMENT_BAND_M = 10.0                      # predicted boundary cells within this of a segment

# One-at-a-time sensitivity (Step 8). Defaults are the energy_path constants as
# committed in 1402165, before any of these runs.
PARAM_VARIANTS = {
    'R':             [('ENERGY_R', 0.3), ('ENERGY_R', None), ('ENERGY_R', 0.7)],
    'delta':         [('ENERGY_DELTA', 0.0), ('ENERGY_DELTA', 1.0)],
    'sigma_t':       [('SIGMA_T_COLUMN', 'sigma_t_mean'), ('SIGMA_T_COLUMN', 'sigma_t_wl'),
                      ('SIGMA_T_COLUMN', 'sigma_t_mean__jamieson1990')],
    'E_relation':    [('ENERGY_E_RELATION', None), ('ENERGY_E_RELATION', 'project_fit')],
    'G_wl':          [('G_WL', 0.1e6), ('G_WL', 0.2e6), ('G_WL', 0.4e6)],
    'KWL_thickness': [('KWL_THICKNESS', 'sequence'), ('KWL_THICKNESS', 'weakest_element'),
                      ('KWL_THICKNESS', 0.04)],
    'use_residual':  [('ENERGY_USE_RESIDUAL', False), ('ENERGY_USE_RESIDUAL', True)],
}


@contextmanager
def setting(module, **kw):
    """Temporarily set module attributes."""
    old = {k: getattr(module, k) for k in kw}
    for k, v in kw.items():
        setattr(module, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(module, k, v)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

class Scene:
    """Everything a run needs, loaded once."""

    def __init__(self, project_dir: Path):
        import rasterio
        from analysis_pipeline import _load_observed_release_polygon, kml_to_mask
        self.cfg_live = ProjectConfig.from_toml(project_dir / 'slopes/little_prof/slope_config.toml')
        self.cfg = ProjectConfig.from_toml(project_dir / 'slopes/little_prof_energy/slope_config.toml')
        with rasterio.open(self.cfg_live.resampled_dir / 'dem_1m.tif') as src:
            self.dem = src.read(1).astype(np.float32)
            self.dem[self.dem == src.nodata] = np.nan
            self.transform, self.profile = src.transform, src.profile.copy()
        self.cluster_map = np.load(self.cfg_live.analysis_dir / 'cluster_map.npy')
        self.sz_mask = kml_to_mask(self.cfg_live.start_zone_kml, self.dem.shape, self.transform)
        self.observed = _load_observed_release_polygon(self.cfg_live, SNAPSHOT)
        self.pixel_index = rg.cluster_pixel_index(self.cluster_map)
        self.slope_grid, self.aspect_grid = rg.compute_slope_aspect(self.dem)

        def _csv(cfg, stem):
            df = pd.read_csv(cfg.analysis_dir / f'{stem}_{SNAPSHOT}.csv', index_col=0)
            return df[~df.index.duplicated()]
        self.feats_live = _csv(self.cfg_live, 'all_start_zone_features')
        self.mel_live = _csv(self.cfg_live, 'meloche_features_all')
        self.feats_dom = _csv(self.cfg, 'all_start_zone_features')
        self.mel_dom = _csv(self.cfg, 'meloche_features_all')
        self.psi = {c: float(self.slope_grid[px].mean()) for c, px in self.pixel_index.items()}
        self.out = self.cfg.output_dir / 'experiments'
        self.out.mkdir(parents=True, exist_ok=True)
        self.frame = SlopeFrame.observed(self)

    def inputs(self, mode, mask_mode):
        """(snap_features, meloche_df) for a mode: legacy-gate = operational CSVs."""
        if mode == 'legacy' and mask_mode == 'gate':
            return self.feats_live, self.mel_live
        return self.feats_dom, self.mel_dom

    def a_ca(self, cid, meloche):
        v = float(meloche['A_ca_brittle'].get(cid, np.nan)) if 'A_ca_brittle' in meloche else np.nan
        return 50.0 if not np.isfinite(v) else min(v, 300.0)

    def crown_clusters(self, frac=0.5):
        from rasterio.features import geometry_mask
        inside = geometry_mask([self.observed.__geo_interface__], self.cluster_map.shape,
                               self.transform, invert=True)
        ids, n = np.unique(self.cluster_map[inside & (self.cluster_map > 0)], return_counts=True)
        tot = {c: len(px[0]) for c, px in self.pixel_index.items()}
        return {int(c) for c, k in zip(ids, n) if k / tot[int(c)] >= frac}


# ---------------------------------------------------------------------------
# Geometry: a fall-line frame, crown / flank offsets, boundary segments
# ---------------------------------------------------------------------------

class SlopeFrame:
    """u = downslope distance along a fall line, v = cross-slope to the right looking downslope."""

    def __init__(self, aspect_deg, origin_xy):
        a = np.radians(aspect_deg)
        self.f = np.array([np.sin(a), np.cos(a)])          # downslope, map frame (E, N)
        self.r = np.array([self.f[1], -self.f[0]])          # right of downslope
        self.o = np.asarray(origin_xy, float)
        self.aspect = float(aspect_deg)

    @classmethod
    def at_cluster(cls, scene, cid):
        rows, cols = scene.pixel_index[cid]
        asp = float(scene.aspect_grid[int(rows.mean()), int(cols.mean())])
        return cls(asp, rg.pixel_to_utm(int(round(rows.mean())), int(round(cols.mean())),
                                        scene.transform))

    @classmethod
    def observed(cls, scene):
        """Frame of the observed crown: circular-mean aspect of its pixels."""
        from rasterio.features import geometry_mask
        inside = geometry_mask([scene.observed.__geo_interface__], scene.dem.shape,
                               scene.transform, invert=True)
        a = np.radians(scene.aspect_grid[inside])
        a = a[np.isfinite(a)]
        asp = np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean())) % 360
        c = scene.observed.centroid
        return cls(asp, (c.x, c.y))

    def uv(self, xy):
        d = np.asarray(xy, float) - self.o
        return d @ self.f, d @ self.r

    def to_uv(self, geom):
        from shapely import affinity
        g = affinity.translate(geom, -self.o[0], -self.o[1])
        # rows of the 2x2 map: u = f.x, v = r.x
        return affinity.affine_transform(g, [self.f[0], self.f[1], self.r[0], self.r[1], 0, 0])


def crown_flank_offsets(pred, obs, frame, n=N_CROSS):
    """Signed crown offsets (pred - obs, + = predicted crown further downslope) at n
    cross-slope positions spanning the observed polygon, and flank offsets per side
    (+ = predicted flank outside the observed one)."""
    from shapely.geometry import LineString
    P, O = frame.to_uv(pred), frame.to_uv(obs)
    v0, v1 = O.bounds[1], O.bounds[3]
    crown = []
    for q in np.linspace(0.1, 0.9, n):
        v = v0 + q * (v1 - v0)
        line = LineString([(-1e4, v), (1e4, v)])
        po, pp = O.intersection(line), P.intersection(line)
        crown.append(np.nan if pp.is_empty or po.is_empty else pp.bounds[0] - po.bounds[0])
    return {'crown_offsets_m': crown,
            'crown_offset_median_m': float(np.nanmedian(crown)) if np.isfinite(crown).any() else np.nan,
            'flank_left_m': O.bounds[1] - P.bounds[1],
            'flank_right_m': P.bounds[3] - O.bounds[3]}


def boundary_segments(obs, frame, step=1.0):
    """Observed boundary points (map frame) labelled crown / stauchwall / flank_left / flank_right
    by their outward normal in the frame."""
    ring = obs.exterior if obs.exterior.is_ccw else obs.exterior.reverse()   # ccw: outward = right
    pts = [ring.interpolate(d).coords[0] for d in np.arange(0, ring.length, step)]
    pts = np.array(pts)
    t = np.roll(pts, -1, axis=0) - np.roll(pts, 1, axis=0)
    nrm = np.c_[t[:, 1], -t[:, 0]]                      # right of travel = outward for ccw
    nu, nv = nrm @ frame.f, nrm @ frame.r
    lab = np.where(np.abs(nu) >= np.abs(nv),
                   np.where(nu < 0, 'crown', 'stauchwall'),
                   np.where(nv > 0, 'flank_right', 'flank_left'))
    return pts, lab


def mechanism_table(rejected: dict, scene, seg_pts, seg_lab, band=SEGMENT_BAND_M):
    """Per observed segment: share of rejected (boundary) clusters within `band` m, by reason."""
    from scipy.spatial import cKDTree
    if not rejected:
        return {}
    tree = cKDTree(seg_pts)
    cids = list(rejected)
    xy = np.array([rg.pixel_to_utm(int(round(scene.pixel_index[c][0].mean())),
                                   int(round(scene.pixel_index[c][1].mean())), scene.transform)
                   for c in cids])
    d, i = tree.query(xy)
    out = {}
    for seg in ('crown', 'flank_left', 'flank_right', 'stauchwall'):
        sel = [rejected[c] for c, dd, ii in zip(cids, d, i) if dd <= band and seg_lab[ii] == seg]
        out[seg] = pd.Series(sel, dtype=object).value_counts().to_dict() if sel else {}
    return out


def frac_outside(poly, scene):
    from analysis_pipeline import frac_outside_mask
    return frac_outside_mask(poly, scene.sz_mask, scene.transform)


# ---------------------------------------------------------------------------
# One run
# ---------------------------------------------------------------------------

def run(scene, trigger, mode, mask_mode='gate'):
    """Polygon, rejected-reason map and metrics for one trigger / mode / mask mode."""
    from release_geometry import make_release_polygon_2d
    feats, mel = scene.inputs(mode, mask_mode)
    state, legacy_rej, reached = {}, {}, set()

    def _cb(_ring, admitted, arrested):
        reached.update(admitted)
        for a in arrested:
            legacy_rej[a['cid']] = a['reason']

    with setting(rg, ARREST_MODE=mode, MASK_MODE=mask_mode):
        poly = make_release_polygon_2d(
            trigger_cluster_id=trigger, A_ca=scene.a_ca(trigger, mel), meloche_df=mel,
            cluster_map=scene.cluster_map, dem=scene.dem, transform=scene.transform,
            start_zone_mask=scene.sz_mask, snap_features=feats,
            pixel_index=scene.pixel_index, state_out=state,
            wave_callback=_cb if mode == 'legacy' else None)
    rejected = state.get('rejected')
    if rejected is None:                                  # legacy: arrested and never admitted
        rejected = {c: r for c, r in legacy_rej.items() if c not in reached}
    row = {'trigger': trigger, 'mode': mode, 'mask_mode': mask_mode,
           'area_m2': 0.0 if poly is None else poly.area, 'iou': 0.0,
           'over_m2': 0.0, 'under_m2': scene.observed.area}
    if poly is not None and not poly.is_empty:
        inter = poly.intersection(scene.observed).area
        row.update(iou=inter / poly.union(scene.observed).area,
                   over_m2=poly.area - inter, under_m2=scene.observed.area - inter,
                   frac_outside_mask=frac_outside(poly, scene),
                   **crown_flank_offsets(poly, scene.observed, SlopeFrame.at_cluster(scene, trigger)))
    row['n_rejected'] = len(rejected)
    row['cap_bound'] = bool(state.get('cap_bound', False))
    return poly, rejected, row


# ---------------------------------------------------------------------------
# Step 6a: fixed triggers
# ---------------------------------------------------------------------------

def sampled_crown_triggers(scene, n, seed=SEED):
    """n distinct clusters from pixels drawn uniformly inside the observed crown."""
    from rasterio.features import geometry_mask
    inside = geometry_mask([scene.observed.__geo_interface__], scene.dem.shape,
                           scene.transform, invert=True) & (scene.cluster_map > 0)
    rr, cc = np.nonzero(inside)
    rng = np.random.default_rng(seed)
    picked = []
    for k in rng.permutation(len(rr)):
        c = int(scene.cluster_map[rr[k], cc[k]])
        if c not in picked:
            picked.append(c)
        if len(picked) == n:
            break
    return picked


def step_fixed(scene, n_sampled=20):
    seg_pts, seg_lab = boundary_segments(scene.observed, scene.frame)
    trig = [('shipped', t) for t in SHIPPED_TRIGGERS] + \
           [('sampled', t) for t in sampled_crown_triggers(scene, n_sampled)]
    rows, mech = [], []
    for src, t in trig:
        for mask_mode in MASK_MODES:
            for mode in MODES:
                poly, rej, row = run(scene, t, mode, mask_mode)
                row['trigger_set'] = src
                row['has_weak_layer'] = bool(np.isfinite(scene.feats_dom['wl_shear_strength'].get(t, np.nan)))
                rows.append(row)
                for seg, counts in mechanism_table(rej, scene, seg_pts, seg_lab).items():
                    for reason, k in counts.items():
                        mech.append({'trigger': t, 'trigger_set': src, 'mode': mode,
                                     'mask_mode': mask_mode, 'segment': seg,
                                     'reason': reason, 'n': k})
                print(f"  {src:7s} {t:5d} {mask_mode:10s} {mode:19s} "
                      f"area={row['area_m2']:7.0f}  IoU={row['iou']:.3f}", flush=True)
    df = pd.DataFrame(rows)
    df['crown_offsets_m'] = df['crown_offsets_m'].apply(
        lambda v: json.dumps([None if not np.isfinite(x) else round(x, 2) for x in v])
        if isinstance(v, list) else v)
    df.to_csv(scene.out / 'fixed_triggers.csv', index=False)
    pd.DataFrame(mech).to_csv(scene.out / 'fixed_mechanisms.csv', index=False)
    return df


# ---------------------------------------------------------------------------
# Step 6b: trigger-sensitivity map
# ---------------------------------------------------------------------------

def envelope_pool(scene):
    """Start-zone clusters with psi >= stauchwall and a weak layer."""
    sz = {int(c) for c in np.unique(scene.cluster_map[scene.sz_mask]) if c > 0}
    wl = scene.feats_dom['wl_shear_strength'].notna()
    return sorted(c for c in sz if scene.psi.get(c, 0.0) >= rg.STAUCHWALL_DEG
                  and bool(wl.get(c, False)))


def step_envelope(scene, n=300, modes=MODES, seed=SEED):
    import rasterio
    pool = envelope_pool(scene)
    rng = np.random.default_rng(seed)
    trig = sorted(rng.choice(pool, size=min(n, len(pool)), replace=False).tolist())
    mask_for = {'legacy': 'gate'}          # legacy in its operational form; the rest unmasked
    summary = []
    for mode in modes:
        mask_mode = mask_for.get(mode, 'diagnostic')
        hits = np.zeros(scene.dem.shape, np.float32)
        n_ok = 0
        for k, t in enumerate(trig):
            poly, _, row = run(scene, t, mode, mask_mode)
            summary.append(row)
            if poly is not None and not poly.is_empty:
                hits += rg.rasterize_release_polygon(
                    poly, np.ones(scene.dem.shape, np.float32), scene.dem.shape,
                    scene.transform) > 0
                n_ok += 1
            if k % 25 == 0:
                print(f"  {mode} {k}/{len(trig)}", flush=True)
        frac = hits / len(trig)
        prof = scene.profile.copy()
        prof.update(dtype='float32', count=1, nodata=-1.0, compress='lzw')
        path = scene.out / f'envelope_{mode}_{mask_mode}.tif'
        with rasterio.open(path, 'w', **prof) as dst:
            dst.write(np.where(np.isfinite(scene.dem), frac, -1.0)[None].astype(np.float32))
        _plot_envelope(scene, frac, f'{mode} ({mask_mode}), {len(trig)} triggers',
                       path.with_suffix('.png'))
        print(f"  {mode}: {n_ok}/{len(trig)} triggers gave a polygon -> {path.name}", flush=True)
    pd.DataFrame(summary).drop(columns=['crown_offsets_m'], errors='ignore') \
        .to_csv(scene.out / 'envelope_runs.csv', index=False)
    pd.Series(trig, name='trigger').to_csv(scene.out / 'envelope_triggers.csv', index=False)


def _plot_envelope(scene, frac, title, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from rasterio.plot import plotting_extent
    ext = plotting_extent(scene.dem, scene.transform)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(np.where(np.isfinite(scene.dem), scene.dem, np.nan), extent=ext, cmap='Greys',
              alpha=0.35)
    im = ax.imshow(np.where(frac > 0, frac, np.nan), extent=ext, cmap='viridis', vmin=0,
                   vmax=max(float(frac.max()), 1e-6))
    x, y = scene.observed.exterior.xy
    ax.plot(x, y, color='crimson', lw=1.5, label='observed crown (6 934 m²)')
    minx, miny, maxx, maxy = scene.observed.buffer(250).bounds
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_title(title)
    ax.legend(loc='lower left')
    fig.colorbar(im, ax=ax, shrink=0.7, label='fraction of triggers whose polygon contains the cell')
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Step 8: parameter sensitivity (ranges, not fits)
# ---------------------------------------------------------------------------

def ln_pi_by_cluster(scene):
    """ln Pi for every cluster with features, under the current energy_path constants."""
    out = {}
    for cid, row in scene.feats_dom.iterrows():
        if cid in scene.psi:
            out[cid] = ep.cell_energy(row, scene.psi[cid])['ln_Pi']
    return pd.Series(out, dtype=float)


def boundary_band(scene, band=SEGMENT_BAND_M):
    """Clusters whose centroid lies within `band` m of the observed boundary, split inside/outside."""
    from shapely.geometry import Point
    ring = scene.observed.exterior
    ins, outs = set(), set()
    for c, (rr, cc) in scene.pixel_index.items():
        x, y = rg.pixel_to_utm(int(round(rr.mean())), int(round(cc.mean())), scene.transform)
        p = Point(x, y)
        if ring.distance(p) <= band:
            (ins if scene.observed.contains(p) else outs).add(c)
    return ins, outs


def step_params(scene, n_sampled=20):
    crown = scene.crown_clusters()
    b_in, b_out = boundary_band(scene)
    trig = list(SHIPPED_TRIGGERS) + sampled_crown_triggers(scene, n_sampled)
    pi_rows, run_rows = [], []
    for param, variants in PARAM_VARIANTS.items():
        for attr, val in variants:
            with setting(ep, **{attr: val}):
                lp = ln_pi_by_cluster(scene).dropna()
                inside, outside = lp[lp.index.isin(crown)], lp[~lp.index.isin(crown)]
                bi, bo = lp[lp.index.isin(b_in)], lp[lp.index.isin(b_out)]
                pi_rows.append({
                    'param': param, 'value': str(val),
                    'n_in': len(inside), 'n_out': len(outside),
                    'lnPi_in_p10': inside.quantile(.1), 'lnPi_in_p50': inside.median(),
                    'lnPi_in_p90': inside.quantile(.9),
                    'lnPi_out_p10': outside.quantile(.1), 'lnPi_out_p50': outside.median(),
                    'lnPi_out_p90': outside.quantile(.9),
                    'frac_Pi_ge1_in': float((inside >= 0).mean()),
                    'frac_Pi_ge1_out': float((outside >= 0).mean()),
                    'band_frac_Pi_ge1_in': float((bi >= 0).mean()) if len(bi) else np.nan,
                    'band_frac_Pi_ge1_out': float((bo >= 0).mean()) if len(bo) else np.nan,
                    # Pi "crosses 1 at the boundary": admitted just inside, refused just outside
                    'crosses_at_boundary': bool(len(bi) and len(bo)
                                                and (bi >= 0).mean() > 0.5 > (bo >= 0).mean()),
                })
                for t in trig:
                    _, _, row = run(scene, t, 'energy', 'diagnostic')
                    run_rows.append({'param': param, 'value': str(val), 'trigger': t,
                                     'area_m2': row['area_m2'], 'iou': row['iou'],
                                     'crown_offset_median_m': row.get('crown_offset_median_m', np.nan)})
            print(f"  {param}={val}: ln Pi in-crown median {pi_rows[-1]['lnPi_in_p50']:.2f}, "
                  f"Pi>=1 in {pi_rows[-1]['frac_Pi_ge1_in']:.2%} / out "
                  f"{pi_rows[-1]['frac_Pi_ge1_out']:.2%}", flush=True)
    pd.DataFrame(pi_rows).to_csv(scene.out / 'params_lnpi.csv', index=False)
    pd.DataFrame(run_rows).to_csv(scene.out / 'params_runs.csv', index=False)


# ---------------------------------------------------------------------------
# Step 9: spot checks and units
# ---------------------------------------------------------------------------

def step_check(scene, trigger=None):
    """Hand recomputation of tau_net, G_c, G_slab for three reached clusters, and crown medians."""
    from arrest_indices import G_ACC, plane_stress_modulus
    crown = sorted(scene.crown_clusters())
    f = scene.feats_dom.loc[[c for c in crown if c in scene.feats_dom.index]]
    f = f[f['wl_shear_strength'].notna()]
    med = {
        'E_eff_MPa': f['E_eff'].median() / 1e6,
        'sigma_t_mean_kPa': f['sigma_t_mean'].median() / 1e3,
        'slab_thickness_m': f['slab_thickness'].median(),
        'wl_thickness_cm': f['wl_thickness'].median() * 100,
        'wl_shear_strength_Pa': f['wl_shear_strength'].median(),
        'slab_density': f['slab_density'].median(),
    }
    e = pd.DataFrame({c: ep.cell_energy(f.loc[c], scene.psi[c]) for c in f.index}).T
    med.update(G_slab=e['G_slab_path'].median(), G_c=e['G_c'].median(),
               R_G_slab=(ep.energy_r() * e['G_slab_path']).median(),
               frac_Pi_lt1=float((e['ln_Pi'] < 0).mean()), n=len(f))
    print('Crown medians (shipped-default comparison in the brief: E 11 MPa, sigma_t 4.6 kPa,'
          ' h 1.6 m, D_wl 7.5 cm -> G_slab 1.4, G_c 1.65, R G_slab 0.7):')
    for k, v in med.items():
        print(f"  {k:22s} {v:.3f}" if isinstance(v, float) else f"  {k:22s} {v}")
    t = trigger or SHIPPED_TRIGGERS[0]
    state = {}
    with setting(rg, ARREST_MODE='energy', MASK_MODE='diagnostic'):
        rg.make_release_polygon_2d(
            trigger_cluster_id=t, A_ca=50.0, meloche_df=scene.mel_dom,
            cluster_map=scene.cluster_map, dem=scene.dem, transform=scene.transform,
            start_zone_mask=scene.sz_mask, snap_features=scene.feats_dom,
            pixel_index=scene.pixel_index, state_out=state)
    # Reached cells first; energy mode often reaches only the trigger, so top up from the crown.
    picks = list(state['cells'].items())[:3]
    picks += [(c, None) for c in f.index if c not in state['cells']][:3 - len(picks)]
    checks = []
    for c, rec in picks:
        r = scene.feats_dom.loc[c]
        rho, h, psi = r['slab_density'], r['slab_thickness'], scene.psi[c]
        tn = rho * G_ACC * h * (np.sin(np.radians(psi)) - np.cos(np.radians(psi))
                                * np.tan(np.radians(ep.PHI_DEG)))
        D = r['wl_thickness'] if np.isfinite(r['wl_thickness']) else ep.D_WL_FALLBACK
        gc = r['wl_shear_strength']**2 * (1 + ep.ENERGY_DELTA) / (2 * ep.G_WL / D)
        gs = r['sigma_t_mean']**2 * h / (2 * plane_stress_modulus(r['E_eff'], ep.NU))
        rec = rec or ep.cell_energy(r, psi)
        checks.append({'cluster': c, 'tau_net_hand': tn, 'tau_net': rec['tau_net'],
                       'G_c_hand': gc, 'G_c': rec['G_c'], 'G_slab_hand': gs,
                       'G_slab': rec['G_slab_path']})
    ck = pd.DataFrame(checks).set_index('cluster')
    print(ck.round(4).to_string())
    (scene.out / 'check.json').write_text(json.dumps(
        {'crown_medians': {k: float(v) for k, v in med.items()},
         'spot_checks': ck.reset_index().to_dict(orient='records')}, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('step', choices=('fixed', 'envelope', 'params', 'check'))
    ap.add_argument('--project-dir', type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument('--n-sampled', type=int, default=20)
    ap.add_argument('--n', type=int, default=300)
    ap.add_argument('--modes', nargs='+', default=list(MODES))
    args = ap.parse_args()
    scene = Scene(args.project_dir)
    if args.step == 'fixed':
        step_fixed(scene, args.n_sampled)
    elif args.step == 'envelope':
        step_envelope(scene, args.n, tuple(args.modes))
    elif args.step == 'params':
        step_params(scene, args.n_sampled)
    else:
        step_check(scene)


if __name__ == '__main__':
    main()

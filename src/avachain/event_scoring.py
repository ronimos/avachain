"""
event_scoring.py — Event-level scores for one scenario set.

Per-trigger IoU scores a rejected trigger 0 against the one observed polygon,
whether "nothing releases here" was right or not, and it says nothing on a
day with no avalanche. These scores read the scenario set as the forecast it
is: scenario weights (trigger x size x depth) sum to 1 and are probabilities
GIVEN a release, so the per-pixel sum of the weights of the scenarios that
cover it, P(pixel in release | release), is the release map.

Event day (an observed polygon on that date): Brier score and AUC of that map
against the observed mask, expected area vs observed, how much of the
observed release gets any probability, expected false area, and the trigger
weight on triggers inside the release.

No-event day: the model has no P(release), so the only correct answer is no
trigger at all, which step_scenarios signals by raising "No candidate trigger
clusters". Otherwise the score is what it would release: expected area and
the area with any probability.

The domain is start zone OR observed polygon, so release outside the start
zone (which no scenario can reach) still counts as missed.

Usage:
    python event_scoring.py --toml slopes/little_prof/slope_config.toml \
        --scen-dir outputs/little_prof/scenarios/2026-01-18/<tag> [--date 2026-01-18]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))


def load_scenario_set(scen_dir: Path) -> pd.DataFrame:
    """summary.csv rows with each scenario's release polygon (shapely)."""
    import geopandas as gpd
    scen_dir = Path(scen_dir)
    s = pd.read_csv(scen_dir / 'summary.csv')
    s['geometry'] = [gpd.read_file(scen_dir / 'scenarios' / sid / 'release.geojson').union_all()
                     for sid in s.scenario_id]
    return s


def release_probability(scen: pd.DataFrame, shape, transform) -> np.ndarray:
    """P(pixel in release | release): sum of the weights of the scenarios whose
    polygon covers the pixel. Depth percentiles share a polygon, so each
    (trigger, size factor) polygon is rasterised once with its summed weight."""
    from rasterio.features import geometry_mask
    p = np.zeros(shape, dtype=np.float64)
    for _, g in scen.groupby(['trigger_cluster', 'size_factor'], sort=False):
        geom = g.geometry.iloc[0]
        if geom is None or geom.is_empty:
            continue
        p[geometry_mask([geom.__geo_interface__], shape, transform, invert=True)] += g.weight.sum()
    return p


def auc(p: np.ndarray, o: np.ndarray) -> float:
    """Area under the ROC curve (Mann-Whitney, ties averaged); NaN if o is constant."""
    from scipy.stats import rankdata
    o = o.astype(bool)
    n1, n0 = int(o.sum()), int((~o).sum())
    if n1 == 0 or n0 == 0:
        return float('nan')
    r = rankdata(p)
    return float((r[o].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def score(prob: np.ndarray, start_zone: np.ndarray, observed: np.ndarray | None = None,
          pixel_area: float = 1.0, trigger_weights: dict | None = None,
          trigger_inside: dict | None = None) -> dict:
    """Scores over the domain start_zone | observed. observed None = no event.

    trigger_weights {cid: weight} and trigger_inside {cid: bool} add the
    trigger weight on triggers inside the release (event days only)."""
    sz = start_zone.astype(bool)
    obs = None if observed is None else observed.astype(bool)
    dom = sz if obs is None else (sz | obs)
    p = prob[dom]
    out = {'domain_m2': float(dom.sum() * pixel_area),
           'expected_area_m2': float(p.sum() * pixel_area),
           'any_prob_area_m2': float((p > 0).sum() * pixel_area)}
    if obs is None:
        out['brier'] = float(np.mean(p ** 2))
        return out
    o = obs[dom].astype(float)
    clim = o.mean()
    bs, bs_ref = float(np.mean((p - o) ** 2)), float(np.mean((clim - o) ** 2))
    out.update({
        'observed_m2': float(o.sum() * pixel_area),
        'brier': bs,
        'brier_skill': 1.0 - bs / bs_ref if bs_ref > 0 else float('nan'),
        'auc': auc(p, o),
        'expected_hit_m2': float((p * o).sum() * pixel_area),
        'expected_false_m2': float((p * (1 - o)).sum() * pixel_area),
        'observed_with_any_prob': float(np.mean(p[o > 0] > 0)),
        'mean_prob_inside': float(p[o > 0].mean()),
        'mean_prob_outside': float(p[o == 0].mean()),
        'observed_outside_start_zone_m2': float((obs & ~sz).sum() * pixel_area),
    })
    if trigger_weights and trigger_inside is not None:
        tot = sum(trigger_weights.values())
        out['trigger_weight_inside'] = (sum(w for c, w in trigger_weights.items()
                                            if trigger_inside.get(c, False)) / tot
                                        if tot > 0 else float('nan'))
    return out


def observed_on(cfg, date: str):
    """The observed release polygon dated exactly `date`, or None.

    Not cfg.release_geojsons_for_date(...)[-1]: that is the latest polygon ON
    OR BEFORE the date, so every day after Jan 18 would be scored against Jan
    18's avalanche."""
    stamp = pd.Timestamp(date).strftime('%Y%m%d')
    hits = [p for p in cfg.release_geojsons_for_date(date) if stamp in p.name]
    if not hits:
        return None
    from analysis_pipeline import _load_observed_release_polygon
    return _load_observed_release_polygon(cfg, date)


def score_scenario_dir(cfg, scen_dir: Path, date: str) -> dict:
    """Load grids from cfg, score scen_dir for `date`."""
    import rasterio
    from rasterio.features import geometry_mask
    from snowpack_io import kml_to_mask
    with rasterio.open(cfg.resampled_dir / 'dem_1m.tif') as src:
        shape, transform = (src.height, src.width), src.transform
    pixel_area = abs(transform.a * transform.e)
    sz = kml_to_mask(cfg.start_zone_kml, shape, transform)
    scen = load_scenario_set(scen_dir)
    prob = release_probability(scen, shape, transform)
    obs_poly = observed_on(cfg, date)
    if obs_poly is None:
        out = score(prob, sz, None, pixel_area)
    else:
        obs = geometry_mask([obs_poly.__geo_interface__], shape, transform, invert=True)
        cmap = np.load(cfg.analysis_dir / 'cluster_map.npy')
        tw = scen.groupby('trigger_cluster').weight.sum().to_dict()
        inside = {int(c): bool(obs[cmap == c].mean() >= 0.5) for c in tw}
        out = score(prob, sz, obs, pixel_area, tw, inside)
        out['expected_iou'] = float(sum(
            w * (g.intersection(obs_poly).area / g.union(obs_poly).area)
            for w, g in zip(scen.weight, scen.geometry) if g is not None and not g.is_empty))
    out.update({'date': date, 'scen_dir': str(scen_dir), 'event': obs_poly is not None,
                'n_triggers': int(scen.trigger_cluster.nunique()),
                'triggers': [int(t) for t in scen.trigger_cluster.unique()]})
    return out


def main():
    from config import ProjectConfig
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--toml', required=True)
    ap.add_argument('--scen-dir', required=True)
    ap.add_argument('--date', default=None, help='default: the scenario metadata snapshot_date')
    args = ap.parse_args()
    cfg = ProjectConfig.from_toml(args.toml)
    scen_dir = Path(args.scen_dir)
    date = args.date or json.loads((scen_dir / 'metadata.json').read_text())['snapshot_date']
    import contextlib
    with contextlib.redirect_stdout(sys.stderr):     # loader chatter; stdout is the JSON
        out = score_scenario_dir(cfg, scen_dir, date)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()

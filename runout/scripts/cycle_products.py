"""
Cycle Products

Author: Valerie Foley
Last Updated: 10/2026

Description:
    Builds the forecast products for one cycle from the per-horizon
    outputs and manifest.json: report.html (header, reach to road per
    horizon, probability maps, exceedance curve, reference points),
    details.html (configuration, histograms, sensitivity, scenario
    tables), scenarios.xlsx (one sheet per horizon), and viewer.html (one
    Leaflet viewer with a horizon switch). publish() copies the products to
    the web folder. Everything is read back from disk so products rebuild
    without re-simulating.
"""

# --------- Load Libraries --------
import base64
import json
import logging
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from jinja2 import Template

# The simulation modules import each other flat, so the package folder
# goes on the path before they load.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import avalanche_helpers as h
import build_viewer
import report_interactive as ri

from .manifest import PIPELINE_VERSION


# One font, black on white. Road orange and the viridis ramp are the only
# colors that carry data; horizons are told apart by line style.
COLORS = {'road': '#E87722', 'reach': '#A6192E', 'miss': '#7A8893',
          'point': '#1E2A33'}
HORIZON_STYLES = ['-', '--', '-.', ':', (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (1, 1))]
FONT = 'DejaVu Sans'


PUBLISHED_FILES = ('report.html', 'details.html', 'viewer.html', 'scenarios.xlsx')


# --------- Loading ---------

def horizon_label(label):
    # 'T_2' -> 'T+2'
    # @param label: folder label
    # @returns: str
    return 'T+' + label.split('_', 1)[1]


def no_release_reasons(manifest):
    # Why each day without a forecast has none, from the release side
    # @param manifest: cycle manifest
    # @returns: dict {label: reason} for horizons whose release set was
    #           published empty (status 'no release')
    out = {}
    for label, h in (manifest.get('horizons') or {}).items():
        rel = h.get('release') or {}
        if rel.get('status') == 'no_scenarios' and rel.get('reason'):
            out[label] = rel['reason']
    return out


PENDING_NOTE = "Still running; this page is updated when it finishes."


def pending_days(manifest):
    # Horizons listed but not run yet (the report is published after each
    # horizon, Ron 2026-11-04)
    # @param manifest: cycle manifest
    # @returns: dict {label: note}
    return {label: PENDING_NOTE for label, h in (manifest.get('horizons') or {}).items()
            if h.get('status') == 'pending'}


def load_cycle(cycle_dir):
    # Read manifest.json and every completed horizon's outputs
    # @param cycle_dir: Path to outputs/forecasts/<slug>/<YYYYMMDD>/
    # @returns: tuple (manifest dict, list of horizon dicts in numeric order)
    # @raises FileNotFoundError: if manifest.json is missing
    # a horizon counts as completed when its probability.tif and
    #       road_impact_summary.json exist, whatever the manifest says
    cycle_dir = Path(cycle_dir)
    mpath = cycle_dir / 'manifest.json'
    if not mpath.exists():
        raise FileNotFoundError(f"no manifest.json in {cycle_dir}")
    manifest = json.loads(mpath.read_text())

    horizons = []
    labels = sorted(manifest.get('horizons', {}),
                    key=lambda s: int(s.split('_', 1)[1]))
    for label in labels:
        hdir = cycle_dir / label
        if not (hdir / 'probability.tif').exists() or \
                not (hdir / 'road_impact_summary.json').exists():
            if manifest['horizons'][label].get('status') == 'pending':
                logging.info(f"[{label}] still running; shown as pending")
            elif manifest['horizons'][label].get('status') == 'no release':
                logging.info(f"[{label}] no release scenarios; reported with the reason")
            else:
                logging.warning(f"[{label}] no ensemble outputs; left out of products")
            continue
        cfg = yaml.safe_load((hdir / 'run_config.yaml').read_text())
        road = json.loads((hdir / 'road_impact_summary.json').read_text())
        rp_path = hdir / 'reference_points_summary.json'
        refpoints = json.loads(rp_path.read_text()) if rp_path.exists() else {}
        csv_path = hdir / 'scenarios_summary.csv'
        df = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()
        dem = hdir / 'merged_dem.tif'
        if not dem.exists():
            dem = Path(cfg['paths']['dem'])
        horizons.append({
            'label': label,
            'name': horizon_label(label),
            'dir': hdir,
            'config': cfg,
            'road': road,
            'refpoints': refpoints,
            'df': df,
            'dem': dem,
            'manifest': manifest['horizons'][label],
        })
    return manifest, horizons


def nearest_miss_with_rank(hz, dist=None):
    # Nearest miss plus its weight and ensemble rank (1 = highest weight)
    # @param hz: horizon dict from load_cycle
    # @param dist: optional {scenario_id: metres} to measure the miss by
    #              (the report passes distances to the path's road crossing);
    #              default is the stack's closest approach to any road point
    # @returns: dict or None
    nm = hz['road'].get('nearest_miss')
    if dist is not None and not hz['df'].empty:
        reached = dict(zip(hz['df']['scenario_id'], hz['df'].get(
            'reached_road', [False] * len(hz['df']))))
        cand = [(d, sid) for sid, d in dist.items()
                if d is not None and not bool(reached.get(sid))]
        nm = {'scenario': min(cand)[1], 'distance_short_m': min(cand)[0]} if cand else None
    if not nm or hz['df'].empty:
        return nm
    df = hz['df']
    out = dict(nm)
    row = df[df['scenario_id'] == nm['scenario']]
    if not row.empty and 'scenario_probability' in df.columns:
        w = float(row['scenario_probability'].iloc[0])
        ranks = df['scenario_probability'].rank(ascending=False, method='min')
        out['scenario_probability'] = w
        out['rank'] = int(ranks[row.index[0]])
        out['n'] = int(len(df))
    return out


def release_to_road_m(hz):
    # Distance from the release centroid to the road, from the reference
    # points when a 'road' point exists, otherwise from the geometries
    # @param hz: horizon dict
    # @returns: float or None
    rp = hz['refpoints'].get('road')
    if rp and rp.get('distance_from_release_m') is not None:
        return float(rp['distance_from_release_m'])
    try:
        road = gpd.read_file(hz['config']['paths']['road']).geometry.unary_union
        rel_dir = Path(hz['config']['paths']['scenarios_dir'])
        first = sorted(rel_dir.glob('scenario_*'))[0]
        for sub in ('release', 'inputs', '.'):
            rel = first / sub / 'release.geojson'
            if rel.exists():
                c = gpd.read_file(rel).geometry.iloc[0].centroid
                return float(c.distance(road))
    except Exception as e:
        logging.debug(f"release-to-road distance skipped: {e}")
    return None


# --------- Plots ---------

def fig_to_b64(fig, dpi=120):
    # Encode a figure as base64 PNG
    # @param fig: matplotlib Figure
    # @param dpi: int
    # @returns: str
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _extent(meta):
    # imshow extent from a read_raster meta dict
    # @param meta: dict
    # @returns: list [left, right, bottom, top]
    t = meta['transform']
    return [t.c, t.c + meta['shape'][1] * t.a,
            t.f + meta['shape'][0] * t.e, t.f]


def _hillshade(ax, dem_path):
    # Draw a grayscale hillshade on an axis
    # @param ax: matplotlib Axes
    # @param dem_path: Path
    # @returns: None
    dem, meta = h.read_raster(dem_path)
    bad = ~np.isfinite(dem)
    if meta['nodata'] is not None:
        bad |= (dem == meta['nodata'])
    if bad.any():
        dem = np.where(bad, dem[~bad].mean() if (~bad).any() else 0.0, dem)
    cell = abs(meta['transform'].a)
    shade = LightSource(azdeg=315, altdeg=45).hillshade(dem * 1.3, dx=cell, dy=cell)
    ax.imshow(np.ma.masked_where(bad, shade), extent=_extent(meta),
              origin='upper', cmap='gray', vmin=0, vmax=1, zorder=1)


def plot_small_multiples(horizons, dpi=120):
    # One map per horizon: hillshade, probability, road, impact points,
    # shared colorbar
    # @param horizons: list of horizon dicts
    # @param dpi: int
    # @returns: str base64 PNG
    n = len(horizons)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 6.2), squeeze=False,
                             constrained_layout=True)
    im = None
    for ax, hz in zip(axes[0], horizons):
        prob, meta = h.read_raster(hz['dir'] / 'probability.tif')
        try:
            _hillshade(ax, hz['dem'])
        except Exception as e:
            logging.warning(f"[{hz['label']}] hillshade skipped: {e}")
        im = ax.imshow(np.ma.masked_where(~(prob > 0), prob), extent=_extent(meta),
                       origin='upper', cmap='viridis', vmin=0, vmax=1,
                       alpha=0.8, zorder=2)
        road = gpd.read_file(hz['config']['paths']['road'])
        road.plot(ax=ax, color=COLORS['road'], linewidth=2.2, zorder=4)
        pts = [p['xy'] for p in hz['road'].get('impact_points', []) if p.get('xy')]
        if pts:
            ax.scatter([p[0] for p in pts], [p[1] for p in pts],
                       c=COLORS['reach'], s=28, zorder=5, linewidths=0)
        rp_file = hz['config']['paths'].get('reference_points')
        if rp_file and Path(rp_file).exists():
            try:
                rp = gpd.read_file(rp_file)
                if rp.crs is not None and meta['crs'] is not None and rp.crs != meta['crs']:
                    rp = rp.to_crs(meta['crs'])
                for _, r in rp.iterrows():
                    ax.plot(r.geometry.x, r.geometry.y, marker='^', ms=6,
                            color=COLORS['point'], zorder=6, linestyle='none')
                    ax.annotate(str(r['ref_name']), (r.geometry.x, r.geometry.y),
                                xytext=(4, 4), textcoords='offset points',
                                fontsize=7, color=COLORS['point'], zorder=6)
            except Exception as e:
                logging.debug(f"reference points not drawn: {e}")
        pw = hz['road'].get('fraction_reached_weighted')
        p = pw if pw is not None else hz['road'].get('fraction_reached', 0.0)
        ax.set_title(f"{hz['name']}  valid {hz['manifest']['valid_date']}\n"
                     f"P(reach road) = {p:.2f}", fontsize=10, fontfamily=FONT)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        for s in ax.spines.values():
            s.set_linewidth(0.6)
    if im is not None:
        cb = fig.colorbar(im, ax=axes[0].tolist(), shrink=0.75, pad=0.02)
        cb.set_label('P(reach)', fontsize=9)
    return fig_to_b64(fig, dpi)


def _weight_denominator(df, w):
    # sum(w), or the set's fixed ensemble_weight_total when its weights are
    # absolute (forced explosive shots: the rest of 1 is no release)
    # @param df: scenarios table
    # @param w: weights array
    # @returns: float
    if 'ensemble_weight_total' in df.columns:
        t = pd.to_numeric(df['ensemble_weight_total'], errors='coerce').max()
        if pd.notna(t) and t > 0:
            return float(t)
    return float(w.sum())


def road_approach_curve(df, dist=None):
    # Weighted P(closest approach <= d) as a step curve
    # @param df: scenarios table (distance_to_road_m, reached_road,
    #            scenario_probability)
    # @param dist: optional {scenario_id: metres} replacing
    #              distance_to_road_m (distances to the path's road crossing)
    # @returns: (xs, ys) for a 'post' step plot, xs[0] = 0 with ys[0] =
    #          P(reach road); or None when the table has no distances
    if df.empty or 'distance_to_road_m' not in df.columns:
        return None
    if dist is not None:
        d = df['scenario_id'].map(lambda k: dist.get(k)).astype(float).to_numpy()
    else:
        d = df['distance_to_road_m'].to_numpy(dtype=float)
    if 'reached_road' in df.columns:
        d = np.where(df['reached_road'].astype(bool).to_numpy(), 0.0, d)
    w = df['scenario_probability'].to_numpy(dtype=float) \
        if 'scenario_probability' in df.columns else np.ones_like(d)
    w = np.where(np.isfinite(w) & (w > 0), w, 1.0)
    ok = np.isfinite(d)
    d, w = d[ok], w[ok]
    if not len(d):
        return None
    order = np.argsort(d)
    d, w = d[order], w[order]
    den = _weight_denominator(df, w)
    cum = np.cumsum(w) / den
    p0 = float(w[d <= 0].sum() / den)
    return np.concatenate([[0.0], d]), np.concatenate([[p0], cum])


def plot_road_approach(horizons, dpi=120, dists=None):
    # Weighted P(flow comes within d of the road), one line per horizon.
    # Replaces the runout exceedance in the report (2026-11-02): that plot
    # set each scenario's runout (farthest reach cell from ITS OWN release
    # centroid, in any direction) against one road line (scenario 001's
    # centroid to the nearest road point). Releases differ, so on Jan 18
    # runouts passed the road line while no scenario reached the road.
    # distance_to_road_m and reached_road are the same measure the reach
    # table uses, so d = 0 here is P(reach road).
    # @param horizons: list of horizon dicts
    # @param dpi: int
    # @param dists: optional {label: {scenario_id: metres to the crossing}}
    # @returns: str base64 PNG or None
    fig, ax = plt.subplots(figsize=(8, 4.2))
    drew, xmax = False, 0.0
    for i, hz in enumerate(horizons):
        curve = road_approach_curve(hz['df'], (dists or {}).get(hz['label']))
        if curve is None:
            continue
        xs, ys = curve
        ax.step(xs, ys, where='post', color=ri.horizon_color(hz['label']),
                linestyle=HORIZON_STYLES[i % len(HORIZON_STYLES)], linewidth=1.8,
                label=f"{hz['name']}  ({len(xs) - 1} scenarios)")
        xmax = max(xmax, float(xs.max()))
        drew = True
    if not drew:
        plt.close(fig)
        return None
    ax.axvline(0, color=COLORS['road'], linewidth=2)
    ax.text(0, 1.02, 'road', ha='center', va='bottom', fontsize=8, color=COLORS['road'])
    # Reversed so the path runs left to right: up the path on the left,
    # the road on the right (Ron, 2026-11-03)
    ax.set_xlim(xmax * 1.05, -xmax * 0.01)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Runout distance from where the avalanche path crosses the road (m), "
                  "the flow's closest approach" if dists else
                  "Runout distance from the road (m), the flow's closest approach", fontsize=9)
    ax.set_ylabel('P(flow comes at least this close)', fontsize=9)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8, frameon=False, loc='upper right')
    ax.grid(axis='y', linewidth=0.4, color='#DDDDDD')
    return fig_to_b64(fig, dpi)


def plot_exceedance(horizons, dpi=120):
    # Weighted P(runout >= x), one line per horizon, with road and
    # reference points as vertical marks
    # @param horizons: list of horizon dicts
    # @param dpi: int
    # @returns: str base64 PNG or None when no scenario table exists
    fig, ax = plt.subplots(figsize=(8, 4.2))
    drew = False
    xmax = 0.0
    for i, hz in enumerate(horizons):
        df = hz['df']
        if df.empty or 'runout_max_m' not in df.columns:
            continue
        r = df['runout_max_m'].to_numpy(dtype=float)
        w = df['scenario_probability'].to_numpy(dtype=float) \
            if 'scenario_probability' in df.columns else np.ones_like(r)
        w = np.where(np.isfinite(w) & (w > 0), w, 1.0)
        order = np.argsort(r)
        r, w = r[order], w[order]
        # P(X >= x) steps down at each scenario's runout: on [r_i, r_i+1)
        # the share of weight with runout >= r_i+1, and 0 past the longest
        den = _weight_denominator(df, w)
        tail = np.cumsum(w[::-1])[::-1] / den
        xs = np.concatenate([[0.0], r])
        ys = np.concatenate([[tail[0]], tail[1:], [0.0]])
        ax.step(xs, ys, where='post', color='black',
                linestyle=HORIZON_STYLES[i % len(HORIZON_STYLES)],
                linewidth=1.4, label=f"{hz['name']}  ({len(df)} scenarios)")
        xmax = max(xmax, float(r.max()))
        drew = True
    if not drew:
        plt.close(fig)
        return None

    marks = {}
    for hz in horizons:
        for name, rp in hz['refpoints'].items():
            d = rp.get('distance_from_release_m')
            if d is not None:
                marks[name] = float(d)
    if 'road' not in marks:
        d = release_to_road_m(horizons[0])
        if d is not None:
            marks['road'] = d
    for name, d in marks.items():
        col = COLORS['road'] if name == 'road' else COLORS['point']
        ax.axvline(d, color=col, linewidth=1.2 if name == 'road' else 0.8,
                   linestyle='-' if name == 'road' else ':')
        ax.text(d, 1.02, name, rotation=0, ha='center', va='bottom',
                fontsize=8, color=col)
        xmax = max(xmax, d)
    ax.set_xlim(0, xmax * 1.05)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel('Runout distance from release (m)', fontsize=9)
    ax.set_ylabel('P(runout at least this far)', fontsize=9)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(axis='y', linewidth=0.4, color='#DDDDDD')
    return fig_to_b64(fig, dpi)


def plot_chainage(horizons, dpi=120):
    # Weighted histogram of impact chainage, one panel per horizon that
    # has any impact
    # @param horizons: list of horizon dicts
    # @param dpi: int
    # @returns: str base64 PNG or None when nothing reached
    rows = []
    for hz in horizons:
        pts = hz['road'].get('impact_points', [])
        if not pts:
            continue
        wmap = {}
        if not hz['df'].empty and 'scenario_probability' in hz['df'].columns:
            wmap = dict(zip(hz['df']['scenario_id'], hz['df']['scenario_probability']))
        ch = [p['chainage_m'] for p in pts if p.get('chainage_m') is not None]
        w = [float(wmap.get(p['scenario'], 1.0) or 1.0) for p in pts
             if p.get('chainage_m') is not None]
        if ch:
            rows.append((hz, ch, w))
    if not rows:
        return None
    # Shared 10 m bins so horizons line up and a single hit stays readable
    allc = np.concatenate([np.asarray(ch, dtype=float) for _, ch, _ in rows])
    lo = np.floor(allc.min() / 10.0) * 10.0 - 20.0
    hi = np.ceil(allc.max() / 10.0) * 10.0 + 20.0
    bins = np.arange(lo, hi + 10.0, 10.0)
    fig, axes = plt.subplots(1, len(rows), figsize=(5 * len(rows), 3.4),
                             squeeze=False, sharey=True)
    for ax, (hz, ch, w) in zip(axes[0], rows):
        ax.hist(ch, bins=bins, weights=w, color='#444444', edgecolor='white')
        ax.set_title(hz['name'], fontsize=10)
        ax.set_xlabel('Chainage along road (m)', fontsize=9)
        ax.tick_params(labelsize=8)
    axes[0][0].set_ylabel('Weighted share of scenarios', fontsize=9)
    return fig_to_b64(fig, dpi)


def plot_runout_hist(df, dpi=120):
    # Stacked histogram of runout, reached vs missed
    # @param df: scenarios DataFrame
    # @param dpi: int
    # @returns: str base64 PNG or None
    if df.empty or 'runout_max_m' not in df.columns:
        return None
    fig, ax = plt.subplots(figsize=(7, 3.4))
    reached = df[df['reached_road'] == True]['runout_max_m']
    missed = df[df['reached_road'] == False]['runout_max_m']
    ax.hist([reached, missed], bins=15, stacked=True,
            label=['Reached road', 'Did not reach'],
            color=[COLORS['reach'], COLORS['miss']], edgecolor='white')
    ax.set_xlabel('Max runout distance (m)', fontsize=9)
    ax.set_ylabel('Scenarios', fontsize=9)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8, frameon=False)
    return fig_to_b64(fig, dpi)


# Columns that may vary across the ensemble, plotted only when they do
SENSITIVITY_AXES = [
    ('size_factor', 'Size factor'),
    ('depth_percentile', 'Depth percentile'),
    ('release_volume_m3', 'Release volume (m3)'),
    ('mean_depth_m', 'Mean release depth (m)'),
    ('scour_depth_params_m', 'Scour depth (m)'),
    ('hand_hardness_value', 'Hand hardness index'),
    ('rho_kgm3', 'Release density (kg/m3)'),
    ('scenario_probability', 'Scenario weight'),
]


def plot_sensitivity(df, dpi=120):
    # Runout against every input that varies across the ensemble
    # @param df: scenarios DataFrame
    # @param dpi: int
    # @returns: str base64 PNG or None when nothing varies
    usable = [(k, lbl) for k, lbl in SENSITIVITY_AXES
              if k in df.columns and df[k].notna().sum() > 1
              and df[k].nunique(dropna=True) > 1]
    if not usable or df.empty:
        return None
    cols = min(3, len(usable))
    rows = int(np.ceil(len(usable) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 3.4 * rows),
                             squeeze=False)
    colors = [COLORS['reach'] if r else COLORS['miss'] for r in df['reached_road']]
    for ax, (key, label) in zip(axes.flat, usable):
        ax.scatter(df[key], df['runout_max_m'], c=colors, s=26, linewidths=0)
        ax.set_xlabel(label, fontsize=9)
        ax.set_ylabel('Runout (m)', fontsize=9)
        ax.tick_params(labelsize=8)
    for ax in list(axes.flat)[len(usable):]:
        ax.axis('off')
    fig.tight_layout()
    return fig_to_b64(fig, dpi)


# --------- HTML ---------

CSS = """
  body { font-family: Arial, Helvetica, sans-serif; font-size: 13px;
         color: #000; background: #fff; max-width: 1200px;
         margin: 1.5em auto; padding: 0 1.2em 3em; line-height: 1.4; }
  h1 { font-size: 20px; font-weight: bold; margin: 0 0 0.2em; }
  h2 { font-size: 15px; font-weight: bold; margin: 1.6em 0 0.5em;
       border-bottom: 1px solid #000; padding-bottom: 0.2em; }
  .hdr { border: 1px solid #000; padding: 0.6em 0.9em; margin-bottom: 1em; }
  .hdr table { width: auto; }
  .hdr td { border: none; padding: 0.1em 1.2em 0.1em 0; text-align: left; }
  table { border-collapse: collapse; width: 100%; font-size: 12px; }
  th, td { border: 1px solid #000; padding: 0.35em 0.6em; text-align: right; }
  th { background: #fff; font-weight: bold; text-align: center; }
  td.l, th.l { text-align: left; }
  td.c { text-align: center; }
  .big { font-size: 18px; font-weight: bold; }
  img { max-width: 100%; height: auto; display: block; margin: 0.6em 0; }
  .meta { font-size: 11px; margin-top: 2em; border-top: 1px solid #000;
          padding-top: 0.5em; }
  .warn { border: 1px solid #000; padding: 0.4em 0.8em; margin: 0.6em 0;
          font-weight: bold; }
  a { color: #000; }
  ul.caveats { margin: 0.5em 0 0 1.2em; padding: 0; }
  ul.caveats li { margin: 0.15em 0; }
"""

REPORT_TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>{{ project }} runout forecast {{ cycle_date }}</title>
<style>{{ css }}{{ imap_css }}</style></head>
<body>
<h1>AVALANCHE RUNOUT FORECAST: {{ project|upper }}</h1>
<div class="hdr"><table>
<tr><td>Path</td><td>{{ project }}{% if subtitle %}, {{ subtitle }}{% endif %}</td></tr>
<tr><td>Issued</td><td>{{ issued }}</td></tr>
<tr><td>Forecast date</td><td>{{ cycle_date }}</td></tr>
<tr><td>{{ snow_line[0] }}</td><td>{{ snow_line[1] }}</td></tr>
{% for hz in horizons %}<tr><td>{{ hz.name }}</td><td>valid {{ hz.valid }}</td></tr>{% endfor %}
</table></div>

{% if failed %}<div class="warn">Horizons not run: {{ failed }}</div>{% endif %}
{% if pending %}<div class="warn"><b>{{ pending }} still running.</b> This page is updated as each day finishes.</div>{% endif %}
{% for name, valid, reason in no_release %}<div class="warn"><b>No forecast for {{ name }}{% if valid %} ({{ valid }}){% endif %}.</b> {{ reason }}</div>
{% endfor %}

<h2>Probability of reach</h2>
{% if imap_html %}{{ imap_html }}
<div class="imap-print" style="display:none"><img src="data:image/png;base64,{{ maps_b64 }}" alt="Probability maps"></div>
{% else %}<img src="data:image/png;base64,{{ maps_b64 }}" alt="Probability maps">{% endif %}

<h2>Reach to road</h2>
<table>
<thead><tr>
  <th class="l">Horizon</th><th>Valid</th><th>Scenarios</th>
  <th>Reached road</th><th>P(reach road)<br>weighted</th>
  <th>P(reach road)<br>unweighted, 95% CI</th>
  <th>Nearest miss{% if nm_ref %}<br>(to where the path<br>crosses the road){% endif %}</th><th>Max runout (m)</th>
</tr></thead>
<tbody>
{% for hz in horizons %}
<tr>
  <td class="l">{{ hz.name }}</td>
  <td class="c">{{ hz.valid }}</td>
  <td>{{ hz.n }}</td>
  <td>{{ hz.n_reached }}</td>
  <td>{{ hz.p_w }}</td>
  <td>{{ hz.p_u }} [{{ hz.ci_lo }}, {{ hz.ci_hi }}]</td>
  <td class="l">{% if hz.nm %}{{ hz.nm.scenario }}: {{ "%.0f"|format(hz.nm.distance_short_m) }} m short,
      weight {{ "%.4f"|format(hz.nm.scenario_probability) }}{% if hz.nm.rank %}, rank {{ hz.nm.rank }} of {{ hz.nm.n }}{% endif %}{% else %}-{% endif %}</td>
  <td>{{ hz.max_runout }}</td>
</tr>
{% endfor %}
</tbody></table>

{% if exceed_b64 %}
<h2>Runout distance from road</h2>
<img src="data:image/png;base64,{{ exceed_b64 }}" alt="Probability of coming within a distance of the road">
{% endif %}

{% if ihist_html %}
<h2>Scenarios by closest approach to the road</h2>
{{ ihist_html }}
{% endif %}

{% if refpoint_rows %}
<h2>Reference points, P(reach)</h2>
<table>
<thead><tr><th class="l">Point</th><th>Distance from release (m)</th>
{% for hz in horizons %}<th>{{ hz.name }}</th>{% endfor %}</tr></thead>
<tbody>
{% for row in refpoint_rows %}
<tr><td class="l">{{ row.name }}</td><td>{{ row.dist }}</td>
{% for v in row.cells %}<td>{{ v }}</td>{% endfor %}</tr>
{% endfor %}
</tbody></table>
{% endif %}

{% if chainage_b64 %}
<h2>Where the road is hit</h2>
<img src="data:image/png;base64,{{ chainage_b64 }}" alt="Chainage histogram">
{% endif %}

<h2>Inputs and caveats</h2>
<table style="width:auto">
{% for k, v in inputs %}<tr><td class="l">{{ k }}</td><td class="l">{{ v }}</td></tr>{% endfor %}
</table>
{% if caveats %}<ul class="caveats">{% for c in caveats %}<li>{{ c }}</li>{% endfor %}</ul>{% endif %}

<div class="meta">
Physics {{ physics_name }} [{{ physics_hash }}], {{ friction_model }}{{ friction_params }},
sim type {{ sim_type }}, reach threshold {{ reach_threshold }} m, entrainment {{ ent_mode }},
depths {{ depth_convention }}. Pipeline v{{ version }}{% if git %} ({{ git }}){% endif %}{% if avaframe %}, AvaFrame {{ avaframe }}{% endif %}.
{% if absolute_weights %}Probabilities are per shot fired: each explosive-control shot carries an equal share, and a shot that does not release counts as not reaching.{% else %}Probabilities are the weighted fraction of snowpack scenarios whose flow reaches a location, conditional on release.{% endif %}
<a href="details.html">Details</a>, <a href="viewer.html">Viewer</a>, <a href="scenarios.xlsx">Scenario table</a>.
</div>
{% if imap_data %}<script type="application/json" id="imap-data">{{ imap_data }}</script>
<script>{{ imap_js }}</script>{% endif %}
</body></html>
"""

DETAILS_TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>{{ project }} runout details {{ cycle_date }}</title>
<style>{{ css }}</style></head>
<body>
<h1>RUNOUT DETAILS: {{ project|upper }}, {{ cycle_date }}</h1>
<p>Issued {{ issued }}. <a href="report.html">Back to the forecast</a>.</p>

{% for hz in horizons %}
<h2>{{ hz.name }}, valid {{ hz.valid }}</h2>
<table style="width:auto">
{% for k, v in hz.config_rows %}<tr><td class="l">{{ k }}</td><td class="l">{{ v }}</td></tr>{% endfor %}
</table>

{% if hz.hist_b64 %}<img src="data:image/png;base64,{{ hz.hist_b64 }}" alt="Runout histogram">{% endif %}
{% if hz.sens_b64 %}<img src="data:image/png;base64,{{ hz.sens_b64 }}" alt="Sensitivity">{% endif %}

<table>
<thead><tr>
  <th class="l">Scenario</th><th>Weight</th><th>Size factor</th><th>Depth pct</th><th>Depth (m)</th><th>Vol (m3)</th>
  <th>Scour (m)</th><th>Hardness</th><th>Runout (m)</th><th>Peak v (m/s)</th>
  <th>Peak depth (m)</th><th>Reached</th><th>Dist to road (m)</th>
  <th>Depth at road (m)</th><th>v at road (m/s)</th>
  {% for p in hz.points %}<th>{{ p }}</th>{% endfor %}
</tr></thead>
<tbody>
{% for r in hz.rows %}
<tr>
  <td class="l">{{ r.scenario_id }}</td>
  <td>{{ r.scenario_probability }}</td>
  <td>{{ r.size_factor }}</td>
  <td>{{ r.depth_percentile }}</td>
  <td>{{ r.mean_depth_m }}</td>
  <td>{{ r.release_volume_m3 }}</td>
  <td>{{ r.scour_depth_params_m }}</td>
  <td>{{ r.hand_hardness_value }}</td>
  <td>{{ r.runout_max_m }}</td>
  <td>{{ r.peak_velocity_ms }}</td>
  <td>{{ r.peak_flow_depth_m }}</td>
  <td class="c">{{ 'yes' if r.reached_road else '-' }}</td>
  <td>{{ r.distance_to_road_m }}</td>
  <td>{{ r.depth_at_road_m }}</td>
  <td>{{ r.velocity_at_road_ms }}</td>
  {% for p in hz.points %}<td class="c">{{ 'yes' if r['reached_' ~ p] else '-' }}</td>{% endfor %}
</tr>
{% endfor %}
</tbody></table>
{% endfor %}
</body></html>
"""


def _fmt(v, nd=2):
    # Number to fixed decimals, '-' for missing
    # @param v: value
    # @param nd: decimals
    # @returns: str
    try:
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return '-'
        return f"{float(v):.{nd}f}"
    except Exception:
        return str(v)


def release_set_info(raster_path):
    # The release handoff set an entrainment raster came from, when it is
    # one (handoff/<path>/release/<date>/<set>/scour_depth.tif with a
    # release manifest.json beside it)
    # @param raster_path: str or None
    # @returns: dict (set, snapshot_date, snapshot_ts, store, clamped,
    #          created_utc, commit) or None
    if not raster_path:
        return None
    m = Path(raster_path).parent / 'manifest.json'
    try:
        man = json.loads(m.read_text())
    except Exception:
        return None
    if man.get('producer') != 'release':
        return None
    return {'set': man.get('horizon') or Path(raster_path).parent.name,
            'snapshot_date': man.get('snapshot_date'),
            'snapshot_ts': man.get('snapshot_ts'),
            'store': man.get('store'),
            'clamped': bool(man.get('clamped')),
            'created_utc': man.get('created_utc'),
            'commit': (man.get('commit') or '')[:7]}


def entrainment_rule(ent):
    # One line saying which entrainment rule the run used
    # @param ent: run_config entrainment section
    # @returns: str
    mode = ent.get('mode', 'auto')
    raster = Path(ent['snow_height_raster']).name if ent.get('snow_height_raster') else None
    if mode == 'raster' and raster:
        if ent.get('raster_cap', True):
            return f"{raster}, capped at each scenario's scour depth"
        return f"{raster} as is, no cap"
    if mode == 'polygon':
        return "constant scour depth inside the path's entrainment polygon"
    return "constant scour depth along the auto-detected corridor"


def inputs_and_caveats(manifest, horizons):
    # Rows for the report's inputs box, the caveats under it, and the
    # header line for the snowpack input
    # @param manifest: cycle manifest
    # @param horizons: completed horizon dicts
    # @returns: dict {rows: [(label, value)], caveats: [str], snow_line: (label, value)}
    cfg = horizons[0]['config']
    ent = cfg.get('entrainment') or {}
    rows = []
    caveats_extra = []
    # each horizon has its own release set (and scour raster) under rule B
    rels = [(hz, release_set_info((hz['config'].get('entrainment') or {})
                                  .get('snow_height_raster'))) for hz in horizons]
    if all(r for _, r in rels):
        def _when(r):
            # forecast days: the snapshot time and the store it came from
            if r.get('snapshot_ts'):
                return (f"{r['snapshot_ts'].replace('T', ' ')} UTC"
                        + (f" ({r['store']})" if r.get('store') else ''))
            return r['snapshot_date']
        sets = {r['set'] for _, r in rels}
        snow_line = ('Snowpack', ', '.join(f"{hz['name']} {_when(r)}" for hz, r in rels)
                     + ('' if len(sets) > 1 else f" (release set {rels[0][1]['set']})"))
        for hz, rel in rels:
            if rel.get('clamped'):
                caveats_extra.append(
                    f"{hz['name']}: the WRF outlook ends before this day's snapshot hour, so "
                    f"its snowpack is the outlook's last step, {_when(rel)}.")
            rows.append((f"Release scenarios {hz['name']}",
                         f"set {rel['set']}, snowpack {_when(rel)}, "
                         f"built {(rel['created_utc'] or '').replace('T', ' ').replace('Z', ' UTC')}"
                         f"{', commit ' + rel['commit'] if rel['commit'] else ''}"))
    else:
        snow_line = ('Last snow height survey', (manifest.get('survey') or {}).get('date') or 'none')
    rows.append(('Scenarios', ', '.join(f"{hz['name']} {len(hz['df'])}" for hz in horizons)))
    rows.append(('Entrainment', entrainment_rule(ent)))
    fr = cfg.get('friction') or {}
    fparams = ', '.join(f"{k} = {fr[k]}" for k in ('mu', 'xi', 'tau0') if fr.get(k) is not None)
    rows.append(('Friction', f"{cfg['com1dfa']['friction_model']}" + (f" ({fparams})" if fparams else '')))
    rows.append(('Runout DEM', Path(cfg['paths']['dem']).name))
    rows.append(('Reach', f"flow depth &ge; {cfg['reach']['flow_depth_threshold_m']} m"))
    rows.append(('Software', f"pipeline v{manifest.get('pipeline_version', PIPELINE_VERSION)}"
                 f"{' (' + manifest['git_commit'] + ')' if manifest.get('git_commit') else ''}"
                 f", AvaFrame {manifest.get('avaframe_version') or 'unknown'}"))
    caveats = caveats_extra + cycle_caveats(manifest)
    reasons = {**no_release_reasons(manifest), **pending_days(manifest)}
    done = {hz['label'] for hz in horizons}
    missing = [f"T_{n}" for n in range(3) if f"T_{n}" not in done]
    for label in missing:
        if label in reasons:
            caveats.insert(0, f"{horizon_label(label)}: {reasons[label]}")
    unexplained = [horizon_label(l) for l in missing if l not in reasons]
    if unexplained:
        caveats.insert(0, f"No forecast for {', '.join(unexplained)}: no release scenarios were "
                          f"delivered for those days.")
    return {'rows': rows, 'caveats': caveats, 'snow_line': snow_line}


def cycle_caveats(manifest):
    # The path's caveats for this cycle: display.event_caveats for the
    # cycle's own date first, then display.caveats, which hold for every
    # cycle. A dated note (the 2026-01-18 debris) never reaches another
    # day's report.
    # @param manifest: cycle manifest (cycle_date 'YYYYMMDD' or 'YYYY-MM-DD')
    # @returns: list of caveat strings
    display = manifest.get('display') or {}
    d = str(manifest.get('cycle_date') or '').replace('-', '')
    key = f"{d[:4]}-{d[4:6]}-{d[6:8]}" if len(d) == 8 else None
    dated = {str(k): v for k, v in (display.get('event_caveats') or {}).items()}
    return list(dated.get(key) or []) + list(display.get('caveats') or [])


def build_report(cycle_dir, manifest, horizons, dpi=120):
    # Render report.html for the cycle
    # @param cycle_dir: Path to the cycle folder
    # @param manifest: dict
    # @param horizons: list of horizon dicts (completed ones)
    # @param dpi: int
    # @returns: Path to report.html
    try:
        payload = ri.build_payload(horizons, manifest['cycle_date'],
                                   reasons={**no_release_reasons(manifest), **pending_days(manifest)},
                                   pending=set(pending_days(manifest)))
        imap = ri.render_parts(payload)
    except Exception as e:
        # the static maps still render; the interactive part is a bonus
        logging.warning(f"interactive map skipped: {e}", exc_info=True)
        payload, imap = None, ri.render_parts(None)
    # Distances to the path's road crossing, the same reference as the
    # cursor readout and the histogram (Ron, 2026-11-02)
    dists = None
    if payload and payload.get('dist_ref') == 'crossing':
        dists = {sl['label']: {q['id']: q['d'] for q in sl.get('scenarios', [])}
                 for sl in payload['slots'] if not sl.get('missing')}

    rows = []
    for hz in horizons:
        road = hz['road']
        pw = road.get('fraction_reached_weighted')
        ci = road.get('wilson_95_ci', [0, 0])
        rows.append({
            'name': hz['name'], 'valid': hz['manifest']['valid_date'],
            'n': road.get('n_scenarios'), 'n_reached': road.get('n_reached'),
            'p_w': _fmt(pw if pw is not None else road.get('fraction_reached')),
            'p_u': _fmt(road.get('fraction_reached')),
            'ci_lo': _fmt(ci[0]), 'ci_hi': _fmt(ci[1]),
            'nm': nearest_miss_with_rank(hz, (dists or {}).get(hz['label'])),
            'max_runout': _fmt(hz['df']['runout_max_m'].max(), 0)
            if not hz['df'].empty else '-',
        })

    # Reference points: rows ordered by distance, one column per horizon
    names = {}
    for hz in horizons:
        for n, rp in hz['refpoints'].items():
            names[n] = rp.get('distance_from_release_m')
    ref_rows = []
    for n in sorted(names, key=lambda k: names[k] if names[k] is not None else 1e9):
        vals = []
        for hz in horizons:
            rp = hz['refpoints'].get(n)
            if not rp:
                vals.append('-')
                continue
            pw = rp.get('fraction_reached_weighted')
            vals.append(_fmt(pw if pw is not None else rp.get('fraction_reached')))
        ref_rows.append({'name': n, 'dist': _fmt(names[n], 0), 'cells': vals})

    reasons = no_release_reasons(manifest)
    pend = pending_days(manifest)
    failed = [f"{horizon_label(k)} ({v['status']})"
              for k, v in manifest.get('horizons', {}).items()
              if k not in {hz['label'] for hz in horizons} and k not in reasons and k not in pend]
    no_release = [(horizon_label(k), manifest['horizons'][k].get('valid_date'), r)
                  for k, r in sorted(reasons.items(),
                                     key=lambda kv: int(kv[0].split('_', 1)[1]))]

    cfg = horizons[0]['config']
    fm = cfg['com1dfa']['friction_model']
    fr = cfg.get('friction') or {}
    fparams = ', '.join(f"{k}={fr[k]}" for k in ('mu', 'xi', 'tau0')
                        if fr.get(k) is not None)
    box = inputs_and_caveats(manifest, horizons)
    absolute_weights = any(
        'ensemble_weight_total' in hz['df'].columns
        and pd.to_numeric(hz['df']['ensemble_weight_total'], errors='coerce').notna().any()
        for hz in horizons if not hz['df'].empty)
    html = Template(REPORT_TEMPLATE).render(
        css=CSS, **imap, inputs=box['rows'], caveats=box['caveats'],
        absolute_weights=absolute_weights,
        snow_line=box['snow_line'],
        project=manifest.get('display', {}).get('project_name', manifest['path']),
        subtitle=manifest.get('display', {}).get('subtitle', ''),
        cycle_date=f"{manifest['cycle_date'][:4]}-{manifest['cycle_date'][4:6]}-{manifest['cycle_date'][6:]}",
        issued=manifest.get('issued', '').replace('T', ' '),
        survey_date=(manifest.get('survey') or {}).get('date'),
        horizons=rows, failed=', '.join(failed), no_release=no_release,
        pending=', '.join(horizon_label(k) for k in sorted(pend, key=lambda l: int(l.split('_', 1)[1]))),
        maps_b64=plot_small_multiples(horizons, dpi),
        exceed_b64=plot_road_approach(horizons, dpi, dists),
        nm_ref=' to the road crossing' if dists else '',
        chainage_b64=plot_chainage(horizons, dpi),
        refpoint_rows=ref_rows,
        physics_name=manifest['physics']['name'],
        physics_hash=manifest['physics']['hash'],
        friction_model=fm, friction_params=f" ({fparams})" if fparams else '',
        sim_type=cfg['com1dfa']['sim_type'],
        reach_threshold=cfg['reach']['flow_depth_threshold_m'],
        ent_mode=manifest['profile'].get('entrainment_mode'),
        depth_convention=manifest['physics'].get('depth_convention'),
        version=manifest.get('pipeline_version', PIPELINE_VERSION),
        git=manifest.get('git_commit'), avaframe=manifest.get('avaframe_version'),
    )
    out = Path(cycle_dir) / 'report.html'
    out.write_text(html)
    logging.info(f"wrote {out}")
    return out


def build_details(cycle_dir, manifest, horizons, dpi=120):
    # Render details.html: config, histograms, sensitivity, scenario tables
    # @param cycle_dir: Path
    # @param manifest: dict
    # @param horizons: list of horizon dicts
    # @param dpi: int
    # @returns: Path to details.html
    blocks = []
    for hz in horizons:
        cfg = hz['config']
        fr = cfg.get('friction') or {}
        ent = cfg.get('entrainment') or {}
        config_rows = [
            ('Friction model', cfg['com1dfa']['friction_model']),
            ('Friction parameters', ', '.join(f"{k}={fr[k]}" for k in ('mu', 'xi', 'tau0')
                                              if fr.get(k) is not None) or 'model defaults'),
            ('Simulation type', cfg['com1dfa']['sim_type']),
            ('Mesh cell size', f"{cfg['com1dfa']['mesh_cell_size_m']} m"),
            ('Reach threshold', f"{cfg['reach']['flow_depth_threshold_m']} m flow depth"
             + (f", {cfg['reach']['pressure_threshold_kpa']} kPa" if cfg['reach'].get('pressure_threshold_kpa') else '')),
            ('Entrainment', f"{ent.get('mode')}" + (f", {Path(ent['snow_height_raster']).name}" if ent.get('snow_height_raster') else '')),
            ('Scenario weighting', 'by scenario probability' if cfg['probability'].get('weight_by_scenario_probability') else 'equal'),
            ('Road buffer', f"{cfg['road']['buffer_m']} m"),
            ('Scenario source', hz['manifest'].get('source')),
            ('Outputs', str(hz['dir'])),
        ]
        df = hz['df'].copy()
        points = [c[len('reached_'):] for c in df.columns
                  if c.startswith('reached_') and c != 'reached_road']
        recs = []
        for _, r in df.iterrows():
            d = {k: (_fmt(v) if isinstance(v, float) else v) for k, v in r.items()}
            d['scenario_probability'] = _fmt(r.get('scenario_probability'), 4)
            d['release_volume_m3'] = _fmt(r.get('release_volume_m3'), 0)
            d['runout_max_m'] = _fmt(r.get('runout_max_m'), 0)
            d['distance_to_road_m'] = _fmt(r.get('distance_to_road_m'), 0)
            recs.append(d)
        blocks.append({
            'name': hz['name'], 'valid': hz['manifest']['valid_date'],
            'config_rows': config_rows, 'rows': recs, 'points': points,
            'hist_b64': plot_runout_hist(df, dpi),
            'sens_b64': plot_sensitivity(df, dpi),
        })
    html = Template(DETAILS_TEMPLATE).render(
        css=CSS,
        project=manifest.get('display', {}).get('project_name', manifest['path']),
        cycle_date=manifest['cycle_date'],
        issued=manifest.get('issued', '').replace('T', ' '),
        horizons=blocks,
    )
    out = Path(cycle_dir) / 'details.html'
    out.write_text(html)
    logging.info(f"wrote {out}")
    return out


def write_xlsx(cycle_dir, manifest, horizons):
    # scenarios.xlsx: summary sheet, reference points, one sheet per horizon
    # @param cycle_dir: Path
    # @param manifest: dict
    # @param horizons: list of horizon dicts
    # @returns: Path to the workbook
    out = Path(cycle_dir) / 'scenarios.xlsx'
    summary = []
    for hz in horizons:
        road = hz['road']
        nm = nearest_miss_with_rank(hz) or {}
        summary.append({
            'horizon': hz['name'], 'valid_date': hz['manifest']['valid_date'],
            'n_scenarios': road.get('n_scenarios'), 'n_reached': road.get('n_reached'),
            'p_reach_weighted': road.get('fraction_reached_weighted'),
            'p_reach_unweighted': road.get('fraction_reached'),
            'ci_low': road.get('wilson_95_ci', [None, None])[0],
            'ci_high': road.get('wilson_95_ci', [None, None])[1],
            'nearest_miss': nm.get('scenario'),
            'nearest_miss_short_m': nm.get('distance_short_m'),
            'nearest_miss_weight': nm.get('scenario_probability'),
            'nearest_miss_rank': nm.get('rank'),
            'max_runout_m': hz['df']['runout_max_m'].max() if not hz['df'].empty else None,
        })
    refs = []
    for hz in horizons:
        for n, rp in hz['refpoints'].items():
            refs.append({'horizon': hz['name'], 'ref_name': n,
                         'distance_from_release_m': rp.get('distance_from_release_m'),
                         'n_reached': rp.get('n_reached'), 'n_scenarios': rp.get('n_scenarios'),
                         'p_reach_weighted': rp.get('fraction_reached_weighted'),
                         'p_reach_unweighted': rp.get('fraction_reached')})
    with pd.ExcelWriter(out, engine='openpyxl') as xw:
        pd.DataFrame(summary).to_excel(xw, sheet_name='summary', index=False)
        if refs:
            pd.DataFrame(refs).to_excel(xw, sheet_name='reference_points', index=False)
        for hz in horizons:
            hz['df'].to_excel(xw, sheet_name=hz['name'].replace('+', '_'), index=False)
        pd.DataFrame([{
            'issued': manifest.get('issued'), 'cycle_date': manifest['cycle_date'],
            'path': manifest['path'], 'physics': manifest['physics']['name'],
            'physics_hash': manifest['physics']['hash'],
            'survey_date': (manifest.get('survey') or {}).get('date'),
            'pipeline_version': manifest.get('pipeline_version'),
            'git_commit': manifest.get('git_commit'),
        }]).T.to_excel(xw, sheet_name='manifest', header=False)
    logging.info(f"wrote {out}")
    return out


# --------- Viewer ---------

def _scenarios_for_viewer(hz):
    # Rebuild the scenario dicts build_viewer needs from the work tree
    # @param hz: horizon dict
    # @returns: list of scenario dicts
    out = []
    work = Path(hz['config']['paths'].get('scenario_work_dir') or hz['config']['paths']['scenarios_dir'])
    for sd in sorted(work.glob('scenario_*')):
        rel = None
        for sub in ('release', 'inputs', '.'):
            p = sd / sub / 'release.geojson'
            if p.exists():
                rel = p
                break
        if rel is None or not (sd / 'outputs' / 'summary.json').exists():
            continue
        out.append({'scenario_id': sd.name, 'scenario_dir': str(sd),
                    'work_dir': str(sd), 'release_geojson': str(rel)})
    return out


CYCLE_VIEWER = """<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>{{ project }} runout viewer {{ cycle_date }}</title>
<style>
  html, body { margin: 0; height: 100%; font-family: Arial, Helvetica, sans-serif; }
  #bar { height: 40px; background: #fff; border-bottom: 1px solid #000;
         display: flex; align-items: center; gap: 8px; padding: 0 12px;
         font-size: 13px; }
  #bar b { margin-right: 12px; }
  #bar button { font: inherit; padding: 4px 14px; border: 1px solid #000;
                background: #fff; cursor: pointer; }
  #bar button.on { background: #000; color: #fff; }
  #bar span.v { margin-left: auto; font-size: 11px; }
  iframe { border: none; width: 100%; height: calc(100% - 41px); display: none; }
  iframe.on { display: block; }
</style></head>
<body>
<div id="bar"><b>{{ project }} {{ cycle_date }}</b>
{% for hz in horizons %}<button data-h="{{ hz.label }}">{{ hz.name }} ({{ hz.valid }})</button>{% endfor %}
<span class="v">issued {{ issued }}</span></div>
{% for hz in horizons %}<iframe id="frame-{{ hz.label }}"></iframe>{% endfor %}
<script>
var docs = {{ docs_json }};
var loaded = {};
function show(h) {
  document.querySelectorAll('#bar button').forEach(function(b) {
    b.classList.toggle('on', b.getAttribute('data-h') === h); });
  document.querySelectorAll('iframe').forEach(function(f) {
    f.classList.toggle('on', f.id === 'frame-' + h); });
  if (!loaded[h]) { document.getElementById('frame-' + h).srcdoc = docs[h]; loaded[h] = true; }
}
document.querySelectorAll('#bar button').forEach(function(b) {
  b.addEventListener('click', function() { show(b.getAttribute('data-h')); }); });
show('{{ horizons[0].label }}');
</script>
</body></html>
"""


def build_cycle_viewer(cycle_dir, manifest, horizons):
    # Build each horizon's viewer into its folder, then one viewer.html
    # with a horizon switch that embeds them
    # @param cycle_dir: Path
    # @param manifest: dict
    # @param horizons: list of horizon dicts
    # @returns: Path to viewer.html, or None when no horizon viewer built
    docs = {}
    built = []
    for hz in horizons:
        cfg = hz['config']
        shared = {'dem': Path(hz['dem']), 'forest': Path(cfg['paths']['forest']),
                  'road': Path(cfg['paths']['road'])}
        try:
            path = build_viewer.build_viewer(
                outputs_dir=hz['dir'], scenarios=_scenarios_for_viewer(hz),
                shared_paths=shared, impact_points=hz['road'].get('impact_points', []),
                config=cfg)
            docs[hz['label']] = Path(path).read_text()
            built.append(hz)
        except Exception as e:
            logging.error(f"[{hz['label']}] viewer failed: {e}", exc_info=True)
    if not built:
        return None
    html = Template(CYCLE_VIEWER).render(
        project=manifest.get('display', {}).get('project_name', manifest['path']),
        cycle_date=manifest['cycle_date'],
        issued=manifest.get('issued', '').replace('T', ' '),
        horizons=[{'label': hz['label'], 'name': hz['name'],
                   'valid': hz['manifest']['valid_date']} for hz in built],
        docs_json=json.dumps(docs),
    )
    out = Path(cycle_dir) / 'viewer.html'
    out.write_text(html)
    logging.info(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    return out


def build_cycle_products(cycle_dir, products=('report', 'viewer')):
    # Build every requested product for one cycle
    # @param cycle_dir: Path to outputs/forecasts/<slug>/<YYYYMMDD>/
    # @param products: iterable of 'report', 'viewer' (details and the
    #                  workbook come with the report)
    # @returns: list of Paths written
    # @raises RuntimeError: if no horizon has ensemble outputs
    manifest, horizons = load_cycle(cycle_dir)
    if not horizons:
        if no_release_reasons(manifest) and 'report' in products:
            # every day ended with no release: say so and why
            return [build_no_release_report(cycle_dir, manifest)]
        raise RuntimeError(f"no completed horizons in {cycle_dir}")
    dpi = int((horizons[0]['config'].get('output') or {}).get('report_plots_dpi', 120))
    written = []
    if 'report' in products:
        written.append(build_report(cycle_dir, manifest, horizons, dpi))
        written.append(build_details(cycle_dir, manifest, horizons, dpi))
        written.append(write_xlsx(cycle_dir, manifest, horizons))
    if 'viewer' in products:
        p = build_cycle_viewer(cycle_dir, manifest, horizons)
        if p:
            written.append(p)
    return written


NO_RELEASE_TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>{{ project }} runout forecast {{ cycle_date }}</title>
<style>{{ css }}</style></head>
<body>
<h1>AVALANCHE RUNOUT FORECAST: {{ project|upper }}</h1>
<div class="hdr"><table>
<tr><td>Path</td><td>{{ project }}{% if subtitle %}, {{ subtitle }}{% endif %}</td></tr>
<tr><td>Issued</td><td>{{ issued }}</td></tr>
<tr><td>Forecast date</td><td>{{ cycle_date }}</td></tr>
</table></div>

<h2>No release scenarios</h2>
<p>The release model produced no avalanche release for any forecast day, so no
runout was simulated. This is a model result, not a failure.</p>
<table>
<thead><tr><th class="l">Day</th><th>Valid</th><th class="l">Why there is no forecast</th></tr></thead>
<tbody>
{% for d in days %}<tr><td class="l">{{ d.name }}</td><td class="c">{{ d.valid or '-' }}</td><td class="l">{{ d.reason }}</td></tr>
{% endfor %}
</tbody></table>

{% if caveats %}<h2>Caveats</h2>
<ul class="caveats">{% for c in caveats %}<li>{{ c }}</li>{% endfor %}</ul>{% endif %}

<div class="meta">Pipeline v{{ version }}{% if git %} ({{ git }}){% endif %}.</div>
</body></html>
"""


def build_no_release_report(cycle_dir, manifest):
    # report.html for a cycle where no day had release scenarios
    # @param cycle_dir: Path to the cycle folder
    # @param manifest: cycle manifest
    # @returns: Path to report.html
    reasons = no_release_reasons(manifest)
    days = []
    for label, h in sorted((manifest.get('horizons') or {}).items(),
                           key=lambda kv: int(kv[0].split('_', 1)[1])):
        days.append({'name': horizon_label(label), 'valid': h.get('valid_date'),
                     'reason': reasons.get(label) or pending_days(manifest).get(label)
                     or f"not run ({h.get('status')})"})
    cd = manifest['cycle_date']
    html = Template(NO_RELEASE_TEMPLATE).render(
        css=CSS, days=days, caveats=cycle_caveats(manifest),
        project=manifest.get('display', {}).get('project_name', manifest['path']),
        subtitle=manifest.get('display', {}).get('subtitle', ''),
        cycle_date=f"{cd[:4]}-{cd[4:6]}-{cd[6:]}",
        issued=manifest.get('issued', '').replace('T', ' '),
        version=manifest.get('pipeline_version', PIPELINE_VERSION),
        git=manifest.get('git_commit'))
    out = Path(cycle_dir) / 'report.html'
    out.write_text(html)
    logging.info(f"no-release report written: {out}")
    return out


# --------- Publishing ---------

INDEX_TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><title>{{ slug }} runout forecasts</title>
<style>{{ css }}</style></head>
<body>
<h1>RUNOUT FORECASTS: {{ slug|upper }}</h1>
<p><a href="latest/report.html">Latest forecast</a></p>
<table style="width:auto">
<thead><tr><th class="l">Cycle</th><th class="l">Report</th><th class="l">Details</th><th class="l">Viewer</th><th class="l">Table</th></tr></thead>
<tbody>
{% for d, have in dates %}<tr><td class="l">{{ d }}</td>
{% for f, t in [('report.html', 'report'), ('details.html', 'details'), ('viewer.html', 'viewer'), ('scenarios.xlsx', 'xlsx')] %}<td class="l">{% if f in have %}<a href="{{ d }}/{{ f }}">{{ t }}</a>{% else %}-{% endif %}</td>
{% endfor %}</tr>
{% endfor %}
</tbody></table>
</body></html>
"""


def publish(cycle_dir, publish_root, slug, cycle_date):
    # Copy the cycle products to the web folder and refresh latest/ and
    # the per-path index
    # @param cycle_dir: Path to outputs/forecasts/<slug>/<YYYYMMDD>/
    # @param publish_root: Path from local.yaml
    # @param slug: path slug
    # @param cycle_date: str YYYYMMDD
    # @returns: Path to the published cycle folder, or None when nothing
    #          was there to publish
    # latest/ is a copy, not a symlink, so it works on web servers
    #       that do not follow links; the index lists dates newest first
    import shutil
    cycle_dir = Path(cycle_dir)
    present = [f for f in PUBLISHED_FILES if (cycle_dir / f).exists()]
    if not present:
        logging.warning(f"nothing to publish in {cycle_dir}")
        return None
    path_root = Path(publish_root) / 'forecasts' / slug
    dest = path_root / cycle_date
    dest.mkdir(parents=True, exist_ok=True)
    for f in present:
        shutil.copy2(cycle_dir / f, dest / f)
    # a rerun that now has fewer products (a no-release day) must not leave
    # the earlier run's details/viewer/table beside the new report
    for f in PUBLISHED_FILES:
        if f not in present and (dest / f).exists():
            (dest / f).unlink()

    # latest/ = newest dated folder present under the path
    dates = sorted((d.name for d in path_root.iterdir()
                    if d.is_dir() and d.name.isdigit() and len(d.name) == 8),
                   reverse=True)
    latest = path_root / 'latest'
    if dates:
        newest = path_root / dates[0]
        if latest.exists():
            shutil.rmtree(latest)
        shutil.copytree(newest, latest)

    # a no-release cycle has only report.html: link what each date has
    rows = [(d, {f.name for f in (path_root / d).iterdir()}) for d in dates]
    (path_root / 'index.html').write_text(
        Template(INDEX_TEMPLATE).render(css=CSS, slug=slug, dates=rows))
    logging.info(f"published {', '.join(present)} -> {dest} (latest = {dates[0] if dates else '-'})")
    return dest

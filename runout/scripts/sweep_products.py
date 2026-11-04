"""
Sweep Products

Author: Valerie Foley
Last Updated: 10/2026

Description:
    Builds the products of a depth sweep from the per-cell outputs:
    sweep_table.csv (one row per simulation), lookup.csv (smallest depth
    that reaches each point per class and entrainment), sweep.png (runout
    against depth by class) and sweep.html. publish() copies them to the
    web folder.
"""

# --------- Load Libraries --------
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from jinja2 import Template

from .cycle_products import CSS, fig_to_b64
from .manifest import PIPELINE_VERSION


PUBLISHED_FILES = ('sweep.html', 'sweep.png', 'sweep_table.csv', 'lookup.csv')


def load_sweep(sweep_dir):
    # Read the manifest and every completed cell's per-scenario summaries
    # @param sweep_dir: Path to outputs/calibration/<slug>/sweep_<name>/
    # @returns: tuple (manifest, DataFrame with one row per simulation,
    #          dict of point name -> distance from release)
    # @raises FileNotFoundError: if manifest.json is missing
    sweep_dir = Path(sweep_dir)
    mpath = sweep_dir / 'manifest.json'
    if not mpath.exists():
        raise FileNotFoundError(f"no manifest.json in {sweep_dir}")
    manifest = json.loads(mpath.read_text())

    rows = []
    point_dist = {}
    for label, cell in (manifest.get('cells') or {}).items():
        work = Path(cell['work_dir'])
        n = 0
        for sd in sorted(work.glob('scenario_*')):
            sp = sd / 'outputs' / 'summary.json'
            if not sp.exists():
                continue
            s = json.loads(sp.read_text())
            params = s.get('params') or {}
            row = {
                'cell': label,
                'class': cell['class'], 'tau0_pa': cell['tau0'],
                'density_kgm3': cell['density'], 'entrainment': cell['entrainment'],
                'scenario_id': s['scenario_id'],
                'depth_m': params.get('depth_m', params.get('mean_depth_m')),
                'volume_m3': params.get('total_volume_m3'),
                'runout_m': s.get('runout_max_distance_m'),
                'peak_velocity_ms': s.get('peak_velocity_max_ms'),
                'peak_flow_depth_m': s.get('peak_flow_depth_max_m'),
                'reached_road': bool(s['road']['reached']),
                'short_road_m': s['road']['distance_to_road_m'],
                'velocity_at_road_ms': s['road'].get('velocity_at_road_ms'),
            }
            for r in s.get('reference_points') or []:
                nm = r['ref_name']
                point_dist.setdefault(nm, r.get('distance_from_release_m'))
                row[f'reached_{nm}'] = bool(r['reached'])
                row[f'short_{nm}_m'] = r.get('distance_short_m')
            rows.append(row)
            n += 1
        if n == 0:
            logging.warning(f"[{label}] no completed scenarios; left out")
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(['class', 'entrainment', 'depth_m']).reset_index(drop=True)
    return manifest, df, point_dist


def point_names(df):
    # Reference point names present in the table, road first, then by
    # the order the columns appear
    # @param df: sweep table
    # @returns: list of names
    names = [c[len('reached_'):] for c in df.columns if c.startswith('reached_')]
    if 'road' in names:
        names.remove('road')
        names = ['road'] + names
    return names


def build_lookup(df, points):
    # Smallest depth that reaches each point, per class and entrainment
    # @param df: sweep table
    # @param points: list of point names
    # @returns: DataFrame with class, tau0, entrainment, point, first_depth_m,
    #          runout_at_max_depth_m, max_depth_m
    out = []
    for (cls, ent), g in df.groupby(['class', 'entrainment'], sort=False):
        g = g.sort_values('depth_m')
        tau0 = g['tau0_pa'].iloc[0]
        for p in points:
            col = f'reached_{p}'
            hit = g[g[col] == True] if col in g.columns else g.iloc[0:0]
            out.append({
                'class': cls, 'tau0_pa': tau0, 'entrainment': ent, 'point': p,
                'first_depth_m': float(hit['depth_m'].iloc[0]) if not hit.empty else None,
                'max_depth_m': float(g['depth_m'].max()),
                'runout_at_max_depth_m': float(g['runout_m'].iloc[-1]),
            })
    return pd.DataFrame(out)


def plot_sweep(df, point_dist, dpi=120):
    # Runout against depth, one line per class; solid with entrainment,
    # dashed without; reference points as horizontal lines
    # @param df: sweep table
    # @param point_dist: dict of point name -> distance from release
    # @param dpi: int
    # @returns: str base64 PNG
    fig, ax = plt.subplots(figsize=(8.5, 5))
    classes = list(dict.fromkeys(df['class']))
    greys = np.linspace(0.75, 0.0, max(len(classes), 2))
    for i, cls in enumerate(classes):
        for ent, ls in (('on', '-'), ('off', '--')):
            g = df[(df['class'] == cls) & (df['entrainment'] == ent)].sort_values('depth_m')
            if g.empty:
                continue
            tau0 = g['tau0_pa'].iloc[0]
            ax.plot(g['depth_m'], g['runout_m'], linestyle=ls, marker='o', ms=3,
                    color=str(greys[i]), linewidth=1.4,
                    label=f"{cls} (tau0 {tau0:g} Pa){'' if ent == 'on' else ', no entrainment'}")
    for name, d in sorted(point_dist.items(), key=lambda kv: kv[1] or 0):
        if d is None:
            continue
        col = '#E87722' if name == 'road' else '#1E2A33'
        ax.axhline(d, color=col, linewidth=1.2 if name == 'road' else 0.8,
                   linestyle='-' if name == 'road' else ':')
        ax.text(ax.get_xlim()[0], d, f' {name}', va='bottom', ha='left',
                fontsize=8, color=col)
    ax.set_xlabel('Start zone depth (m, slope-normal)', fontsize=9)
    ax.set_ylabel('Runout distance from release (m)', fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(linewidth=0.4, color='#DDDDDD')
    ax.legend(fontsize=7, frameon=False, ncol=2)
    return fig_to_b64(fig, dpi)


SWEEP_TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>{{ project }} depth sweep {{ name }}</title>
<style>{{ css }}</style></head>
<body>
<h1>START ZONE DEPTH SWEEP: {{ project|upper }}</h1>
<div class="hdr"><table>
<tr><td>Path</td><td>{{ project }}{% if subtitle %}, {{ subtitle }}{% endif %}</td></tr>
<tr><td>Sweep</td><td>{{ name }}</td></tr>
<tr><td>Run</td><td>{{ issued }}</td></tr>
<tr><td>Release area</td><td>{{ release }} ({{ area }} m2)</td></tr>
<tr><td>Depths</td><td>{{ depths }} m, slope-normal</td></tr>
<tr><td>Entrainment</td><td>polygon at {{ ent_depth }} m; each class run with and without</td></tr>
</table></div>

{% if failed %}<div class="warn">Cells not run: {{ failed }}</div>{% endif %}

<h2>Friction classes</h2>
<table style="width:auto">
<thead><tr><th class="l">Class</th><th>Slab hardness</th><th>tau0 (Pa)</th><th>Density (kg/m3)</th></tr></thead>
<tbody>{% for c in classes %}<tr><td class="l">{{ c.name }}</td><td class="c">{{ c.hardness }}</td><td>{{ c.tau0 }}</td><td>{{ c.density }}</td></tr>{% endfor %}</tbody>
</table>

<h2>Smallest start zone depth that reaches each point (m)</h2>
<table>
<thead><tr><th class="l">Class</th><th>Entrainment</th>{% for p in points %}<th>{{ p }}</th>{% endfor %}<th>Runout at {{ max_depth }} m</th></tr></thead>
<tbody>
{% for r in lookup_rows %}<tr><td class="l">{{ r.cls }}</td><td class="c">{{ r.ent }}</td>{% for v in r.cells %}<td>{{ v }}</td>{% endfor %}<td>{{ r.runout }} m</td></tr>
{% endfor %}
</tbody></table>
<p>"&gt; {{ max_depth }}" means the point was not reached at any depth in the sweep.</p>

<h2>Runout against depth</h2>
<img src="data:image/png;base64,{{ fig_b64 }}" alt="Sweep figure">

<div class="meta">
tau0 values are placeholders keyed to slab hand hardness, inside the dry-snow cohesion range of Bartelt et al. (2015) and bracketing the 125 Pa fit to the 2026-01-18 Little Professor event. They are replaced row by row as calibration runs land.
Runout is the straight-line distance from the release centroid to the farthest reached cell, with reach defined as peak flow depth of at least {{ reach_threshold }} m.
Physics {{ physics_name }} [{{ physics_hash }}], {{ friction_model }}, classes from {{ classes_file }}. Pipeline v{{ version }}.
<a href="sweep_table.csv">Full table</a>, <a href="lookup.csv">Lookup</a>.
</div>
</body></html>
"""

HARDNESS = {-1: 'F', -2: '4F', -3: '1F', -4: 'P', -5: 'K'}


def build_sweep_products(sweep_dir, dpi=120):
    # Build the table, lookup, figure and page for one sweep
    # @param sweep_dir: Path to outputs/calibration/<slug>/sweep_<name>/
    # @param dpi: int
    # @returns: list of Paths written
    # @raises RuntimeError: if no cell has completed scenarios
    sweep_dir = Path(sweep_dir)
    manifest, df, point_dist = load_sweep(sweep_dir)
    if df.empty:
        raise RuntimeError(f"no completed cells in {sweep_dir}")
    sw = manifest.get('sweep') or {}
    points = point_names(df)
    lookup = build_lookup(df, points)

    table_path = sweep_dir / 'sweep_table.csv'
    df.to_csv(table_path, index=False)
    lookup_path = sweep_dir / 'lookup.csv'
    lookup.to_csv(lookup_path, index=False)

    fig_b64 = plot_sweep(df, point_dist, dpi)
    import base64
    png_path = sweep_dir / 'sweep.png'
    png_path.write_bytes(base64.b64decode(fig_b64))

    max_depth = float(df['depth_m'].max())
    lookup_rows = []
    for (cls, ent), g in lookup.groupby(['class', 'entrainment'], sort=False):
        cells = []
        for p in points:
            v = g[g['point'] == p]['first_depth_m']
            v = v.iloc[0] if not v.empty else None
            cells.append(f"{v:.2f}" if v is not None and not pd.isna(v) else f"> {max_depth:g}")
        lookup_rows.append({'cls': cls, 'ent': ent, 'cells': cells,
                            'runout': f"{g['runout_at_max_depth_m'].iloc[0]:.0f}"})

    classes = [{'name': c['name'],
                'hardness': HARDNESS.get(c.get('hand_hardness'), '-'),
                'tau0': c['tau0'], 'density': c['density']}
               for c in sw.get('classes') or []]
    failed = [f"{k} ({v['status']})" for k, v in (manifest.get('cells') or {}).items()
              if v['status'] != 'ok' and k not in set(df['cell'])]

    phys = manifest['physics']
    area = df['volume_m3'].iloc[0] / df['depth_m'].iloc[0] if df['depth_m'].iloc[0] else 0
    html = Template(SWEEP_TEMPLATE).render(
        css=CSS,
        project=manifest.get('display', {}).get('project_name', manifest['path']),
        subtitle=manifest.get('display', {}).get('subtitle', ''),
        name=sw.get('name'), issued=manifest.get('issued', '').replace('T', ' '),
        release=Path(sw.get('release', '')).name, area=f"{area:,.0f}",
        depths=', '.join(f"{d:g}" for d in sw.get('depths') or []),
        ent_depth=sw.get('entrainment_depth_m'),
        failed=', '.join(failed), classes=classes, points=points,
        lookup_rows=lookup_rows, max_depth=f"{max_depth:g}", fig_b64=fig_b64,
        reach_threshold=phys['resolved']['reach']['flow_depth_threshold_m'],
        physics_name=phys['name'], physics_hash=phys['hash'],
        friction_model=phys['resolved']['com1dfa']['friction_model'],
        classes_file=sw.get('classes_file'),
        version=manifest.get('pipeline_version', PIPELINE_VERSION),
    )
    page = sweep_dir / 'sweep.html'
    page.write_text(html)
    for p in (table_path, lookup_path, png_path, page):
        logging.info(f"wrote {p}")
    return [table_path, lookup_path, png_path, page]


INDEX_TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><title>{{ slug }} sweeps</title>
<style>{{ css }}</style></head>
<body>
<h1>DEPTH SWEEPS: {{ slug|upper }}</h1>
<table style="width:auto">
<thead><tr><th class="l">Sweep</th><th class="l">Page</th><th class="l">Table</th><th class="l">Lookup</th></tr></thead>
<tbody>{% for n in names %}<tr><td class="l">{{ n }}</td>
<td class="l"><a href="sweep_{{ n }}/sweep.html">page</a></td>
<td class="l"><a href="sweep_{{ n }}/sweep_table.csv">csv</a></td>
<td class="l"><a href="sweep_{{ n }}/lookup.csv">csv</a></td></tr>{% endfor %}</tbody>
</table>
</body></html>
"""


def publish(sweep_dir, publish_root, slug, name):
    # Copy the sweep products to the web folder and refresh the index
    # @param sweep_dir: Path to the sweep outputs
    # @param publish_root: Path from local.yaml
    # @param slug: path slug
    # @param name: sweep name
    # @returns: Path to the published folder, or None when nothing to copy
    sweep_dir = Path(sweep_dir)
    present = [f for f in PUBLISHED_FILES if (sweep_dir / f).exists()]
    if not present:
        logging.warning(f"nothing to publish in {sweep_dir}")
        return None
    path_root = Path(publish_root) / 'calibration' / slug
    dest = path_root / f'sweep_{name}'
    dest.mkdir(parents=True, exist_ok=True)
    for f in present:
        shutil.copy2(sweep_dir / f, dest / f)
    names = sorted(d.name[len('sweep_'):] for d in path_root.iterdir()
                   if d.is_dir() and d.name.startswith('sweep_'))
    (path_root / 'index.html').write_text(
        Template(INDEX_TEMPLATE).render(css=CSS, slug=slug, names=names))
    logging.info(f"published {', '.join(present)} -> {dest}")
    return dest

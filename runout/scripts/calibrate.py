"""
Calibration

Author: Valerie Foley
Last Updated: 10/2026

Description:
    Calibrate mode. Runs one scenario across a parameter grid from the
    profile and scores every cell against an observed event, then records
    the fit in a catalog. The grid can name any physics key, so the same
    mode fits tau0 under samosAT or mu and xi under Voellmy. The scenario
    is either a named ensemble member or one built from the observed
    release polygon. Scores are taken at several reach thresholds from one
    simulation. The recommended fit and its range follow what was
    observed. The operational physics file is never edited here.
"""

# --------- Load Libraries --------
import base64
import datetime
import itertools
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from jinja2 import Template

from . import config as cfgmod
from . import events
from . import generate_scenarios as gen
from . import manifest as mf
from . import profiles as prof
from . import scoring
from .cycle_products import CSS, fig_to_b64
from .manifest import PIPELINE_VERSION


DEFAULT_TOLERANCE = {'iou': 0.05, 'toe_m': 10.0}


# --------- Grid ---------

def expand_grid(grid):
    # Cartesian product of a {'section.key': [values]} grid
    # @param grid: dict
    # @returns: list of (label, overrides) where overrides is
    #          {section: {key: value}} and label is e.g. 'tau0_125'
    # @raises prof.ProfileError: if a key is malformed
    if not grid:
        raise prof.ProfileError("calibrate needs a non-empty `grid` in the profile "
                                "(e.g. grid: {friction.tau0: [75, 100, 125, 150]})")
    keys = list(grid)
    for k in keys:
        if '.' not in k or k.split('.', 1)[0] not in prof.PROFILE_SECTIONS:
            raise prof.ProfileError(f"grid key {k!r} must be section.key")
    values = [list(grid[k]) if isinstance(grid[k], (list, tuple)) else [grid[k]]
              for k in keys]
    cells = []
    for combo in itertools.product(*values):
        overrides = {}
        parts = []
        for k, v in zip(keys, combo):
            section, key = k.split('.', 1)
            overrides.setdefault(section, {})[key] = v
            parts.append(f"{key}_{v:g}" if isinstance(v, (int, float)) else f"{key}_{v}")
        cells.append(('__'.join(parts), overrides))
    return cells


def grid_columns(cells):
    # Parameter names varied by the grid, for tables
    # @param cells: list from expand_grid
    # @returns: list of 'section.key'
    cols = []
    for _, ov in cells:
        for section, kv in ov.items():
            for key in kv:
                name = f"{section}.{key}"
                if name not in cols:
                    cols.append(name)
    return cols


# --------- Scenario source ---------

def stage_scenario(calib, event, local, slug, path_profile, inputs_dir, args):
    # Put the one scenario to calibrate on under inputs_dir
    # @param calib: profile `calibrate` dict
    # @param event: dict from events.load_event
    # @param local: dict from config.load_local
    # @param slug: path slug
    # @param path_profile: dict
    # @param inputs_dir: Path to write scenario_###/ under
    # @param args: parsed args (--scenario, --scenario-date override)
    # @returns: tuple (scenario_id, description)
    # @raises cfgmod.ConfigError: if the member or the polygon is missing
    sid = args.scenario or calib.get('scenario') or 'synthetic'
    if inputs_dir.exists():
        shutil.rmtree(inputs_dir)
    inputs_dir.mkdir(parents=True)

    if sid == 'synthetic':
        release = event['release']
        if release is None:
            raise cfgmod.ConfigError("calibrate.scenario is synthetic but the event "
                                     "record has no release_polygon")
        slab = event['slab'] or {}
        depth = slab.get('depth_m')
        if depth is None:
            raise cfgmod.ConfigError("synthetic scenario needs slab.depth_m in event.yaml")
        rho = float(slab.get('density_kgm3', 300.0))
        rel_path = inputs_dir / '_release.geojson'
        gpd.GeoDataFrame({'name': ['observed']}, geometry=[release],
                         crs=path_profile['_crs']).to_file(rel_path, driver='GeoJSON')
        gen.generate_scenarios(out_dir=inputs_dir, release_path=rel_path,
                               dem_path=path_profile['terrain']['dem'],
                               depths=[float(depth)], densities=[rho],
                               scour_depth_m=float(slab.get('scour_depth_m', 0.0)),
                               release_label='observed')
        return 'scenario_001', f"synthetic from observed release, {depth} m, {rho:g} kg/m3"

    date = args.scenario_date or calib.get('scenario_date') or event['date'].replace('-', '')
    horizon = calib.get('horizon', 'T_0')
    src = Path(local['scenario_root']) / path_profile['scenario_slug'] / date
    cand = [src / horizon / sid, src / horizon / 'scenarios' / sid, src / 'scenarios' / sid]
    found = next((c for c in cand if c.is_dir()), None)
    if found is None:
        raise cfgmod.ConfigError(f"scenario {sid} not found for {slug} {date} {horizon} "
                                 f"(tried {', '.join(str(c) for c in cand)})")
    inp = found / 'inputs' if (found / 'inputs').is_dir() else found
    dst = inputs_dir / sid / 'release'
    dst.mkdir(parents=True)
    for f in inp.iterdir():
        if f.is_file():
            shutil.copy2(f, dst / f.name)
    return sid, f"ensemble member {sid}, {date} {horizon}"


# --------- Recommendation ---------

def recommend(scores, obs_types, tolerance, threshold):
    # Pick the recommended cell and the range of cells within tolerance
    # @param scores: DataFrame of scores (one row per cell and threshold)
    # @param obs_types: list from the event record
    # @param tolerance: dict with iou and toe_m
    # @param threshold: the reach threshold to judge at
    # @returns: dict with rule, best (cell label or None), range (list of
    #          cell labels), metric, best_value
    s = scores[scores['threshold_m'] == threshold].copy()
    if s.empty:
        return {'rule': 'none', 'best': None, 'range': [], 'metric': None, 'best_value': None}
    if 'polygon' in obs_types:
        s = s.sort_values(['iou', 'abs_toe'] if 'abs_toe' in s.columns else ['iou'],
                          ascending=[False, True] if 'abs_toe' in s.columns else [False])
        best = s.iloc[0]
        rng = s[s['iou'] >= best['iou'] - tolerance['iou']]
        return {'rule': 'highest IoU', 'best': best['cell'], 'range': list(rng['cell']),
                'metric': 'iou', 'best_value': float(best['iou'])}
    if 'toe' in obs_types:
        s = s.sort_values('abs_toe')
        best = s.iloc[0]
        rng = s[s['abs_toe'] <= best['abs_toe'] + tolerance['toe_m']]
        return {'rule': 'smallest |toe error|', 'best': best['cell'], 'range': list(rng['cell']),
                'metric': 'toe_error_m', 'best_value': float(best['toe_error_m'])}
    feasible = s[s['points_agree'] == 1.0]
    return {'rule': 'all points agree', 'best': None, 'range': list(feasible['cell']),
            'metric': 'points_agree', 'best_value': None}


def param_range(scores, cells_in_range, col):
    # Low and high of one grid parameter over the cells in range
    # @param scores: DataFrame
    # @param cells_in_range: list of cell labels
    # @param col: parameter column name
    # @returns: tuple (low, high) or (None, None)
    sub = scores[scores['cell'].isin(cells_in_range)]
    if sub.empty or col not in sub.columns:
        return None, None
    vals = pd.to_numeric(sub[col], errors='coerce').dropna()
    if vals.empty:
        return None, None
    return float(vals.min()), float(vals.max())


# --------- Catalog ---------

def catalog_path(local):
    # The catalog file
    # @param local: dict from config.load_local
    # @returns: Path
    return Path(local['avachain_root']) / 'outputs' / 'calibration' / 'catalog.csv'


def append_catalog(local, row):
    # Add or replace one catalog row (keyed on path, event, run_name)
    # @param local: dict from config.load_local
    # @param row: dict
    # @returns: Path to the catalog
    p = catalog_path(local)
    p.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(p) if p.exists() else pd.DataFrame()
    if not df.empty:
        keep = ~((df['path'] == row['path']) & (df['event_date'] == row['event_date'])
                 & (df['run_name'] == row['run_name']))
        df = df[keep]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(p, index=False)
    return p


CATALOG_TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><title>Calibration catalog</title>
<style>{{ css }}</style></head>
<body>
<h1>CALIBRATION CATALOG</h1>
<p>One row per calibration run. Rows are comparable only within the same physics hash (same model, entrainment and thresholds).</p>
{% if fig_b64 %}<img src="data:image/png;base64,{{ fig_b64 }}" alt="Catalog figure">{% endif %}
<table>
<thead><tr>{% for c in cols %}<th class="l">{{ c }}</th>{% endfor %}</tr></thead>
<tbody>{% for r in rows %}<tr>{% for c in cols %}<td class="l">{{ r[c] }}</td>{% endfor %}</tr>{% endfor %}</tbody>
</table>
</body></html>
"""


def build_catalog_page(local):
    # catalog.html beside catalog.csv, with fitted value against slab
    # hardness when tau0 was fitted
    # @param local: dict from config.load_local
    # @returns: Path or None
    p = catalog_path(local)
    if not p.exists():
        return None
    df = pd.read_csv(p)
    fig_b64 = None
    sub = df[(df['fitted'] == 'friction.tau0') & df['best'].notna()] if 'fitted' in df.columns else pd.DataFrame()
    if not sub.empty and 'slab_hand_hardness' in sub.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        for fam, g in sub.groupby('physics_hash'):
            x = pd.to_numeric(g['slab_hand_hardness'], errors='coerce')
            y = pd.to_numeric(g['best'], errors='coerce')
            lo = pd.to_numeric(g['low'], errors='coerce')
            hi = pd.to_numeric(g['high'], errors='coerce')
            ax.errorbar(x, y, yerr=[(y - lo).fillna(0), (hi - y).fillna(0)],
                        fmt='o', capsize=3, label=f"physics {fam}")
            for xi, yi, lab in zip(x, y, g['event_date']):
                ax.annotate(str(lab), (xi, yi), xytext=(4, 4), textcoords='offset points', fontsize=7)
        ax.set_xlabel('Slab hand hardness (SNOWPACK index: -1 F ... -5 K)', fontsize=9)
        ax.set_ylabel('Fitted tau0 (Pa) with range', fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(linewidth=0.4, color='#DDDDDD')
        ax.legend(fontsize=8, frameon=False)
        fig_b64 = fig_to_b64(fig)
    cols = [c for c in df.columns if c not in ('resolved_physics',)]
    html = Template(CATALOG_TEMPLATE).render(
        css=CSS, cols=cols, rows=df[cols].fillna('').to_dict(orient='records'), fig_b64=fig_b64)
    out = p.with_suffix('.html')
    out.write_text(html)
    return out


# --------- Page ---------

CAL_TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>{{ project }} calibration {{ event_date }} {{ run_name }}</title>
<style>{{ css }}</style></head>
<body>
<h1>CALIBRATION: {{ project|upper }}, EVENT {{ event_date }}</h1>
<div class="hdr"><table>
<tr><td>Run</td><td>{{ run_name }}, {{ issued }}</td></tr>
<tr><td>Observed</td><td>{{ obs_types }}{% if source %} ({{ source }}){% endif %}</td></tr>
<tr><td>Slab</td><td>{{ slab }}</td></tr>
<tr><td>Scenario</td><td>{{ scenario_desc }}</td></tr>
<tr><td>Grid</td><td>{{ grid_desc }} ({{ n_cells }} cells)</td></tr>
<tr><td>Physics</td><td>{{ physics_name }} [{{ physics_hash }}], entrainment {{ ent_mode }}</td></tr>
</table></div>

{% if failed %}<div class="warn">Cells not run: {{ failed }}</div>{% endif %}

<h2>Recommended fit at reach threshold {{ threshold }} m</h2>
<table style="width:auto">
<tr><td class="l">Rule</td><td class="l">{{ rec.rule }}</td></tr>
<tr><td class="l">Best cell</td><td class="l">{{ rec.best or 'none (range only)' }}{% if rec.best_value is not none %} ({{ rec.metric }} = {{ "%.3f"|format(rec.best_value) }}){% endif %}</td></tr>
{% for p in param_ranges %}<tr><td class="l">{{ p.name }}</td><td class="l">best {{ p.best }}, range {{ p.low }} to {{ p.high }} ({{ p.n }} of {{ n_cells }} cells within tolerance)</td></tr>{% endfor %}
</table>
<p>The operational physics file is not changed by this run. Catalog row written for path {{ slug }}, event {{ event_date }}, run {{ run_name }}.</p>

{% if fig_b64 %}<h2>Scores across the grid</h2><img src="data:image/png;base64,{{ fig_b64 }}" alt="Grid scores">{% endif %}

<h2>All cells, all thresholds</h2>
<table>
<thead><tr>{% for c in cols %}<th class="l">{{ c }}</th>{% endfor %}</tr></thead>
<tbody>{% for r in rows %}<tr>{% for c in cols %}<td class="l">{{ r[c] }}</td>{% endfor %}</tr>{% endfor %}</tbody>
</table>

<div class="meta">
IoU: overlap of modeled reach and debris over their union. Recovery: share of the debris the model covered. Over-prediction: share of the modeled reach outside the debris. Toe error: modeled farthest reach minus observed toe, both as straight-line distance from the release centroid; positive runs long. Points agree: share of reference points whose modeled reach matches the observation.
Pipeline v{{ version }}. <a href="scores.csv">scores.csv</a>, <a href="../../catalog.html">catalog</a>.
</div>
</body></html>
"""


def plot_grid(scores, pcols, thresholds):
    # Score against the grid parameter, one line per threshold, when the
    # grid varies one parameter; nothing otherwise
    # @param scores: DataFrame
    # @param pcols: list of parameter columns
    # @param thresholds: list
    # @returns: str base64 PNG or None
    if len(pcols) != 1:
        return None
    col = pcols[0]
    metrics = [m for m in ('iou', 'toe_error_m', 'points_agree') if m in scores.columns]
    if not metrics:
        return None
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.8 * len(metrics), 3.6), squeeze=False)
    for ax, m in zip(axes[0], metrics):
        for i, t in enumerate(thresholds):
            g = scores[scores['threshold_m'] == t].sort_values(col)
            ax.plot(pd.to_numeric(g[col], errors='coerce'), g[m], marker='o', ms=3,
                    color='black', linestyle=['-', '--', '-.', ':'][i % 4],
                    label=f"reach >= {t:g} m")
        if m == 'toe_error_m':
            ax.axhline(0, color='#E87722', linewidth=1)
        ax.set_xlabel(col, fontsize=9)
        ax.set_ylabel(m, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(linewidth=0.4, color='#DDDDDD')
        ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    return fig_to_b64(fig)


# --------- Runner ---------

def run_one_event(args, profile, sections, physics, local, config_dir,
                  run_pipeline, write_run_config, slug, event_date, path_profile):
    # Calibrate against one event
    # @returns: str status ('ok', 'partial', 'failed', 'dry-run')
    calib = profile.get('calibrate') or {}
    run_name = args.run_name
    issued = datetime.datetime.now()
    thresholds = [float(t) for t in calib.get('thresholds') or
                  [sections['reach']['flow_depth_threshold_m']]]
    judge_t = float(calib.get('judge_threshold') or sections['reach']['flow_depth_threshold_m'])
    if judge_t not in thresholds:
        thresholds.append(judge_t)
    tolerance = {**DEFAULT_TOLERANCE, **(calib.get('tolerance') or {})}

    try:
        import avalanche_helpers as h
        crs = h.get_crs_of(path_profile['terrain']['dem'])
        path_profile['_crs'] = crs
        event = events.load_event(local['avachain_root'], slug, event_date, crs)
        cells = expand_grid(profile.get('grid'))
    except (events.EventError, prof.ProfileError) as e:
        logging.error(str(e))
        return 'failed'

    tag = f'calibrate_{event_date}_{run_name}'
    work_root = Path(local['avachain_root']) / 'intermediate' / 'runout' / slug / tag
    out_root = Path(local['avachain_root']) / 'outputs' / 'calibration' / slug / event_date / run_name
    inputs_dir = work_root / 'inputs'

    logging.info("")
    logging.info("#" * 70)
    logging.info(f" Calibrate {slug} event {event_date}  run {run_name}  "
                 f"{len(cells)} cell(s), observed: {', '.join(event['obs_types'])}")
    logging.info("#" * 70)

    try:
        if not args.post_only:
            sid, scenario_desc = stage_scenario(calib, event, local, slug, path_profile,
                                                inputs_dir, args)
        else:
            sid = next(d.name for d in inputs_dir.iterdir() if d.name.startswith('scenario_'))
            scenario_desc = f"{sid} (existing)"
    except (cfgmod.ConfigError, StopIteration) as e:
        logging.error(str(e))
        return 'failed'
    logging.info(f"scenario: {scenario_desc}")

    manifest = mf.build_manifest(
        profile=profile, sections=sections, physics=physics, path_profile=path_profile,
        local=local, cycle_date=event_date, issued=issued, source_dir=None,
        snow_height=None, survey_date=None, run_name=run_name,
        repo_dir=Path(__file__).resolve().parent.parent)
    manifest['calibrate'] = {
        'event_date': event_date, 'event_record': str(event['folder'] / 'event.yaml'),
        'obs_types': event['obs_types'], 'slab': event['slab'],
        'scenario_id': sid, 'scenario_desc': scenario_desc,
        'grid': profile.get('grid'), 'thresholds': thresholds,
        'judge_threshold': judge_t, 'tolerance': tolerance,
    }
    manifest['cells'] = {}
    manifest_path = out_root / 'manifest.json'
    mf.write(manifest, manifest_path)

    results = {}
    for label, overrides in cells:
        logging.info("")
        logging.info("=" * 70)
        logging.info(f" Cell {label}")
        logging.info("=" * 70)
        work_dir = work_root / label
        outputs_dir = out_root / label
        cell_sections = prof.merge_sections(sections, overrides)
        # Any fitted friction parameter is a fixed value in that cell
        for key in (overrides.get('friction') or {}):
            cell_sections['friction'].setdefault('sources', {})[key] = 'fixed'
        try:
            cfg = cfgmod.assemble(cell_sections, path_profile, scenarios_dir=inputs_dir,
                                  outputs_dir=outputs_dir, work_dir=work_dir,
                                  snow_height=args.snow_height)
        except cfgmod.ConfigError as e:
            logging.error(f"[{label}] {e}")
            results[label] = 'FAILED (config)'
            continue
        cfg['probability']['weight_by_scenario_probability'] = False
        work_dir.mkdir(parents=True, exist_ok=True)
        rc = run_pipeline(write_run_config(cfg, outputs_dir), args, [])
        results[label] = ('dry-run' if args.dry_run else 'ok') if rc == 0 else f'FAILED (exit {rc})'
        manifest['cells'][label] = {'overrides': overrides, 'work_dir': str(work_dir),
                                    'outputs_dir': str(outputs_dir), 'status': results[label]}
        mf.write(manifest, manifest_path)

    ok = [l for l, s in results.items() if s == 'ok']
    failed = [l for l, s in results.items() if s.startswith('FAILED')]
    if args.dry_run:
        status = 'dry-run' if not failed else 'failed'
        mf.finalize(manifest, status)
        mf.write(manifest, manifest_path)
        return status
    if not ok:
        mf.finalize(manifest, 'failed')
        mf.write(manifest, manifest_path)
        return 'failed'

    # Scoring
    pcols = grid_columns(cells)
    rows = []
    for label, overrides in cells:
        if label not in ok:
            continue
        sdir = work_root / label / sid / 'outputs'
        pft = sdir / 'peak_flow_depth.tif'
        ppr = sdir / 'peak_pressure.tif'
        if not pft.exists():
            logging.warning(f"[{label}] no peak_flow_depth.tif; not scored")
            continue
        rel = next(p for p in (inputs_dir / sid / 'release' / 'release.geojson',
                               inputs_dir / sid / 'inputs' / 'release.geojson')
                   if p.exists())
        centroid = gpd.read_file(rel).geometry.iloc[0].centroid
        rp = path_profile['infrastructure'].get('reference_points')
        for sc in scoring.score_scenario(
                pft, ppr if ppr.exists() else None, event, centroid, thresholds,
                refpoints_path=rp if rp and Path(rp).exists() else None,
                radius_m=float((path_profile.get('reference_points') or {}).get('radius_m', 10.0)),
                road_path=path_profile['infrastructure']['road'],
                road_buffer_m=float((path_profile.get('road') or {}).get('buffer_m', 1.0)),
                pressure_threshold_kpa=float(sections['reach'].get('pressure_threshold_kpa') or 0)):
            row = {'cell': label}
            for section, kv in overrides.items():
                for k, v in kv.items():
                    row[f"{section}.{k}"] = v
            detail = sc.pop('points_detail', None)
            row.update(sc)
            if detail:
                row['points_mismatch'] = ', '.join(d['ref_name'] for d in detail if not d['agree']) or '-'
            if 'toe_error_m' in row:
                row['abs_toe'] = abs(row['toe_error_m'])
            rows.append(row)
    scores = pd.DataFrame(rows)
    out_root.mkdir(parents=True, exist_ok=True)
    scores.drop(columns=['abs_toe'], errors='ignore').to_csv(out_root / 'scores.csv', index=False)

    rec = recommend(scores, event['obs_types'], tolerance, judge_t)
    param_ranges = []
    for col in pcols:
        lo, hi = param_range(scores, rec['range'], col)
        best_row = scores[(scores['cell'] == rec['best']) & (scores['threshold_m'] == judge_t)]
        param_ranges.append({'name': col,
                             'best': best_row[col].iloc[0] if not best_row.empty else None,
                             'low': lo, 'high': hi, 'n': len(set(rec['range']))})

    # Catalog row: one row per run; the first grid parameter is the fitted one
    best_scores = scores[(scores['cell'] == rec['best']) & (scores['threshold_m'] == judge_t)]
    slab = event['slab'] or {}
    row = {
        'path': slug, 'event_date': event_date, 'run_name': run_name,
        'issued': issued.isoformat(timespec='seconds'),
        'obs_types': '+'.join(event['obs_types']),
        'scenario': scenario_desc,
        'fitted': pcols[0] if pcols else None,
        'grid': json.dumps(profile.get('grid')),
        'rule': rec['rule'],
        'best': param_ranges[0]['best'] if param_ranges else None,
        'low': param_ranges[0]['low'] if param_ranges else None,
        'high': param_ranges[0]['high'] if param_ranges else None,
        'judge_threshold_m': judge_t,
        'iou': float(best_scores['iou'].iloc[0]) if 'iou' in best_scores.columns and not best_scores.empty else None,
        'toe_error_m': float(best_scores['toe_error_m'].iloc[0]) if 'toe_error_m' in best_scores.columns and not best_scores.empty else None,
        'points_agree': float(best_scores['points_agree'].iloc[0]) if 'points_agree' in best_scores.columns and not best_scores.empty else None,
        'slab_hand_hardness': slab.get('hand_hardness'),
        'slab_density_kgm3': slab.get('density_kgm3'),
        'slab_depth_m': slab.get('depth_m'),
        'physics_name': profile['physics'],
        'physics_hash': manifest['physics']['hash'],
        'entrainment_mode': (sections.get('entrainment') or {}).get('mode'),
        'path_hash': path_profile['_hash'],
        'pipeline_version': PIPELINE_VERSION,
        'run_dir': str(out_root),
    }
    cat = append_catalog(local, row)
    build_catalog_page(local)
    logging.info(f"catalog row written: {cat}")

    # Page
    fig_b64 = plot_grid(scores, pcols, thresholds)
    show_cols = [c for c in scores.columns if c not in ('abs_toe',)]
    disp = scores[show_cols].copy()
    for c in disp.columns:
        if disp[c].dtype.kind == 'f':
            disp[c] = disp[c].map(lambda v: f"{v:.3f}" if pd.notna(v) else '')
    html = Template(CAL_TEMPLATE).render(
        css=CSS, project=path_profile.get('display', {}).get('project_name', slug),
        slug=slug, event_date=event_date, run_name=run_name,
        issued=issued.strftime('%Y-%m-%d %H:%M'),
        obs_types=', '.join(event['obs_types']), source=event.get('source'),
        slab=', '.join(f"{k} {v}" for k, v in slab.items()) or 'not recorded',
        scenario_desc=scenario_desc,
        grid_desc='; '.join(f"{k}: {v}" for k, v in (profile.get('grid') or {}).items()),
        n_cells=len(cells), physics_name=profile['physics'],
        physics_hash=manifest['physics']['hash'],
        ent_mode=(sections.get('entrainment') or {}).get('mode'),
        failed=', '.join(f"{l} ({results[l]})" for l in failed),
        threshold=f"{judge_t:g}", rec=rec, param_ranges=param_ranges,
        fig_b64=fig_b64, cols=show_cols, rows=disp.to_dict(orient='records'),
        version=PIPELINE_VERSION,
    )
    (out_root / 'calibrate.html').write_text(html)
    if fig_b64:
        (out_root / 'calibrate.png').write_bytes(base64.b64decode(fig_b64))
    manifest['recommendation'] = rec
    manifest['param_ranges'] = param_ranges
    status = 'ok' if not failed else 'partial'
    mf.finalize(manifest, status)
    mf.write(manifest, manifest_path)
    logging.info(f"recommended: {rec['rule']}, best {rec['best']}, "
                 f"range {sorted(set(rec['range']))}")
    logging.info(f"wrote {out_root / 'calibrate.html'}")

    if local.get('publish_root'):
        dest = Path(local['publish_root']) / 'calibration' / slug / event_date / run_name
        dest.mkdir(parents=True, exist_ok=True)
        for f in ('calibrate.html', 'calibrate.png', 'scores.csv'):
            if (out_root / f).exists():
                shutil.copy2(out_root / f, dest / f)
        for f in ('catalog.csv', 'catalog.html'):
            src = catalog_path(local).with_name(f)
            if src.exists():
                shutil.copy2(src, Path(local['publish_root']) / 'calibration' / f)
        logging.info(f"published -> {dest}")
    return status


def run_calibrate(args, profile, sections, physics, local, config_dir,
                  run_pipeline, write_run_config):
    # Calibrate mode entry point: one event, or every event for the path
    # @returns: int exit code
    paths = args.path or profile.get('paths') or []
    if len(paths) != 1:
        logging.error("calibrate runs one path at a time; pass --path <slug>")
        return 1
    slug = paths[0]
    if not args.run_name:
        logging.error("calibrate needs --run-name (every run is a named grid)")
        return 1
    if not args.event:
        logging.error("calibrate needs --event YYYYMMDD or --event all")
        return 1
    try:
        path_profile = cfgmod.load_path_profile(slug, local['avachain_root'], config_dir)
    except cfgmod.ConfigError as e:
        logging.error(str(e))
        return 1
    ent_mode = (sections.get('entrainment') or {}).get('mode', 'auto')
    problems = cfgmod.check_path_inputs(path_profile, need_polygon=(ent_mode == 'polygon'))
    if problems:
        for p in problems:
            logging.error(f"[{slug}] {p}")
        return 1

    if args.event == 'all':
        dates = events.list_events(local['avachain_root'], slug)
        if not dates:
            logging.error(f"no event records under {events.observed_root(local['avachain_root'], slug)}")
            return 1
    else:
        dates = [args.event.replace('-', '')]

    cells = expand_grid(profile.get('grid'))
    logging.info(f"{len(dates)} event(s) x {len(cells)} cell(s) = {len(dates) * len(cells)} simulations")
    statuses = {}
    for d in dates:
        statuses[d] = run_one_event(args, profile, sections, physics, local, config_dir,
                                    run_pipeline, write_run_config, slug, d, path_profile)
    logging.info("")
    logging.info("=" * 70)
    for d, s in statuses.items():
        logging.info(f"   event {d}: {s}")
    logging.info("=" * 70)
    return 0 if all(s in ('ok', 'dry-run') for s in statuses.values()) else 1

"""
Forecast Verification

Author: Valerie Foley
Last Updated: 10/2026

Description:
    Verify mode. Reruns a past forecast's T+0 ensemble with the physics
    recorded in its manifest and with each physics file in the profile's
    bracket, then scores every member against the observed event of that
    date: P(reach road) against the outcome with the Brier term, the same
    per reference point, and envelope coverage of the debris when a
    polygon exists. Each member becomes one row in the verification log.
"""

# --------- Load Libraries --------
import datetime
import json
import logging
import shutil
from pathlib import Path

import pandas as pd
from jinja2 import Template

from . import config as cfgmod
from . import discover
from . import events
from . import manifest as mf
from . import profiles as prof
from . import scoring
from .cycle_products import CSS
from .manifest import PIPELINE_VERSION


VERIFY_TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>{{ project }} verification {{ date }} {{ run_name }}</title>
<style>{{ css }}</style></head>
<body>
<h1>VERIFICATION: {{ project|upper }}, FORECAST {{ date }}</h1>
<div class="hdr"><table>
<tr><td>Run</td><td>{{ run_name }}, {{ issued }}</td></tr>
<tr><td>Forecast issued</td><td>{{ fc_issued }} with physics {{ fc_physics }} [{{ fc_hash }}]</td></tr>
<tr><td>Event</td><td>{{ event_date }}: {{ obs_types }}{% if source %} ({{ source }}){% endif %}</td></tr>
<tr><td>Observed at road</td><td>{{ road_obs }}</td></tr>
<tr><td>Members</td><td>{{ members }}</td></tr>
</table></div>

<h2>Reach to road</h2>
<table>
<thead><tr><th class="l">Member</th><th class="l">Physics</th><th>Scenarios</th><th>P(reach road) weighted</th><th>Observed</th><th>Brier (p - o)^2</th>{% if has_env %}<th>Debris in envelope</th><th>Mean P over debris</th>{% endif %}</tr></thead>
<tbody>{% for r in rows %}<tr><td class="l">{{ r.member }}</td><td class="l">{{ r.physics }} [{{ r.hash }}]</td><td>{{ r.n }}</td><td>{{ r.p }}</td><td class="c">{{ r.o }}</td><td>{{ r.brier }}</td>{% if has_env %}<td>{{ r.env }}</td><td>{{ r.meanp }}</td>{% endif %}</tr>{% endfor %}</tbody>
</table>

{% if point_rows %}
<h2>Reference points, forecast P(reach) against observed</h2>
<table>
<thead><tr><th class="l">Point</th><th>Observed</th>{% for m in member_names %}<th>{{ m }}</th>{% endfor %}</tr></thead>
<tbody>{% for r in point_rows %}<tr><td class="l">{{ r.name }}</td><td class="c">{{ r.o }}</td>{% for v in r.cells %}<td>{{ v }}</td>{% endfor %}</tr>{% endfor %}</tbody>
</table>
{% endif %}

<div class="meta">
The forecast member reruns T+0 with the physics recorded in the forecast's manifest, so it reproduces what was issued. Bracket members rerun the same scenarios with other physics files. A Brier term near 0 means the issued probability matched the outcome; 1 is the worst case.
Pipeline v{{ version }}. <a href="scores.csv">scores.csv</a>, <a href="../../verification.csv">verification log</a>.
</div>
</body></html>
"""


def run_verify(args, profile, sections, physics, local, config_dir,
               run_pipeline, write_run_config):
    # Verify mode entry point
    # @returns: int exit code
    paths = args.path or profile.get('paths') or []
    if len(paths) != 1:
        logging.error("verify runs one path at a time; pass --path <slug>")
        return 1
    slug = paths[0]
    if not args.date or not args.run_name:
        logging.error("verify needs --date YYYYMMDD (the forecast cycle) and --run-name")
        return 1
    date = discover.normalize_date(args.date)
    event_date = (args.event or date).replace('-', '')
    run_name = args.run_name
    issued = datetime.datetime.now()

    try:
        path_profile = cfgmod.load_path_profile(slug, local['avachain_root'], config_dir)
    except cfgmod.ConfigError as e:
        logging.error(str(e))
        return 1

    fc_dir = discover.forecast_outputs_dir(local['avachain_root'], slug, date)
    fc_manifest_path = fc_dir / 'manifest.json'
    if not fc_manifest_path.exists():
        logging.error(f"no forecast manifest at {fc_manifest_path}; nothing to verify")
        return 1
    fc = json.loads(fc_manifest_path.read_text())

    import avalanche_helpers as h
    crs = h.get_crs_of(path_profile['terrain']['dem'])
    try:
        event = events.load_event(local['avachain_root'], slug, event_date, crs)
    except events.EventError as e:
        logging.error(str(e))
        return 1

    # Observed outcome at the road: the record's road point, or the debris
    # polygon against the road corridor
    road_obs = event['points'].get('road')
    if road_obs is None and event['debris'] is not None:
        import geopandas as gpd
        road = gpd.read_file(path_profile['infrastructure']['road']).geometry.unary_union
        buf = float((path_profile.get('road') or {}).get('buffer_m', 1.0))
        road_obs = bool(event['debris'].intersects(road.buffer(buf)))
    if road_obs is None:
        logging.error("event record has no road outcome (points.road) and no debris "
                      "polygon; cannot score reach to road")
        return 1

    # Members: the forecast's own physics (from its manifest) plus the bracket
    members = [('forecast', fc['physics']['name'], fc['physics']['resolved'])]
    for name in profile.get('bracket') or []:
        try:
            ph = cfgmod.load_physics(name, config_dir)
        except cfgmod.ConfigError as e:
            logging.error(str(e))
            return 1
        members.append((f'bracket_{name}', name, {s: ph.get(s) for s in prof.PROFILE_SECTIONS
                                                   if s not in prof.NON_RESULT_SECTIONS}))

    # T+0 scenarios, copied once
    src = discover.cycle_source_dir(local['scenario_root'], path_profile['scenario_slug'], date)
    horizons = {label: d for d, label in discover.discover_horizons(src)}
    if 'T_0' not in horizons:
        logging.error(f"no T_0 scenarios under {src}")
        return 1
    tag = f'verify_{date}_{run_name}'
    work_root = Path(local['avachain_root']) / 'intermediate' / 'runout' / slug / tag
    out_root = Path(local['avachain_root']) / 'outputs' / 'verify' / slug / date / run_name
    inputs_dir = work_root / 'inputs'
    if not args.post_only:
        n = discover.copy_horizon(horizons['T_0'], inputs_dir)
        if n == 0:
            logging.error("no scenarios copied")
            return 1

    # Snow height as the forecast used it
    snow_height = None
    sf = (fc.get('survey') or {}).get('file') or {}
    if sf.get('path') and Path(sf['path']).exists():
        snow_height = Path(sf['path'])

    logging.info("")
    logging.info("#" * 70)
    logging.info(f" Verify {slug} forecast {date} against event {event_date}  "
                 f"{len(members)} member(s)")
    logging.info("#" * 70)

    manifest = mf.build_manifest(
        profile=profile, sections=sections, physics=physics, path_profile=path_profile,
        local=local, cycle_date=date, issued=issued, source_dir=horizons['T_0'],
        snow_height=snow_height, survey_date=(fc.get('survey') or {}).get('date'),
        run_name=run_name, repo_dir=Path(__file__).resolve().parent.parent)
    manifest['verify'] = {'forecast_manifest': str(fc_manifest_path),
                          'forecast_issued': fc.get('issued'),
                          'event_date': event_date, 'obs_types': event['obs_types'],
                          'road_observed': road_obs,
                          'members': [{'member': m, 'physics': p} for m, p, _ in members]}
    manifest['members'] = {}
    manifest_path = out_root / 'manifest.json'
    mf.write(manifest, manifest_path)

    results = {}
    for member, pname, resolved in members:
        logging.info("")
        logging.info("=" * 70)
        logging.info(f" Member {member}  physics {pname}")
        logging.info("=" * 70)
        msec = prof.merge_sections(sections, {s: v for s, v in resolved.items() if v})
        # The forecast's entrainment mode wins for the forecast member
        if member == 'forecast':
            msec['entrainment']['mode'] = fc['profile'].get('entrainment_mode', 'auto')
        work_dir = work_root / member
        outputs_dir = out_root / member
        try:
            cfg = cfgmod.assemble(msec, path_profile, scenarios_dir=inputs_dir,
                                  outputs_dir=outputs_dir, work_dir=work_dir,
                                  snow_height=snow_height)
        except cfgmod.ConfigError as e:
            logging.error(f"[{member}] {e}")
            results[member] = 'FAILED (config)'
            continue
        work_dir.mkdir(parents=True, exist_ok=True)
        rc = run_pipeline(write_run_config(cfg, outputs_dir), args, [])
        results[member] = ('dry-run' if args.dry_run else 'ok') if rc == 0 else f'FAILED (exit {rc})'
        manifest['members'][member] = {
            'physics': pname, 'physics_hash': prof.content_hash(prof.canonical(msec, 'verify')),
            'work_dir': str(work_dir), 'outputs_dir': str(outputs_dir), 'status': results[member]}
        mf.write(manifest, manifest_path)

    ok = [m for m, s in results.items() if s == 'ok']
    if args.dry_run or not ok:
        status = 'dry-run' if args.dry_run and not any(s.startswith('FAILED') for s in results.values()) else 'failed'
        mf.finalize(manifest, status)
        mf.write(manifest, manifest_path)
        return 0 if status == 'dry-run' else 1

    # Scores
    rows = []
    point_table = {}
    env_level = float(sections['probability'].get('envelope_min_probability', 0.05))
    for member, pname, _ in members:
        if member not in ok:
            continue
        odir = out_root / member
        road = json.loads((odir / 'road_impact_summary.json').read_text())
        p = road.get('fraction_reached_weighted')
        p = float(p if p is not None else road.get('fraction_reached', 0.0))
        o = 1.0 if road_obs else 0.0
        row = {'member': member, 'physics': pname,
               'physics_hash': manifest['members'][member]['physics_hash'],
               'n_scenarios': road.get('n_scenarios'), 'p_reach_road': p,
               'road_observed': road_obs, 'brier': (p - o) ** 2}
        if event['debris'] is not None and (odir / 'probability.tif').exists():
            row.update(scoring.score_envelope(odir / 'probability.tif', event['debris'], env_level))
        rp_path = odir / 'reference_points_summary.json'
        if rp_path.exists() and event['points']:
            rps = json.loads(rp_path.read_text())
            for name, obs in event['points'].items():
                r = rps.get(name)
                if not r:
                    continue
                pw = r.get('fraction_reached_weighted')
                pv = float(pw if pw is not None else r.get('fraction_reached', 0.0))
                point_table.setdefault(name, {'o': obs, 'cells': {}})['cells'][member] = pv
                row[f'p_{name}'] = pv
                row[f'brier_{name}'] = (pv - (1.0 if obs else 0.0)) ** 2
        rows.append(row)
    scores = pd.DataFrame(rows)
    scores.to_csv(out_root / 'scores.csv', index=False)

    # Verification log: one row per member
    log_path = Path(local['avachain_root']) / 'outputs' / 'verify' / 'verification.csv'
    log_rows = []
    for r in rows:
        log_rows.append({'path': slug, 'forecast_date': date, 'event_date': event_date,
                         'run_name': run_name, 'issued': issued.isoformat(timespec='seconds'),
                         'member': r['member'], 'physics': r['physics'],
                         'physics_hash': r['physics_hash'], 'n_scenarios': r['n_scenarios'],
                         'p_reach_road': r['p_reach_road'], 'road_observed': road_obs,
                         'brier': r['brier'],
                         'debris_in_envelope': r.get('debris_in_envelope'),
                         'mean_p_over_debris': r.get('mean_p_over_debris'),
                         'obs_types': '+'.join(event['obs_types']),
                         'forecast_issued': fc.get('issued'),
                         'pipeline_version': PIPELINE_VERSION})
    old = pd.read_csv(log_path) if log_path.exists() else pd.DataFrame()
    if not old.empty:
        keep = ~((old['path'] == slug) & (old['forecast_date'].astype(str) == date)
                 & (old['run_name'] == run_name))
        old = old[keep]
    pd.concat([old, pd.DataFrame(log_rows)], ignore_index=True).to_csv(log_path, index=False)

    member_names = [r['member'] for r in rows]
    point_rows = [{'name': n, 'o': 'reached' if v['o'] else 'not reached',
                   'cells': [f"{v['cells'].get(m, float('nan')):.2f}" for m in member_names]}
                  for n, v in point_table.items()]
    html = Template(VERIFY_TEMPLATE).render(
        css=CSS, project=path_profile.get('display', {}).get('project_name', slug),
        date=date, run_name=run_name, issued=issued.strftime('%Y-%m-%d %H:%M'),
        fc_issued=(fc.get('issued') or '').replace('T', ' '),
        fc_physics=fc['physics']['name'], fc_hash=fc['physics']['hash'],
        event_date=event_date, obs_types=', '.join(event['obs_types']),
        source=event.get('source'),
        road_obs='reached' if road_obs else 'not reached',
        members=', '.join(f"{m} ({p})" for m, p, _ in members),
        has_env=event['debris'] is not None,
        rows=[{'member': r['member'], 'physics': r['physics'], 'hash': r['physics_hash'],
               'n': r['n_scenarios'], 'p': f"{r['p_reach_road']:.2f}",
               'o': 'reached' if road_obs else 'not reached', 'brier': f"{r['brier']:.3f}",
               'env': f"{r['debris_in_envelope']:.2f}" if r.get('debris_in_envelope') is not None else '-',
               'meanp': f"{r['mean_p_over_debris']:.2f}" if r.get('mean_p_over_debris') is not None else '-'}
              for r in rows],
        member_names=member_names, point_rows=point_rows, version=PIPELINE_VERSION,
    )
    (out_root / 'verify.html').write_text(html)
    status = 'ok' if len(ok) == len(members) else 'partial'
    mf.finalize(manifest, status)
    mf.write(manifest, manifest_path)
    logging.info(f"wrote {out_root / 'verify.html'}; log {log_path}")

    if local.get('publish_root'):
        dest = Path(local['publish_root']) / 'verify' / slug / date / run_name
        dest.mkdir(parents=True, exist_ok=True)
        for f in ('verify.html', 'scores.csv'):
            shutil.copy2(out_root / f, dest / f)
        shutil.copy2(log_path, Path(local['publish_root']) / 'verify' / 'verification.csv')
        logging.info(f"published -> {dest}")
    return 0 if status == 'ok' else 1

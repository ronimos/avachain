"""
Depth Sweep

Author: Valerie Foley
Last Updated: 10/2026

Description:
    Sweep mode. Runs the full start zone of one path at a list of uniform
    depths, once per friction class, with and without entrainment, and
    reports how far each combination runs. One cell is one friction class
    under one entrainment variant; each cell is a complete pipeline run
    with every depth as one scenario. Weights are equal and the
    probability stack is not used.
"""

# --------- Load Libraries --------
import datetime
import logging
from pathlib import Path

from . import config as cfgmod
from . import generate_scenarios as gen
from . import manifest as mf
from . import profiles as prof


def load_classes(name, config_dir, wanted='all'):
    # Read the friction class table and pick the rows to run
    # @param name: file name under config/physics/ without extension
    # @param config_dir: Path to the config directory
    # @param wanted: 'all' (every enabled row) or a list of class names
    # @returns: list of class dicts (name, hand_hardness, tau0, density)
    # @raises cfgmod.ConfigError: if the file or a requested class is missing
    path = Path(config_dir) / 'physics' / f'{name}.yaml'
    if not path.exists():
        raise cfgmod.ConfigError(f"friction classes file not found: {path}")
    data = prof.load_mapping_file(path)
    rows = data.get('classes') or []
    by_name = {r['name']: r for r in rows}
    if wanted == 'all' or not wanted:
        chosen = [r for r in rows if r.get('enabled', True)]
    else:
        missing = [w for w in wanted if w not in by_name]
        if missing:
            raise cfgmod.ConfigError(f"unknown friction class(es) {missing}; "
                                     f"available: {sorted(by_name)}")
        chosen = [by_name[w] for w in wanted]
    for r in chosen:
        for key in ('tau0', 'density'):
            if r.get(key) is None:
                raise cfgmod.ConfigError(f"class {r['name']!r} has no {key}")
    return chosen


def sweep_dirs(local, slug, name):
    # The three roots a sweep writes to
    # @param local: dict from config.load_local
    # @param slug: path slug
    # @param name: sweep name
    # @returns: tuple (scenario_dir, work_dir, outputs_dir)
    tag = f'sweep_{name}'
    return (
        Path(local['scenario_root']) / slug / tag,
        Path(local['avachain_root']) / 'intermediate' / 'runout' / slug / tag,
        Path(local['avachain_root']) / 'outputs' / 'calibration' / slug / tag,
    )


def cell_label(cls, ent_on):
    # Folder name for one class under one entrainment variant
    # @param cls: class dict
    # @param ent_on: bool
    # @returns: str
    return f"{cls['name']}_{'ent' if ent_on else 'noent'}"


def run_sweep(args, profile, sections, physics, local, config_dir,
              run_pipeline, write_run_config):
    # Sweep mode entry point
    # @param args: parsed command-line args
    # @param profile: resolved profile
    # @param sections: merged physics + profile sections
    # @param physics: dict from config.load_physics
    # @param local: dict from config.load_local
    # @param config_dir: Path
    # @param run_pipeline: cli.run_pipeline (subprocess runner)
    # @param write_run_config: cli.write_run_config
    # @returns: int exit code
    # every cell is attempted; failures are logged and the products
    #       are built from the cells that completed
    sw = profile.get('sweep') or {}
    paths = args.path or profile.get('paths') or []
    if len(paths) != 1:
        logging.error("sweep mode runs one path at a time; pass --path <slug>")
        return 1
    slug = paths[0]
    name = args.run_name or sw.get('name') or 'preseason'
    issued = datetime.datetime.now()

    try:
        path_profile = cfgmod.load_path_profile(slug, local['avachain_root'], config_dir)
        classes = load_classes(sw.get('classes_file', 'friction_classes'),
                               config_dir, sw.get('classes', 'all'))
    except cfgmod.ConfigError as e:
        logging.error(str(e))
        return 1

    depths = [float(d) for d in sw.get('depths') or []]
    if not depths:
        logging.error("sweep.depths is empty")
        return 1
    ent_depth = float(sw.get('entrainment_depth_m', 0.2))
    variants = [str(v).lower() in ('on', 'true', '1', 'yes')
                for v in (sw.get('entrainment_variants') or ['on'])]
    variants = sorted(set(variants), reverse=True)

    # Release polygon: a path profile key or a file
    release_key = sw.get('release', 'start_zone')
    release = (path_profile.get('release') or {}).get(release_key) \
        if not Path(str(release_key)).exists() else str(release_key)
    problems = cfgmod.check_path_inputs(path_profile, need_polygon=any(variants))
    if not release or not Path(release).exists():
        problems.append(f"release polygon not found: {release} (sweep.release={release_key!r})")
    if problems:
        for p in problems:
            logging.error(f"[{slug}] {p}")
        return 1

    scen_root, work_root, out_root = sweep_dirs(local, slug, name)
    n_cells = len(classes) * len(variants)
    logging.info("")
    logging.info("#" * 70)
    logging.info(f" Sweep {name}  path {slug}  {len(classes)} class(es) x "
                 f"{len(variants)} entrainment variant(s) x {len(depths)} depths "
                 f"= {n_cells * len(depths)} simulations")
    logging.info("#" * 70)
    physics_hash = prof.content_hash(prof.canonical(sections, profile['mode']))
    logging.info(prof.describe(profile, physics_hash, path_profile['_hash']))

    manifest = mf.build_manifest(
        profile=profile, sections=sections, physics=physics,
        path_profile=path_profile, local=local,
        cycle_date=issued.strftime('%Y%m%d'), issued=issued,
        source_dir=None, snow_height=None, survey_date=None,
        run_name=name, repo_dir=Path(__file__).resolve().parent.parent,
    )
    manifest['sweep'] = {
        'name': name, 'release': str(release), 'depths': depths,
        'entrainment_depth_m': ent_depth,
        'entrainment_variants': ['on' if v else 'off' for v in variants],
        'classes': classes,
        'classes_file': sw.get('classes_file', 'friction_classes'),
    }
    manifest['cells'] = {}
    manifest_path = out_root / 'manifest.json'
    mf.write(manifest, manifest_path)

    results = {}
    for cls in classes:
        # One scenario set per class (density is a class property), shared
        # by both entrainment variants
        scen_dir = scen_root / cls['name']
        if not args.post_only:
            try:
                gen.generate_scenarios(
                    out_dir=scen_dir, release_path=release,
                    dem_path=path_profile['terrain']['dem'],
                    depths=depths, densities=[float(cls['density'])],
                    scour_depth_m=ent_depth, release_label=release_key,
                )
            except Exception as e:
                logging.error(f"[{cls['name']}] scenario generation failed: {e}",
                              exc_info=True)
                for ent_on in variants:
                    results[cell_label(cls, ent_on)] = 'FAILED (generate)'
                continue

        for ent_on in variants:
            label = cell_label(cls, ent_on)
            logging.info("")
            logging.info("=" * 70)
            logging.info(f" Cell {label}  tau0={cls['tau0']} Pa  rho={cls['density']} kg/m3")
            logging.info("=" * 70)
            work_dir = work_root / label
            outputs_dir = out_root / label

            # Cell physics: class tau0 and density; entrainment on or off
            cell_sections = prof.merge_sections(sections, {
                'friction': {'tau0': float(cls['tau0']),
                             'sources': {'tau0': 'fixed'}},
                'defaults': {'scour_depth_fallback_m': ent_depth},
            })
            if not ent_on:
                st = str(cell_sections['com1dfa']['sim_type'])
                cell_sections['com1dfa']['sim_type'] = 'res' if 'res' in st else 'null'
                cell_sections['entrainment']['mode'] = 'auto'
            # Inputs are generated into a folder the pipeline reads
            # straight from; work and inputs are separate here because
            # the generated set is shared by both variants
            try:
                cfg = cfgmod.assemble(
                    cell_sections, path_profile,
                    scenarios_dir=scen_dir, outputs_dir=outputs_dir,
                    work_dir=work_dir, snow_height=None,
                )
            except cfgmod.ConfigError as e:
                logging.error(f"[{label}] {e}")
                results[label] = 'FAILED (config)'
                continue
            cfg['probability']['weight_by_scenario_probability'] = False
            work_dir.mkdir(parents=True, exist_ok=True)
            run_config_path = write_run_config(cfg, outputs_dir)
            rc = run_pipeline(run_config_path, args, [])
            if args.dry_run:
                results[label] = 'dry-run' if rc == 0 else f'FAILED (exit {rc})'
            else:
                results[label] = 'ok' if rc == 0 else f'FAILED (exit {rc})'
            manifest['cells'][label] = {
                'class': cls['name'], 'tau0': cls['tau0'], 'density': cls['density'],
                'entrainment': 'on' if ent_on else 'off',
                'scenarios_dir': str(scen_dir), 'work_dir': str(work_dir),
                'outputs_dir': str(outputs_dir), 'status': results[label],
            }
            mf.write(manifest, manifest_path)

    logging.info("")
    logging.info("=" * 70)
    logging.info(f" Sweep {name}  {slug}")
    for label, status in results.items():
        logging.info(f"   {label}: {status}")
    logging.info("=" * 70)

    ok = [s for s in results.values() if s == 'ok']
    failed = [s for s in results.values() if s.startswith('FAILED')]
    if args.dry_run and not failed:
        status = 'dry-run'
    elif ok and not failed:
        status = 'ok'
    elif ok:
        status = 'partial'
    else:
        status = 'failed'
    mf.finalize(manifest, status)
    mf.write(manifest, manifest_path)

    if status in ('ok', 'partial') and 'report' in (profile.get('products') or []):
        try:
            from . import sweep_products
            sweep_products.build_sweep_products(out_root)
            if local.get('publish_root'):
                sweep_products.publish(out_root, local['publish_root'], slug, name)
        except Exception as e:
            logging.error(f"sweep products failed: {e}", exc_info=True)
            status = 'partial'
    logging.info(f"manifest: {manifest_path}")
    return 0 if status in ('ok', 'dry-run') else 1

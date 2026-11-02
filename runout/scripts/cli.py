"""
Command Line

Author: Valerie Foley
Last Updated: 10/2026

Description:
    Entry point for the runout side (python -m scripts). The run command
    resolves a profile, loads the path and physics layers, and dispatches
    to the mode runner (forecast, sweep, calibrate, verify). Forecast mode
    copies each horizon into the work tree, writes a run_config.yaml, and
    invokes avalanche_runout.py in a subprocess. A failed horizon is
    logged and the others continue. The products command rebuilds cycle
    or sweep products from disk; generate-scenarios writes synthetic
    scenario folders. Every run writes a log under logs/runout/.
"""

# --------- Load Libraries --------
import argparse
import datetime
import logging
import subprocess
import sys
from pathlib import Path

import yaml

from . import config as cfgmod
from . import discover
from . import manifest as mf
from . import profiles as prof


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_DIR = PACKAGE_DIR.parent

# Open run log (set by open_run_log); subprocess output is copied into it
LOG_FILE = None


def open_run_log(avachain_root, profile_name, run_name):
    # Start writing the log to <avachain_root>/logs/runout/ as well
    # @param avachain_root: Path
    # @param profile_name: str
    # @param run_name: str
    # @returns: Path to the log file, or None if the folder cannot be made
    global LOG_FILE
    try:
        log_dir = Path(avachain_root) / 'logs' / 'runout'
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        path = log_dir / f"{stamp}_{profile_name}_{run_name}.log"
        LOG_FILE = open(path, 'a')
        handler = logging.StreamHandler(LOG_FILE)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                               datefmt='%H:%M:%S'))
        logging.getLogger().addHandler(handler)
        logging.info(f"log: {path}")
        return path
    except OSError as e:
        logging.warning(f"could not open a run log under {avachain_root}/logs: {e}")
        return None


def parse_args(argv=None):
    # Parse command-line arguments
    # @param argv: list of strings, or None for sys.argv
    # @returns: argparse.Namespace
    parser = argparse.ArgumentParser(
        prog="avachain",
        description='Runout side of the avachain model chain'
    )
    sub = parser.add_subparsers(dest='command', required=True)

    run = sub.add_parser('run', help='Run a profile for one forecast cycle')
    run.add_argument('--profile', required=True,
                     help='Profile name from config/profiles.yaml')
    run.add_argument('--profile-file', default=None,
                     help='Extra profile file (adds or replaces named profiles)')
    run.add_argument('--set', action='append', default=[], metavar='SECTION.KEY=VALUE',
                     help='Override one config value (needs --run-name)')
    run.add_argument('--run-name', default=None,
                     help='Name for this run; required with --set')
    run.add_argument('--date', default=None,
                     help='Forecast cycle date YYYYMMDD (default: today)')
    run.add_argument('--path', action='append', default=None,
                     help='Run only this path slug (repeatable)')
    run.add_argument('--horizon', action='append', default=None,
                     help='Run only this horizon, e.g. T_0 (repeatable)')
    run.add_argument('--snow-height', default=None,
                     help='Snow height raster for entrainment.mode raster: a file, '
                          'or latest (newest file in data/surveys/snow_height/<slug>/)')
    run.add_argument('--survey-date', default=None,
                     help='Survey date YYYY-MM-DD (default: parsed from the file name)')
    run.add_argument('--event', default=None,
                     help='calibrate/verify: observed event date YYYYMMDD, or all (calibrate)')
    run.add_argument('--scenario', default=None,
                     help='calibrate: ensemble member id, or synthetic (overrides the profile)')
    run.add_argument('--scenario-date', default=None,
                     help='calibrate: cycle the member comes from (default: event date)')
    run.add_argument('--no-copy', action='store_true',
                     help='Reuse scenarios already in intermediate/runout')
    run.add_argument('--dry-run', action='store_true',
                     help='Copy and validate only; no simulations')
    run.add_argument('--post-only', action='store_true',
                     help='Skip simulations; redo post-processing')
    run.add_argument('--no-parallel', action='store_true')
    run.add_argument('--no-log', action='store_true',
                     help='Do not write a run log under <avachain_root>/logs/runout/')
    run.add_argument('--config-dir', default=str(cfgmod.CONFIG_DIR),
                     help=argparse.SUPPRESS)
    run.add_argument('--verbose', '-v', action='store_true')

    prod = sub.add_parser('products',
                          help='Rebuild the cycle report, details, workbook and viewer')
    prod.add_argument('--path', required=True, help='Path slug')
    prod.add_argument('--date', default=None, help='Forecast cycle date YYYYMMDD')
    prod.add_argument('--sweep', default=None, metavar='NAME',
                      help='Rebuild a sweep\'s products instead of a forecast cycle')
    prod.add_argument('--products', default='report,viewer',
                      help='Comma-separated: report, viewer (default both)')
    prod.add_argument('--no-publish', action='store_true',
                      help='Rebuild locally only; do not copy to publish_root')
    prod.add_argument('--config-dir', default=str(cfgmod.CONFIG_DIR),
                      help=argparse.SUPPRESS)
    prod.add_argument('--verbose', '-v', action='store_true')

    gen = sub.add_parser('generate-scenarios',
                         help='Write synthetic scenario folders in the snowpack chain\'s layout')
    gen.add_argument('--path', required=True, help='Path slug')
    gen.add_argument('--out', required=True,
                     help='Folder to write scenario_### folders into')
    gen.add_argument('--depths', required=True,
                     help='Comma-separated uniform depths (m, slope-normal)')
    gen.add_argument('--densities', default='350',
                     help='Comma-separated densities (kg/m3)')
    gen.add_argument('--release', default=None,
                     help='Release polygon file (default: path profile start_zone)')
    gen.add_argument('--scour-depth', type=float, default=0.0,
                     help='scour_depth_m written to params.json (0 = use fallback)')
    gen.add_argument('--config-dir', default=str(cfgmod.CONFIG_DIR),
                     help=argparse.SUPPRESS)
    gen.add_argument('--verbose', '-v', action='store_true')

    return parser.parse_args(argv)


def setup_logging(verbose):
    # Configure logging
    # @param verbose: bool
    # @returns: None
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )


def write_run_config(cfg, outputs_dir):
    # Write the per-horizon config the pipeline runs with
    # @param cfg: dict from config.assemble
    # @param outputs_dir: Path to the horizon's outputs folder
    # @returns: Path to run_config.yaml
    outputs_dir = Path(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    run_config_path = outputs_dir / 'run_config.yaml'
    with open(run_config_path, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return run_config_path


def run_pipeline(run_config_path, args, products):
    # Invoke avalanche_runout.py for one horizon
    # @param run_config_path: Path to the horizon's run_config.yaml
    # @param args: parsed args (flags passed through)
    # @param products: list from the profile (report, viewer, viewer_3d)
    # @returns: int - the pipeline's exit code
    # cwd is the package folder so the flat imports in the simulation
    # modules resolve
    # Per-horizon report and Leaflet viewer are replaced by the cycle
    # products; only the 3D viewer is still built per horizon.
    cmd = [sys.executable, str(PACKAGE_DIR / 'avalanche_runout.py'),
           '--config', str(run_config_path), '--skip-report']
    if 'viewer_3d' not in products:
        cmd.append('--skip-viewers')
    if args.post_only:
        cmd.append('--post-only')
    if args.dry_run:
        cmd.append('--dry-run')
    if args.no_parallel:
        cmd.append('--no-parallel')
    if args.verbose:
        cmd.append('--verbose')
    # Stream the subprocess output to the console and to the run log
    proc = subprocess.Popen(cmd, cwd=str(PACKAGE_DIR), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        line = line.rstrip('\n')
        print(line, flush=True)
        if LOG_FILE is not None:
            LOG_FILE.write(line + '\n')
            LOG_FILE.flush()
    proc.wait()
    return proc.returncode


def survey_dir(avachain_root, slug):
    # Where a path's snow height rasters live
    # @param avachain_root: Path
    # @param slug: path slug
    # @returns: Path  (<avachain_root>/data/surveys/snow_height/<slug>/)
    return Path(avachain_root) / 'data' / 'surveys' / 'snow_height' / slug


def latest_survey(avachain_root, slug):
    # Newest snow height raster for a path, by the date in its file name
    # (260114_..., 20260114_...) and by modification time when no date
    # is found
    # @param avachain_root: Path
    # @param slug: path slug
    # @returns: Path or None
    folder = survey_dir(avachain_root, slug)
    if not folder.is_dir():
        return None
    files = [p for p in folder.iterdir()
             if p.is_file() and p.suffix.lower() in ('.tif', '.tiff', '.asc', '.npy')]
    if not files:
        return None

    def key(p):
        d = mf.survey_date_from_name(p)
        return (d or '', p.stat().st_mtime)
    return max(files, key=key).resolve()


def run_forecast(args, profile, sections, physics, local, config_dir):
    # Forecast mode: every path in the profile, every horizon present
    # @param args: parsed args
    # @param profile: resolved profile
    # @param sections: merged physics + profile sections
    # @param physics: dict from config.load_physics
    # @param local: dict from config.load_local
    # @param config_dir: Path to the config directory
    # @returns: int exit code (0 all ok, 1 any failure or nothing run)
    cycle_date = discover.normalize_date(
        args.date or datetime.date.today().strftime('%Y%m%d')
    )
    issued = datetime.datetime.now()
    run_name = prof.derive_run_name(profile, args.run_name)

    paths = args.path or profile.get('paths') or []
    if not paths:
        logging.error("no paths: set `paths:` in the profile or pass --path")
        return 1
    requested_h = args.horizon or profile.get('horizons') or 'all'

    # Snow height is a per-cycle input. A file path is checked once up
    # front; 'latest' is resolved per path inside the loop.
    snow_height = None
    survey_date = args.survey_date
    if args.snow_height and args.snow_height != 'latest':
        snow_height = Path(args.snow_height).resolve()
        if not snow_height.exists():
            logging.error(f"snow height raster not found: {snow_height}")
            return 1
        survey_date = survey_date or mf.survey_date_from_name(snow_height)
        logging.info(f"snow height raster: {snow_height} (survey {survey_date})")

    ent_mode = (sections.get('entrainment') or {}).get('mode', 'auto')
    overall = []

    for slug in paths:
        logging.info("")
        logging.info("#" * 70)
        logging.info(f" Path {slug}  cycle {cycle_date}  profile {profile['name']}")
        logging.info("#" * 70)

        try:
            path_profile = cfgmod.load_path_profile(slug, local['avachain_root'], config_dir)
        except cfgmod.ConfigError as e:
            logging.error(str(e))
            overall.append(False)
            continue
        problems = cfgmod.check_path_inputs(
            path_profile, need_polygon=(ent_mode == 'polygon')
        )
        if problems:
            for p in problems:
                logging.error(f"[{slug}] {p}")
            overall.append(False)
            continue

        physics_hash = prof.content_hash(prof.canonical(sections, profile['mode']))
        logging.info(prof.describe(profile, physics_hash, path_profile['_hash']))

        if args.snow_height == 'latest':
            snow_height = latest_survey(local['avachain_root'], slug)
            if snow_height is None:
                logging.error(f"[{slug}] no snow height raster under "
                              f"{survey_dir(local['avachain_root'], slug)}")
                overall.append(False)
                continue
            survey_date = args.survey_date or mf.survey_date_from_name(snow_height)
            logging.info(f"[{slug}] snow height raster (latest): {snow_height} "
                         f"(survey {survey_date})")

        source_date_dir = discover.cycle_source_dir(local['scenario_root'], path_profile['scenario_slug'], cycle_date)
        local_date_dir = discover.runout_dir(local['avachain_root'], slug, cycle_date)
        cycle_out = discover.forecast_outputs_dir(local['avachain_root'], slug, cycle_date)

        if args.no_copy:
            discover_root = local_date_dir
        else:
            discover_root = source_date_dir
            if not source_date_dir.exists():
                logging.error(f"[{slug}] source date folder not found: {source_date_dir}")
                overall.append(False)
                continue

        horizons = discover.discover_horizons(discover_root)
        if not horizons:
            logging.error(f"[{slug}] no horizons with scenarios under {discover_root}")
            overall.append(False)
            continue
        try:
            horizons = discover.select_horizons(horizons, requested_h)
        except ValueError as e:
            logging.error(f"[{slug}] {e}")
            overall.append(False)
            continue
        logging.info(f"[{slug}] horizons: {', '.join(label for _, label in horizons)}")

        manifest = mf.build_manifest(
            profile=profile, sections=sections, physics=physics,
            path_profile=path_profile, local=local, cycle_date=cycle_date,
            issued=issued, source_dir=None if args.no_copy else source_date_dir,
            snow_height=snow_height, survey_date=survey_date,
            run_name=run_name, repo_dir=REPO_DIR,
        )
        manifest_path = cycle_out / 'manifest.json'
        mf.write(manifest, manifest_path)

        results = {}
        for src_dir, label in horizons:
            logging.info("")
            logging.info("=" * 70)
            logging.info(f" Horizon {label}  {slug}  {cycle_date}")
            logging.info("=" * 70)

            work_dir = local_date_dir / label
            outputs_dir = cycle_out / label
            valid = mf.valid_date_for(cycle_date, label)

            # Copy step
            if args.no_copy:
                if not discover.existing_copy_ok(work_dir):
                    logging.warning(f"[{label}] --no-copy but no usable copy at "
                                    f"{work_dir}; skipping")
                    results[label] = 'skipped (no copy)'
                    mf.add_horizon(manifest, label, None, 0, valid, outputs_dir,
                                   work_dir, results[label])
                    continue
                n = sum(1 for c in work_dir.iterdir()
                        if c.is_dir() and c.name.startswith('scenario_'))
                logging.info(f"[{label}] using existing copy ({n} scenarios): {work_dir}")
            else:
                n = discover.copy_horizon(src_dir, work_dir)
                if n == 0:
                    logging.warning(f"[{label}] nothing to run, skipping")
                    results[label] = 'skipped (empty)'
                    mf.add_horizon(manifest, label, src_dir, 0, valid, outputs_dir,
                                   work_dir, results[label])
                    continue

            # Config and run
            try:
                cfg = cfgmod.assemble(
                    sections, path_profile,
                    scenarios_dir=work_dir, outputs_dir=outputs_dir,
                    work_dir=work_dir, snow_height=snow_height,
                )
            except cfgmod.ConfigError as e:
                logging.error(f"[{label}] {e}")
                results[label] = 'FAILED (config)'
                mf.add_horizon(manifest, label, src_dir, n, valid, outputs_dir,
                               work_dir, results[label])
                mf.write(manifest, manifest_path)
                continue
            run_config_path = write_run_config(cfg, outputs_dir)
            logging.info(f"[{label}] run config: {run_config_path}")

            rc = run_pipeline(run_config_path, args, profile.get('products') or [])
            if args.dry_run:
                results[label] = 'dry-run' if rc == 0 else f'FAILED (exit {rc})'
            else:
                results[label] = 'ok' if rc == 0 else f'FAILED (exit {rc})'
            mf.add_horizon(manifest, label, src_dir, n, valid, outputs_dir,
                           work_dir, results[label])
            mf.write(manifest, manifest_path)

        # Path summary
        logging.info("")
        logging.info("=" * 70)
        logging.info(f" {slug}  cycle {cycle_date}")
        for label, status in results.items():
            logging.info(f"   {label}: {status}")
        logging.info("=" * 70)

        ok = [s for s in results.values() if s in ('ok', 'dry-run')]
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
        logging.info(f"[{slug}] manifest: {manifest_path}")

        # Cycle products from whatever completed
        if status in ('ok', 'partial'):
            try:
                from . import cycle_products
                cycle_products.build_cycle_products(
                    cycle_out, products=profile.get('products') or [])
                if local.get('publish_root'):
                    cycle_products.publish(cycle_out, local['publish_root'],
                                           slug, cycle_date)
            except Exception as e:
                logging.error(f"[{slug}] cycle products failed: {e}", exc_info=True)
                status = 'partial'
        overall.append(status in ('ok', 'dry-run'))

    return 0 if overall and all(overall) else 1


def cmd_run(args):
    # The `run` subcommand
    # @param args: parsed args
    # @returns: int exit code
    config_dir = Path(args.config_dir)
    try:
        local = cfgmod.load_local(config_dir)
        profile = prof.resolve_profile(
            args.profile, catalogue_path=config_dir / 'profiles.yaml',
            profile_file=args.profile_file, overrides=args.set,
        )
        run_name = prof.derive_run_name(profile, args.run_name)
        physics = cfgmod.load_physics(profile['physics'], config_dir)
        sections = cfgmod.merge_physics_and_profile(physics, profile)
    except (prof.ProfileError, cfgmod.ConfigError) as e:
        logging.error(str(e))
        return 1
    if not args.no_log:
        open_run_log(local['avachain_root'], profile['name'], run_name)

    mode = profile['mode']
    if mode == 'forecast':
        return run_forecast(args, profile, sections, physics, local, config_dir)
    if mode == 'sweep':
        from . import sweep
        return sweep.run_sweep(args, profile, sections, physics, local,
                               config_dir, run_pipeline, write_run_config)
    if mode == 'calibrate':
        from . import calibrate
        return calibrate.run_calibrate(args, profile, sections, physics, local,
                                       config_dir, run_pipeline, write_run_config)
    if mode == 'verify':
        from . import verify
        return verify.run_verify(args, profile, sections, physics, local,
                                 config_dir, run_pipeline, write_run_config)
    logging.error(f"unknown mode {mode!r}")
    return 2


def cmd_products(args):
    # The `products` subcommand: rebuild cycle products from disk
    # @param args: parsed args
    # @returns: int exit code
    from . import cycle_products
    config_dir = Path(args.config_dir)
    try:
        local = cfgmod.load_local(config_dir)
    except cfgmod.ConfigError as e:
        logging.error(str(e))
        return 1
    if args.sweep:
        from . import sweep, sweep_products
        _, _, out_root = sweep.sweep_dirs(local, args.path, args.sweep)
        try:
            written = sweep_products.build_sweep_products(out_root)
        except (FileNotFoundError, RuntimeError) as e:
            logging.error(str(e))
            return 1
        for p in written:
            logging.info(f"  {p}")
        if local.get('publish_root') and not args.no_publish:
            sweep_products.publish(out_root, local['publish_root'], args.path, args.sweep)
        return 0
    if not args.date:
        logging.error("--date is required for forecast products (or --sweep NAME)")
        return 1
    cycle_dir = discover.forecast_outputs_dir(
        local['avachain_root'], args.path, discover.normalize_date(args.date))
    products = [p.strip() for p in args.products.split(',') if p.strip()]
    try:
        written = cycle_products.build_cycle_products(cycle_dir, products)
    except (FileNotFoundError, RuntimeError) as e:
        logging.error(str(e))
        return 1
    for p in written:
        logging.info(f"  {p}")
    if local.get('publish_root') and not args.no_publish:
        cycle_products.publish(cycle_dir, local['publish_root'], args.path,
                               discover.normalize_date(args.date))
    return 0


def cmd_generate(args):
    # The `generate-scenarios` subcommand
    # @param args: parsed args
    # @returns: int exit code
    from . import generate_scenarios as gen
    config_dir = Path(args.config_dir)
    try:
        local = cfgmod.load_local(config_dir)
        path_profile = cfgmod.load_path_profile(args.path, local['avachain_root'], config_dir)
    except cfgmod.ConfigError as e:
        logging.error(str(e))
        return 1
    release = args.release or (path_profile.get('release') or {}).get('start_zone')
    if not release or not Path(release).exists():
        logging.error(f"release polygon not found: {release}")
        return 1
    depths = [float(x) for x in args.depths.split(',') if x.strip()]
    densities = [float(x) for x in args.densities.split(',') if x.strip()]
    n = gen.generate_scenarios(
        out_dir=Path(args.out), release_path=release,
        dem_path=path_profile['terrain']['dem'],
        depths=depths, densities=densities, scour_depth_m=args.scour_depth,
    )
    logging.info(f"wrote {n} scenario folder(s) to {args.out}")
    return 0


def main(argv=None):
    # Entry point
    # @param argv: list of strings, or None for sys.argv
    # @returns: None (exits with status code)
    args = parse_args(argv)
    setup_logging(args.verbose)
    if args.command == 'run':
        sys.exit(cmd_run(args))
    if args.command == 'products':
        sys.exit(cmd_products(args))
    if args.command == 'generate-scenarios':
        sys.exit(cmd_generate(args))
    sys.exit(2)


if __name__ == '__main__':
    main()

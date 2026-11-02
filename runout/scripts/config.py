"""
Config Layers

Author: Valerie Foley
Last Updated: 10/2026

Description:
    Loads the config layers (local roots, path profile, physics file,
    resolved run profile) and assembles the single config dict the
    simulation modules consume. The result has the same shape as the
    original config.yaml so the simulation code did not change.
"""

# --------- Load Libraries --------
import copy
import logging
from pathlib import Path

from . import profiles as prof


# config/ lives beside the scripts/ package in the repo
CONFIG_DIR = Path(__file__).resolve().parent.parent / 'config'

# Keys in the path profile that hold files, grouped by section. Relative
# values resolve against avachain_root.
PATH_FILE_KEYS = {
    'terrain': ('dem', 'dem_priority'),
    'canopy': ('forest',),
    'infrastructure': ('road', 'reference_points'),
    'release': ('start_zone',),
    'entrainment': ('polygon',),
}


class ConfigError(ValueError):
    pass


def load_local(config_dir=CONFIG_DIR):
    # Read config/local.yaml (machine roots)
    # @param config_dir: Path to the config directory
    # @returns: dict with avachain_root and scenario_root as Paths
    # @raises ConfigError: if local.yaml is missing or incomplete
    path = Path(config_dir) / 'local.yaml'
    if not path.exists():
        raise ConfigError(
            f"{path} not found. Copy local.example.yaml to local.yaml and "
            f"set avachain_root and scenario_root for this machine."
        )
    data = prof.load_mapping_file(path)
    for key in ('avachain_root', 'scenario_root'):
        if not data.get(key):
            raise ConfigError(f"{path}: missing {key}")
    publish = data.get('publish_root')
    return {
        'avachain_root': Path(data['avachain_root']).expanduser(),
        'scenario_root': Path(data['scenario_root']).expanduser(),
        'publish_root': Path(publish).expanduser() if publish else None,
        '_file': str(path),
    }


def load_path_profile(slug, avachain_root, config_dir=CONFIG_DIR):
    # Read config/paths/<slug>.yaml and resolve its files to absolute paths
    # @param slug: path slug (little_professor, bethel, ...)
    # @param avachain_root: Path that relative file names resolve against
    # @param config_dir: Path to the config directory
    # @returns: dict (the yaml with absolute file paths, plus _file and _hash)
    # @raises ConfigError: if the file is missing or the slug does not match
    path = Path(config_dir) / 'paths' / f'{slug}.yaml'
    if not path.exists():
        available = sorted(p.stem for p in (Path(config_dir) / 'paths').glob('*.yaml'))
        raise ConfigError(f"no path profile for {slug!r} at {path}; "
                          f"available: {available}")
    data = prof.load_mapping_file(path)
    if data.get('slug') != slug:
        raise ConfigError(f"{path}: slug {data.get('slug')!r} does not match "
                          f"file name {slug!r}")
    for section, keys in PATH_FILE_KEYS.items():
        sec = data.get(section) or {}
        for key in keys:
            val = sec.get(key)
            if val:
                p = Path(val).expanduser()
                if not p.is_absolute():
                    p = Path(avachain_root) / p
                sec[key] = str(p.resolve())
        data[section] = sec
    data['scenario_slug'] = data.get('scenario_slug') or slug
    data['_file'] = str(path)
    data['_hash'] = prof.content_hash(
        {k: v for k, v in data.items() if not k.startswith('_')}
    )
    return data


def load_physics(name, config_dir=CONFIG_DIR):
    # Read config/physics/<name>.yaml
    # @param name: physics file name without extension
    # @param config_dir: Path to the config directory
    # @returns: dict of physics sections plus _file
    # @raises ConfigError: if the file is missing or lacks a section
    path = Path(config_dir) / 'physics' / f'{name}.yaml'
    if not path.exists():
        available = sorted(p.stem for p in (Path(config_dir) / 'physics').glob('*.yaml'))
        raise ConfigError(f"no physics file {name!r} at {path}; "
                          f"available: {available}")
    data = prof.load_mapping_file(path)
    for section in ('com1dfa', 'friction', 'defaults', 'entrainment',
                    'resistance', 'reach', 'probability'):
        if section not in data:
            raise ConfigError(f"{path}: missing section {section!r}")
    data['_file'] = str(path)
    return data


def merge_physics_and_profile(physics, profile):
    # Layer the profile's section overrides on the physics file, then
    # coerce any --set strings against the value they replace.
    # @param physics: dict from load_physics
    # @param profile: resolved profile from profiles.resolve_profile
    # @returns: dict of sections (every PROFILE_SECTIONS key present)
    # @raises ConfigError: if an override names a key the section lacks
    sections = {s: copy.deepcopy(physics.get(s) or {})
                for s in prof.PROFILE_SECTIONS}
    overlay = {s: profile.get(s) for s in prof.PROFILE_SECTIONS
               if profile.get(s)}
    merged = prof.merge_sections(sections, overlay)

    # Coerce --set strings against the physics value they replace. A key
    # the physics file lacks is a typo, except execution, output and
    # entrainment.mode, which only the profile defines.
    set_keys = set()
    for text in profile.get('_overrides') or []:
        section, key, _ = prof.parse_override(text)
        if section != 'run':
            set_keys.add((section, key))
    for section, key in set_keys:
        base_val = (physics.get(section) or {}).get(key, '__absent__')
        if base_val == '__absent__' and section not in prof.NON_RESULT_SECTIONS \
                and not (section == 'entrainment' and key == 'mode'):
            raise ConfigError(f"--set {section}.{key}: not a key in the "
                              f"physics file; check the spelling")
        like = None if base_val == '__absent__' else base_val
        merged[section][key] = prof.coerce(merged[section][key], like)

    merged['depth_convention'] = physics.get('depth_convention', 'unknown')
    return merged


def assemble(sections, path_profile, scenarios_dir, outputs_dir,
             work_dir=None, snow_height=None):
    # Build the legacy config dict for one horizon or run
    # @param sections: dict from merge_physics_and_profile
    # @param path_profile: dict from load_path_profile
    # @param scenarios_dir: Path holding scenario_### folders (inputs)
    # @param outputs_dir: Path for ensemble outputs
    # @param work_dir: Path for per-scenario work; defaults to scenarios_dir
    #                  so avaframe/ and outputs/ sit beside release/
    # @param snow_height: optional Path to the snow height raster
    # @returns: dict shaped like the old config.yaml
    # @raises ConfigError: if the entrainment mode's input is missing
    cfg = copy.deepcopy(sections)
    terrain = path_profile['terrain']
    infra = path_profile['infrastructure']

    cfg['paths'] = {
        'scenarios_dir': str(scenarios_dir),
        'outputs_dir': str(outputs_dir),
        'scenario_work_dir': str(work_dir or scenarios_dir),
        'dem': terrain['dem'],
        'forest': path_profile['canopy']['forest'],
        'road': infra['road'],
        'reference_points': infra.get('reference_points'),
        'dem_feather_cells': terrain.get('dem_feather_cells', 5),
        'dem_vertical_correction_m': terrain.get('dem_vertical_correction_m', 0.0),
    }
    if terrain.get('dem_priority'):
        cfg['paths']['dem_priority'] = terrain['dem_priority']

    cfg['road'] = dict(path_profile.get('road') or {'buffer_m': 1.0,
                                                     'chainage_origin': 'far_from_release'})
    cfg['reference_points'] = dict(path_profile.get('reference_points') or {'radius_m': 10.0})
    cfg['display'] = dict(path_profile.get('display') or {})

    # Entrainment mode wiring. The legacy modules read use_snow_height_dsm
    # and snow_height_raster; mode is the new switch and the legacy keys
    # are derived from it so both agree.
    ent = cfg.setdefault('entrainment', {})
    mode = ent.get('mode', 'auto')
    ent['mode'] = mode
    ent['use_snow_height_dsm'] = (mode == 'raster' and snow_height is not None)
    ent['snow_height_raster'] = str(snow_height) if snow_height else None
    ent['polygon'] = (path_profile.get('entrainment') or {}).get('polygon')
    if mode == 'raster' and 'ent' in cfg['com1dfa']['sim_type'] and not snow_height:
        raise ConfigError("entrainment.mode is 'raster' but no snow height "
                          "raster was given (--snow-height); use a profile "
                          "with mode auto or polygon when no survey exists")
    if mode == 'polygon' and 'ent' in cfg['com1dfa']['sim_type']:
        poly = ent['polygon']
        if not poly or not Path(poly).exists():
            raise ConfigError(f"entrainment.mode is 'polygon' but the path "
                              f"profile has no entrainment polygon on disk: {poly}")

    # com1dfa.friction_model is what the legacy modules read; keep it in
    # sync with friction.model if someone sets the newer key
    if cfg.get('friction', {}).get('model'):
        cfg['com1dfa']['friction_model'] = cfg['friction']['model']

    cfg.setdefault('execution', {'parallel': True, 'max_workers': 4,
                                 'abort_on_crs_mismatch': True})
    cfg.setdefault('output', {'cog': True, 'probability_colormap': 'viridis',
                              'report_plots_dpi': 120,
                              'viewer_basemap': 'hillshade'})
    return cfg


def check_path_inputs(path_profile, need_polygon=False, need_start_zone=False):
    # Confirm the static files a run needs are on disk
    # @param path_profile: dict from load_path_profile
    # @param need_polygon: require the entrainment polygon
    # @param need_start_zone: require the start zone polygon
    # @returns: list of problem strings (empty if fine)
    problems = []
    required = [
        ('terrain.dem', path_profile['terrain'].get('dem')),
        ('canopy.forest', path_profile['canopy'].get('forest')),
        ('infrastructure.road', path_profile['infrastructure'].get('road')),
    ]
    if path_profile['terrain'].get('dem_priority'):
        required.append(('terrain.dem_priority', path_profile['terrain']['dem_priority']))
    if need_polygon:
        required.append(('entrainment.polygon',
                         (path_profile.get('entrainment') or {}).get('polygon')))
    if need_start_zone:
        required.append(('release.start_zone',
                         (path_profile.get('release') or {}).get('start_zone')))
    for label, p in required:
        if not p:
            problems.append(f"{label} is not set in {path_profile['_file']}")
        elif not Path(p).exists():
            problems.append(f"{label} not found: {p}")
    rp = path_profile['infrastructure'].get('reference_points')
    if rp and not Path(rp).exists():
        logging.warning(f"reference_points not found, point analysis skipped: {rp}")
    return problems

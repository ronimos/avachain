"""
Run Profiles

Author: Valerie Foley
Last Updated: 10/2026

Description:
    Named run profiles as first-class config. A profile maps onto the
    physics sections (com1dfa, friction, defaults, entrainment, resistance,
    reach, probability) plus execution and output, and carries run-level
    keys (mode, physics, paths, horizons, products, sweep, calibrate,
    bracket, grid). Profiles resolve in order physics file, profile
    sections, then --set overrides, and the resolved content is hashed so
    two runs with the same numbers hash the same.
"""

# --------- Load Libraries --------
import copy
import hashlib
import json
from pathlib import Path


# Config sections a profile may address. These are the physics file
# sections plus the two run-side sections.
PROFILE_SECTIONS = ('com1dfa', 'friction', 'defaults', 'entrainment',
                    'resistance', 'reach', 'probability', 'execution',
                    'output')

# Sections that do not change model results. Left out of the hash.
NON_RESULT_SECTIONS = ('execution', 'output')

# Run-level keys (not config sections).
PROFILE_RUN_KEYS = ('description', 'mode', 'physics', 'paths', 'horizons',
                    'products', 'bracket', 'grid', 'sweep', 'calibrate',
                    'extends')

KNOWN_MODES = ('forecast', 'sweep', 'calibrate', 'verify')

# Products the run stage knows how to build.
KNOWN_PRODUCTS = ('report', 'viewer', 'viewer_3d')

KNOWN_ENTRAINMENT_MODES = ('auto', 'polygon', 'raster')


class ProfileError(ValueError):
    pass


# --------- Loading ---------

def load_mapping_file(path):
    # Read a .yaml/.yml or .json file into a dict
    # @param path: file path
    # @returns: dict (top-level mapping)
    # @raises ProfileError: if the file is missing or not a mapping
    path = Path(path)
    if not path.exists():
        raise ProfileError(f"file not found: {path}")
    text = path.read_text()
    if path.suffix.lower() in ('.yaml', '.yml'):
        import yaml
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ProfileError(f"{path}: top level must be a mapping")
    return data


def load_profiles(path):
    # Load the profile catalogue {name: profile_mapping}. A file may be a
    # catalogue or a single profile with a `name` key.
    # @param path: profiles file
    # @returns: dict of name -> raw profile mapping
    # @raises ProfileError: if a profile body is not a mapping
    data = load_mapping_file(path)
    if 'name' in data and isinstance(data.get('name'), str):
        single = dict(data)
        name = single.pop('name')
        return {name: single}
    out = {}
    for name, body in data.items():
        if not isinstance(body, dict):
            raise ProfileError(f"{path}: profile {name!r} must be a mapping")
        out[str(name)] = body
    return out


# --------- Overrides ---------

def parse_override(text):
    # Parse "section.key=value" into (section, key, raw_value)
    # @param text: one --set argument
    # @returns: tuple (section, key, raw_value)
    # @raises ProfileError: on a malformed or unknown section
    if '=' not in text:
        raise ProfileError(f"--set expects section.key=value, got {text!r}")
    lhs, raw = text.split('=', 1)
    if '.' not in lhs:
        raise ProfileError(f"--set key must be section.key, got {lhs!r}")
    section, key = lhs.split('.', 1)
    section, key, raw = section.strip(), key.strip(), raw.strip()
    if section not in PROFILE_SECTIONS and section != 'run':
        raise ProfileError(f"--set: unknown section {section!r}; expected one "
                           f"of {PROFILE_SECTIONS + ('run',)}")
    if not key:
        raise ProfileError(f"--set: empty key in {text!r}")
    return section, key, raw


def coerce(raw, like):
    # Coerce a CLI string to the type of an existing config value
    # @param raw: string from --set
    # @param like: current value whose type is the target
    # @returns: coerced value
    # @raises ProfileError: if a bool cannot be parsed
    if not isinstance(raw, str):
        return raw
    if isinstance(like, bool):
        low = raw.lower()
        if low in ('1', 'true', 'yes', 'on'):
            return True
        if low in ('0', 'false', 'no', 'off'):
            return False
        raise ProfileError(f"expected a bool, got {raw!r}")
    if isinstance(like, int) and not isinstance(like, bool):
        return int(float(raw))
    if isinstance(like, float):
        return float(raw)
    if like is None:
        if raw.lower() in ('none', 'null', ''):
            return None
        try:
            return float(raw)
        except ValueError:
            return raw
    if isinstance(like, (list, tuple)):
        parts = [p.strip() for p in raw.split(',') if p.strip()]
        if like and isinstance(like[0], (int, float)):
            parts = [float(p) for p in parts]
        return parts
    if isinstance(like, dict):
        return json.loads(raw)
    return raw


def apply_overrides(profile, overrides):
    # Apply --set overrides to a raw profile mapping. Values stay strings;
    # coercion happens against the physics values in config.assemble.
    # @param profile: raw profile mapping (copied, not mutated)
    # @param overrides: iterable of "section.key=value"
    # @returns: new mapping with the overrides recorded under _overrides
    prof = copy.deepcopy(profile)
    applied = []
    for text in overrides or ():
        section, key, raw = parse_override(text)
        if section == 'run':
            if key not in PROFILE_RUN_KEYS:
                raise ProfileError(f"--set run.{key}: unknown run key; "
                                   f"expected one of {PROFILE_RUN_KEYS}")
            prof[key] = raw
        else:
            prof.setdefault(section, {})[key] = raw
        applied.append(text)
    if applied:
        prof['_overrides'] = applied
    return prof


# --------- Resolution ---------

def merge_sections(base, top):
    # Per-section merge: top's keys win inside each section; run keys
    # replace outright. Nested dicts inside a section (auto_detect,
    # sources) merge one level deeper so a profile can override one key.
    # @param base: lower-priority mapping
    # @param top: higher-priority mapping
    # @returns: merged copy
    out = copy.deepcopy(base)
    for k, v in top.items():
        if k in PROFILE_SECTIONS and isinstance(v, dict):
            sec = out.setdefault(k, {})
            for kk, vv in v.items():
                if isinstance(vv, dict) and isinstance(sec.get(kk), dict):
                    sec[kk] = {**sec[kk], **copy.deepcopy(vv)}
                else:
                    sec[kk] = copy.deepcopy(vv)
        else:
            out[k] = copy.deepcopy(v)
    return out


def resolve_profile(name, catalogue_path, profile_file=None, overrides=None):
    # Build one resolved profile from: catalogue -> optional profile_file
    # (adds or replaces named profiles) -> extends chain -> --set overrides.
    # Validates structure, not values.
    # @param name: profile name; may be None when profile_file holds one
    #              profile with its own `name`
    # @param catalogue_path: config/profiles.yaml
    # @param profile_file: optional extra catalogue or single profile
    # @param overrides: iterable of "section.key=value" strings
    # @returns: resolved mapping with name, sections, run keys, _source
    # @raises ProfileError: on unknown names, cycles, or unknown keys
    catalogue = load_profiles(catalogue_path)
    source = str(catalogue_path)
    if profile_file:
        extra = load_profiles(profile_file)
        catalogue.update(extra)
        if name is None and len(extra) == 1:
            name = next(iter(extra))
        source = f"{source} + {profile_file}"
    if not name:
        raise ProfileError("a profile name is required (--profile NAME); "
                           f"available: {sorted(catalogue)}")
    if name not in catalogue:
        raise ProfileError(f"unknown profile {name!r}; available: "
                           f"{sorted(catalogue)}")

    # extends chain, bounded to catch cycles
    chain = []
    cur = name
    for _ in range(10):
        if cur is None:
            break
        if cur in chain:
            raise ProfileError(f"profile extends cycle: {chain + [cur]}")
        if cur not in catalogue:
            raise ProfileError(f"profile {chain[-1]!r} extends unknown {cur!r}")
        chain.append(cur)
        cur = catalogue[cur].get('extends')
    resolved = {}
    for n in reversed(chain):
        resolved = merge_sections(resolved, catalogue[n])
    resolved.pop('extends', None)

    for k in resolved:
        if k not in PROFILE_SECTIONS and k not in PROFILE_RUN_KEYS \
                and not k.startswith('_'):
            raise ProfileError(f"profile {name!r}: unknown top-level key {k!r}; "
                               f"sections are {PROFILE_SECTIONS}, run keys "
                               f"{PROFILE_RUN_KEYS}")

    resolved = apply_overrides(resolved, overrides or ())
    resolved['name'] = name
    resolved['_source'] = source
    resolved.setdefault('mode', 'forecast')
    resolved.setdefault('physics', 'operational')
    resolved.setdefault('horizons', 'all')
    resolved.setdefault('products', [])
    resolved.setdefault('bracket', [])
    resolved.setdefault('grid', {})
    resolved.setdefault('sweep', {})
    resolved.setdefault('calibrate', {})

    # Normalize list-ish run keys given as strings (from --set)
    for key in ('paths', 'products', 'bracket'):
        if isinstance(resolved.get(key), str):
            resolved[key] = [p.strip() for p in resolved[key].split(',')
                             if p.strip()]
    if isinstance(resolved.get('horizons'), str) and resolved['horizons'] != 'all':
        resolved['horizons'] = [p.strip() for p in resolved['horizons'].split(',')
                                if p.strip()]
    if isinstance(resolved.get('grid'), str):
        resolved['grid'] = json.loads(resolved['grid'])
    if isinstance(resolved.get('sweep'), str):
        resolved['sweep'] = json.loads(resolved['sweep'])
    if isinstance(resolved.get('calibrate'), str):
        resolved['calibrate'] = json.loads(resolved['calibrate'])

    if resolved['mode'] not in KNOWN_MODES:
        raise ProfileError(f"profile {name!r}: unknown mode "
                           f"{resolved['mode']!r}; known: {KNOWN_MODES}")
    for p in resolved['products']:
        if p not in KNOWN_PRODUCTS:
            raise ProfileError(f"profile {name!r}: unknown product {p!r}; "
                               f"known: {KNOWN_PRODUCTS}")
    ent_mode = (resolved.get('entrainment') or {}).get('mode')
    if ent_mode is not None and ent_mode not in KNOWN_ENTRAINMENT_MODES:
        raise ProfileError(f"profile {name!r}: entrainment.mode "
                           f"{ent_mode!r}; known: {KNOWN_ENTRAINMENT_MODES}")
    return resolved


# --------- Hashing / naming ---------

def _norm(v):
    # Normalize scalars so "18" and 18.0 hash identically
    # @param v: any value
    # @returns: normalized value
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        low = v.strip().lower()
        if low in ('true', 'false'):
            return low == 'true'
        try:
            return float(v)
        except ValueError:
            return v.strip()
    if isinstance(v, (list, tuple)):
        return [_norm(x) for x in v]
    if isinstance(v, dict):
        return {k: _norm(x) for k, x in sorted(v.items())}
    return v


def canonical(sections, mode='forecast'):
    # The subset of resolved sections that changes model results
    # @param sections: dict of section -> values (physics + profile merged)
    # @param mode: run mode
    # @returns: dict ready to hash
    body = {}
    for section in PROFILE_SECTIONS:
        if section in NON_RESULT_SECTIONS:
            continue
        body[section] = _norm(sections.get(section) or {})
    body['mode'] = mode
    return body


def content_hash(obj, length=12):
    # Short stable hash of any JSON-serializable object
    # @param obj: dict or list
    # @param length: hex chars to keep
    # @returns: hex string
    blob = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:length]


def derive_run_name(profile, explicit):
    # Run name rule: explicit wins; otherwise the profile name, but only
    # when no --set overrides were applied.
    # @param profile: resolved mapping
    # @param explicit: --run-name value or None
    # @returns: run name
    # @raises ProfileError: if overrides were applied without --run-name
    if explicit:
        return explicit
    if profile.get('_overrides'):
        raise ProfileError("--run-name is required when --set overrides are "
                           f"used (overrides: {profile['_overrides']}); a "
                           "tweaked profile cannot pose as the profile itself")
    return str(profile['name'])


def describe(profile, physics_hash, path_hash):
    # One-block human summary for the run log header
    # @param profile: resolved mapping
    # @param physics_hash: hash of the resolved physics sections
    # @param path_hash: hash of the path profile
    # @returns: str
    lines = [
        f"  Profile:      {profile['name']}  mode={profile['mode']}",
        f"  Physics:      {profile['physics']}  [{physics_hash}]",
        f"  Path hash:    [{path_hash}]",
        f"  Source:       {profile.get('_source', '')}",
    ]
    if profile.get('description'):
        lines.append(f"  Description:  {profile['description']}")
    if profile.get('_overrides'):
        lines.append(f"  Overrides:    {' '.join(profile['_overrides'])}")
    lines.append(f"  Products:     {', '.join(profile.get('products') or []) or '(none)'}")
    return '\n'.join(lines)

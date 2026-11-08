"""
Scenario Discovery

Author: Valerie Foley
Last Updated: 10/2026

Description:
    Finds the scenario sets for one forecast cycle under scenario_root and
    copies them into the runout work tree. Horizons are the T_<n> folders
    present; older layouts with a top-level scenarios folder or nested
    T_n/scenarios still resolve. Each scenario's inputs land in
    scenario_###/release/ with avaframe/ and outputs/ beside it.

    A release set published empty (manifest status "no_scenarios": no
    cluster qualified as a trigger, or every release held too little snow)
    is still a horizon. It has nothing to run, and release_info() says why,
    so the report can say why there is no forecast for that day.
"""

# --------- Load Libraries --------
import json
import logging
import re
import shutil
from pathlib import Path


HORIZON_RE = re.compile(r'^T_(\d+)$')

# avachain analysis_pipeline.MIN_RELEASE_DEPTH_M, used for the wording when
# a release set's metadata does not carry the floor itself
RELEASE_DEPTH_FLOOR_M = 0.10


def normalize_date(text):
    # Accept YYYYMMDD or YYYY-MM-DD and return YYYYMMDD
    # @param text: date string
    # @returns: str YYYYMMDD
    # @raises ValueError: if the text is not one of the two forms
    s = text.strip().replace('-', '')
    if not re.fullmatch(r'\d{8}', s):
        raise ValueError(f"date must be YYYYMMDD or YYYY-MM-DD, got {text!r}")
    return s


def cycle_source_dir(scenario_root, slug, date):
    # The dated scenario folder for one path and cycle. YYYYMMDD is the
    # standard; a YYYY-MM-DD folder is accepted when that is what exists.
    # @param scenario_root: Path from local.yaml
    # @param slug: path slug (or scenario_slug)
    # @param date: YYYYMMDD
    # @returns: Path (the standard spelling when neither exists)
    base = Path(scenario_root) / slug
    plain = base / date
    dashed = base / f"{date[:4]}-{date[4:6]}-{date[6:]}"
    if not plain.exists() and dashed.exists():
        return dashed
    return plain


def runout_dir(avachain_root, slug, date, horizon=None):
    # Work tree for one path and cycle (and horizon)
    # @param avachain_root: Path from local.yaml
    # @param slug: path slug
    # @param date: YYYYMMDD
    # @param horizon: optional horizon label
    # @returns: Path
    p = Path(avachain_root) / 'intermediate' / 'runout' / slug / date
    return p / horizon if horizon else p


def forecast_outputs_dir(avachain_root, slug, date, horizon=None):
    # Outputs folder for one path and cycle (and horizon)
    # @param avachain_root: Path from local.yaml
    # @param slug: path slug
    # @param date: YYYYMMDD
    # @param horizon: optional horizon label
    # @returns: Path
    p = Path(avachain_root) / 'outputs' / 'forecasts' / slug / date
    return p / horizon if horizon else p


def _has_scenarios(d):
    # True when a folder directly holds scenario_* folders
    # @param d: Path
    # @returns: bool
    return d.is_dir() and any(c.is_dir() and c.name.startswith('scenario_')
                              for c in d.iterdir())


def release_info(src_dir):
    # What the release side published for one horizon, from its handoff
    # manifest.json (and metadata.json for the counts)
    # @param src_dir: Path to the horizon folder, or its scenarios/ subfolder
    # @returns: dict (status, n_scenarios, trigger_source, n_trigger_locations,
    #           n_below_depth_floor, snapshot_date, reason) or None when the
    #           folder holds no release manifest
    src_dir = Path(src_dir)
    for d in (src_dir, src_dir.parent):
        try:
            man = json.loads((d / 'manifest.json').read_text())
        except (OSError, ValueError):
            continue
        if man.get('producer') != 'release':
            continue
        try:
            meta = json.loads((d / 'metadata.json').read_text())
        except (OSError, ValueError):
            meta = {}
        info = {
            'status': man.get('status'),
            'n_scenarios': man.get('n_scenarios'),
            'trigger_source': man.get('trigger_source') or meta.get('trigger_source'),
            'n_trigger_locations': meta.get('n_trigger_locations'),
            'n_below_depth_floor': meta.get('n_below_depth_floor'),
            'min_release_depth_m': meta.get('min_release_depth_m', RELEASE_DEPTH_FLOOR_M),
            'snapshot_date': man.get('snapshot_date'),
            'shot_outcomes': meta.get('shot_outcomes'),
            'max_hs_m': meta.get('max_hs_m'),
        }
        info['reason'] = no_release_reason(info) if info['status'] == 'no_scenarios' else None
        return info
    return None


def no_release_reason(info):
    # One sentence saying why a release set has no scenarios
    # @param info: dict from release_info
    # @returns: str
    src = info.get('trigger_source')
    n_low = info.get('n_below_depth_floor') or 0
    n_trig = info.get('n_trigger_locations')
    floor = f"{info.get('min_release_depth_m', RELEASE_DEPTH_FLOOR_M):.2f} m"
    snap = f" (snowpack {info['snapshot_date']})" if info.get('snapshot_date') else ''
    if src == 'no_snow':
        # Bare ground: the start zone holds less snow than any release needs
        hs = info.get('max_hs_m')
        hs_txt = f" (deepest modelled snow {hs * 100:.0f} cm)" if hs is not None else ''
        return (f"No release scenarios{snap}: no snow cover. The start zone holds "
                f"less than {floor} of snow{hs_txt}, so no avalanche can release.")
    if src == 'explosive_forced':
        # Forced explosive shots: each shot initiates; none propagated
        why = {'slab_tensile': 'the slab could not carry the crack (slab tensile support below 1)',
               'no_propagation': 'the crack did not propagate',
               'no_arrest_indices': 'no snowpack cluster had a weak layer to propagate in'}
        outcomes = info.get('shot_outcomes') or {}
        parts = []
        for cid, o in outcomes.items():
            reason = why.get(o.get('no_release_reason'))
            if o.get('n_below_depth_floor') and not reason:
                reason = f'the release held too little snow (mean depth under {floor})'
            parts.append(f"cluster {cid}: {reason or 'no release'}")
        detail = f" ({'; '.join(parts)})" if parts else ''
        return (f"No release scenarios{snap}: none of the explosive-control shots "
                f"would release an avalanche{detail}.")
    if src == 'explosive_fallback' and n_low:
        return (f"No release scenarios{snap}: no snowpack cluster met the trigger "
                f"criteria, and the {n_low} releases tried at the explosive-control "
                f"shot locations held too little snow (mean depth under {floor}).")
    if src == 'model' and n_low:
        where = f" from {n_trig} trigger location{'s' if n_trig != 1 else ''}" if n_trig else ''
        return (f"No release scenarios{snap}: all {n_low} candidate releases{where} "
                f"held too little snow (mean depth under {floor}).")
    if src is None:
        return (f"No release scenarios{snap}: no snowpack cluster met the trigger "
                f"criteria, and no explosive-control shot locations are configured.")
    return (f"No release scenarios{snap}: no release area could be built from the "
            f"trigger locations.")


def set_raster(src_dir):
    # The release set's own entrainment raster for one horizon
    # @param src_dir: Path to the horizon folder, or its scenarios/ subfolder
    # @returns: Path to <set>/scour_depth.tif, or None
    src_dir = Path(src_dir)
    for d in (src_dir, src_dir.parent):
        f = d / 'scour_depth.tif'
        if f.exists() and (d / 'manifest.json').exists():
            return f
    return None


def discover_horizons(date_dir):
    # List the horizons present for one cycle, in numeric order
    # @param date_dir: Path to the dated folder (source, or the local copy)
    # @returns: list of (source_dir, label) for horizons that hold scenarios,
    #           or whose release set was published empty (status no_scenarios)
    # T_n folders are the rule. A top-level 'scenarios' folder with
    #       no T_0 counts as T_0 for cycles written under the old layout.
    date_dir = Path(date_dir)
    if not date_dir.is_dir():
        return []
    found = {}
    for d in date_dir.iterdir():
        if not d.is_dir():
            continue
        m = HORIZON_RE.match(d.name)
        if not m:
            continue
        for cand in (d, d / 'scenarios'):
            if _has_scenarios(cand):
                found[int(m.group(1))] = (cand, d.name)
                break
        else:
            info = release_info(d)
            if info and info['status'] == 'no_scenarios':
                found[int(m.group(1))] = (d, d.name)
    if 0 not in found and _has_scenarios(date_dir / 'scenarios'):
        found[0] = (date_dir / 'scenarios', 'T_0')
    return [found[k] for k in sorted(found)]


def select_horizons(horizons, requested):
    # Apply a horizon filter, failing on labels the cycle does not have
    # @param horizons: list of (source_dir, label) from discover_horizons
    # @param requested: list of labels, or None / 'all' for everything
    # @returns: filtered list in numeric order
    # @raises ValueError: if a requested label is not present
    if not requested or requested == 'all':
        return horizons
    available = {label for _, label in horizons}
    unknown = set(requested) - available
    if unknown:
        raise ValueError(
            f"horizon(s) not found for this cycle: {', '.join(sorted(unknown))} "
            f"(available: {', '.join(sorted(available)) or 'none'})"
        )
    return [(d, label) for d, label in horizons if label in requested]


def copy_horizon(src_dir, dst_dir):
    # Copy one horizon's scenario inputs into the work tree
    # @param src_dir: Path to the horizon folder (read-only)
    # @param dst_dir: Path to intermediate/runout/<slug>/<date>/<T_n>/
    # @returns: int - number of scenario folders copied
    # each scenario's inputs land in scenario_###/release/. The
    #       destination is wiped first so a rerun never mixes stale
    #       scenarios with fresh ones.
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True)

    n = 0
    for sub in sorted(src_dir.iterdir()):
        if not sub.is_dir() or not sub.name.startswith('scenario_'):
            continue
        inp = sub / 'inputs' if (sub / 'inputs').is_dir() else sub
        release_dir = dst_dir / sub.name / 'release'
        release_dir.mkdir(parents=True)
        for f in inp.iterdir():
            if f.is_file():
                shutil.copy2(f, release_dir / f.name)
        n += 1
    logging.info(f"copied {n} scenario folder(s): {src_dir} -> {dst_dir}")
    return n


def existing_copy_ok(dst_dir):
    # True when the work tree already holds scenario inputs for a horizon
    # @param dst_dir: Path to intermediate/runout/<slug>/<date>/<T_n>/
    # @returns: bool
    dst_dir = Path(dst_dir)
    return dst_dir.is_dir() and any(
        (c / 'release' / 'release.geojson').exists()
        for c in dst_dir.iterdir() if c.is_dir()
    )

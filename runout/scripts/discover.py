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
"""

# --------- Load Libraries --------
import logging
import re
import shutil
from pathlib import Path


HORIZON_RE = re.compile(r'^T_(\d+)$')


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


def discover_horizons(date_dir):
    # List the horizons present for one cycle, in numeric order
    # @param date_dir: Path to the dated folder (source, or the local copy)
    # @returns: list of (source_dir, label) for horizons that hold scenarios
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

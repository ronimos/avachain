"""
Run Manifest

Author: Valerie Foley
Last Updated: 10/2026

Description:
    Writes manifest.json for a run: the resolved profile and physics with
    hashes, the path profile, the scenario source and its modification
    time, the survey file and date, status per horizon, the git commit and
    the AvaFrame version. The report header and the verify mode read from
    this file rather than from live config.
"""

# --------- Load Libraries --------
import datetime
import json
import logging
import re
import subprocess
from pathlib import Path

from . import profiles as prof

PIPELINE_VERSION = '3.0.0'


def git_commit(repo_dir):
    # Short git commit of the code, or None outside a checkout
    # @param repo_dir: Path inside the repository
    # @returns: str or None
    try:
        out = subprocess.run(
            ['git', '-C', str(repo_dir), 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True, timeout=5
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def avaframe_version():
    # Installed AvaFrame version, or None when it is not importable
    # @param: None
    # @returns: str or None
    # avaframe has no __version__ (2.1 read 'unknown'); the installed
    # distribution's metadata has it
    try:
        from importlib.metadata import version
        return version('avaframe')
    except Exception:
        pass
    try:
        import avaframe
        return getattr(avaframe, '__version__', 'unknown')
    except Exception:
        return None


def survey_date_from_name(path):
    # Pull a date out of a survey file name like 260114_Professor_HS.tif
    # @param path: Path or str
    # @returns: str YYYY-MM-DD, or None if no date pattern is found
    name = Path(path).name
    m = re.match(r'^(\d{2})(\d{2})(\d{2})[_-]', name)
    if m:
        yy, mm, dd = m.groups()
        return f"20{yy}-{mm}-{dd}"
    m = re.search(r'(\d{4})-?(\d{2})-?(\d{2})', name)
    if m:
        return '-'.join(m.groups())
    return None


def file_record(path):
    # Name, size and modification time of a file, for provenance
    # @param path: Path or str or None
    # @returns: dict or None
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return {'path': str(p), 'exists': False}
    st = p.stat()
    return {
        'path': str(p),
        'exists': True,
        'size_bytes': st.st_size,
        'modified': datetime.datetime.fromtimestamp(st.st_mtime).isoformat(timespec='seconds'),
    }


def build_manifest(profile, sections, physics, path_profile, local,
                   cycle_date, issued, source_dir, snow_height, survey_date,
                   run_name, repo_dir):
    # Assemble the manifest dict for one run (horizon entries added later)
    # @param profile: resolved profile mapping
    # @param sections: merged physics + profile sections
    # @param physics: dict from config.load_physics
    # @param path_profile: dict from config.load_path_profile
    # @param local: dict from config.load_local
    # @param cycle_date: str YYYYMMDD
    # @param issued: datetime the run started
    # @param source_dir: Path to the dated scenario folder (or None in --no-copy)
    # @param snow_height: Path to the snow height raster or None
    # @param survey_date: str or None
    # @param run_name: str
    # @param repo_dir: Path to the code checkout
    # @returns: dict
    physics_hash = prof.content_hash(prof.canonical(sections, profile['mode']))
    return {
        'pipeline_version': PIPELINE_VERSION,
        'git_commit': git_commit(repo_dir),
        'avaframe_version': avaframe_version(),
        'run_name': run_name,
        'mode': profile['mode'],
        'issued': issued.isoformat(timespec='seconds'),
        'cycle_date': cycle_date,
        'path': path_profile['slug'],
        'display': path_profile.get('display') or {},
        'profile': {
            'name': profile['name'],
            'description': profile.get('description'),
            'source': profile.get('_source'),
            'overrides': profile.get('_overrides') or [],
            'products': profile.get('products') or [],
            'entrainment_mode': (sections.get('entrainment') or {}).get('mode'),
        },
        'physics': {
            'name': profile['physics'],
            'file': physics.get('_file'),
            'hash': physics_hash,
            'depth_convention': sections.get('depth_convention'),
            'resolved': {s: sections.get(s) for s in prof.PROFILE_SECTIONS
                         if s not in prof.NON_RESULT_SECTIONS},
        },
        'path_profile': {
            'file': path_profile.get('_file'),
            'hash': path_profile.get('_hash'),
            'resolved': {k: v for k, v in path_profile.items()
                         if not k.startswith('_')},
        },
        'local': {'avachain_root': str(local['avachain_root']),
                  'scenario_root': str(local['scenario_root'])},
        'scenario_source': file_record(source_dir),
        'survey': {
            'file': file_record(snow_height),
            'date': survey_date,
        },
        'horizons': {},
        'status': 'running',
        'finished': None,
    }


def add_horizon(manifest, label, source_dir, n_scenarios, valid_date,
                outputs_dir, work_dir, status):
    # Record one horizon's inputs and result in the manifest
    # @param manifest: dict from build_manifest (modified in place)
    # @param label: horizon label (T_0, T_1, ...)
    # @param source_dir: Path the scenarios came from
    # @param n_scenarios: int
    # @param valid_date: str YYYY-MM-DD the horizon is valid for
    # @param outputs_dir: Path to the horizon's outputs
    # @param work_dir: Path to the horizon's work tree
    # @param status: 'ok', 'FAILED (exit n)', 'skipped (...)', 'dry-run'
    # @returns: None
    manifest['horizons'][label] = {
        'valid_date': valid_date,
        'source': str(source_dir) if source_dir else None,
        'n_scenarios': n_scenarios,
        'outputs_dir': str(outputs_dir),
        'work_dir': str(work_dir),
        'status': status,
    }


def valid_date_for(cycle_date, label):
    # Valid date of a horizon: T_n is cycle date plus n days
    # @param cycle_date: str YYYYMMDD
    # @param label: horizon label
    # @returns: str YYYY-MM-DD
    base = datetime.date(int(cycle_date[:4]), int(cycle_date[4:6]), int(cycle_date[6:8]))
    n = int(label.split('_', 1)[1])
    return (base + datetime.timedelta(days=n)).isoformat()


def finalize(manifest, status):
    # Stamp the finish time and overall status
    # @param manifest: dict (modified in place)
    # @param status: 'ok', 'partial', 'failed', 'dry-run'
    # @returns: None
    manifest['status'] = status
    manifest['finished'] = datetime.datetime.now().isoformat(timespec='seconds')


def write(manifest, path):
    # Write the manifest as JSON
    # @param manifest: dict
    # @param path: Path to manifest.json
    # @returns: Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(manifest, f, indent=2, default=str)
    logging.debug(f"manifest written: {path}")
    return path

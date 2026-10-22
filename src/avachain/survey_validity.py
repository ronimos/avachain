"""
survey_validity.py — which clusters a UAS survey can be trusted for.

A survey's snow height is the surface minus the snow-free DEM, and
spatial_model.resample clips it at min_valid_hs_m = 0. Where the survey
surface is biased low (registration, the snow-free DEM, processing), whole
patches go negative and are stored as exact zeros, and the positive cells in
the same patch read low by the same error. Measured over all 23 surveys of
the 2025-26 season and 6565 clusters (signed jump from the time-interpolation
of the surveys either side; clean cluster-surveys +2 cm):

    share of zero cells   0-10%   10-25%   25-50%   50-75%   >75%
    median bias            -22      -45      -61      -78     -99 cm

The share of clusters affected varies by survey (02-12, 02-19, 03-08: over
21%), not with snow cover, so it is the survey, not bare ground. On 03-02,
465 clusters read entirely 0 between ~0.7 m (02-21) and ~0.85 m (03-05).

A cluster-survey is invalid when more than MAX_ZERO_SHARE of its cells are
exact zeros or NaN (no data). Two exceptions keep real bare ground valid:
- the first survey of the season (nothing earlier to carry forward, and the
  early-season snow really is near zero: 58% of clusters cross 25% zeros);
- clusters whose last valid HS was already below NEAR_BARE_M.
At 10%, 11% of cluster-surveys are invalid, and invalid stretches are short:
3528 clusters' longest is one survey, 689 two, 442 three or more.

Consumers: gap-fill carries invalid clusters forward from their last valid
survey instead of pinning to the survey; the survey anchor and reinit skip
them.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

MAX_ZERO_SHARE = 0.10
NEAR_BARE_M = 0.15
VALIDITY_FILE = "survey_validity.json"


def _per_cluster(grid: np.ndarray, cluster_map: np.ndarray):
    """(cluster ids, no-data share, median of finite cells) per cluster.

    No data = exact zero (a clipped negative) or NaN (a 1 m cell with too
    few valid survey pixels, see spatial_model.load_and_resample_survey).
    A cluster with no finite cell at all is fully no-data (median NaN).
    """
    inside = cluster_map > 0
    ids = cluster_map[inside]
    vals = grid[inside]
    fin = np.isfinite(vals)
    size = ids.max() + 1
    n_all = np.bincount(ids, minlength=size)
    n_bad = np.bincount(ids[~fin | (vals == 0)], minlength=size)
    order = np.argsort(ids[fin], kind="stable")
    sid, sval = ids[fin][order], vals[fin][order]
    u_fin, start = np.unique(sid, return_index=True)
    med = {int(c): float(np.median(v)) for c, v in zip(u_fin, np.split(sval, start[1:]))}
    uniq = np.unique(ids)
    share = {int(c): n_bad[c] / n_all[c] for c in uniq}
    med = {int(c): med.get(int(c), np.nan) for c in uniq}
    return uniq, share, med


def compute_validity(grids: dict, cluster_map: np.ndarray,
                     max_zero_share: float = MAX_ZERO_SHARE,
                     near_bare_m: float = NEAR_BARE_M) -> dict:
    """{date: sorted list of invalid cluster ids}, surveys in date order."""
    last_valid: dict[int, float] = {}
    out = {}
    for i, date in enumerate(sorted(grids)):
        uniq, share, med = _per_cluster(grids[date], cluster_map)
        bad = []
        for c in uniq:
            c = int(c)
            suspect = share[c] > max_zero_share
            bare_before = last_valid.get(c, np.inf) < near_bare_m
            if suspect and i > 0 and not bare_before:
                bad.append(c)
            else:
                last_valid[c] = med[c]
        out[date] = sorted(bad)
    return out


def write_validity(analysis_dir: Path, validity: dict) -> Path:
    p = Path(analysis_dir) / VALIDITY_FILE
    p.write_text(json.dumps(validity, indent=1))
    return p


def invalid_clusters(analysis_dir: Path, date: str) -> set:
    """Invalid cluster ids for a survey date; empty if never computed."""
    p = Path(analysis_dir) / VALIDITY_FILE
    if not p.exists():
        return set()
    return set(json.loads(p.read_text()).get(date, []))


def invalid_cells(cluster_map: np.ndarray, cids) -> np.ndarray:
    """Boolean cell mask of the given clusters."""
    if not cids:
        return np.zeros(cluster_map.shape, dtype=bool)
    return np.isin(cluster_map, np.fromiter(cids, dtype=cluster_map.dtype))


def carry_forward(hs_start: np.ndarray, transport, inv_start: np.ndarray,
                  inv_end: np.ndarray, prev_end):
    """Adjust one gap-fill pair for invalid survey clusters.

    hs_start   survey grid the pair starts from
    transport  the pair's transport field (None for station-only gap-fill)
    inv_start  cells of clusters invalid on the start survey
    inv_end    cells of clusters invalid on the end survey
    prev_end   the previous pair's predicted grid at the start survey, or None

    Invalid start cells take prev_end (their last valid survey carried
    forward), when there is one. Transport is fitted from the observed dHS,
    which a biased survey corrupts at either end (a patch reading 0 on 03-02
    is a +0.85 m "deposit" by 03-05), so cells invalid at either end get none
    and follow the station.
    """
    if prev_end is not None and inv_start.any():
        hs_start = np.where(inv_start, prev_end, hs_start)
    if transport is not None:
        transport = np.where(inv_start | inv_end, 0.0, transport)
    return hs_start, transport

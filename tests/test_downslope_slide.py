"""Downslope slide test (model_params.USE_DOWNSLOPE_SLIDE, Ron 2026-11-16; same
as ~/avalanche-release-areas tests/test_downslope_slide.py).

A fractured cluster below atan(mu) = 31.8 deg stays in the release only while
the slab above still pushes it: the net drive N = rho g h A / cos th
(sin th - mu cos th) is routed down the fall line, and a cluster with
P = N + inflow <= 0 is held. Pinned on a synthetic 4-neighbour grid where the
push runs straight down each column, so the carried length is exact: per unit
rho g h A, N(38 deg) = 0.1613 and N(30 deg) = -0.0426, so two steep rows
carry 7 bench rows (0.3226 / 0.0426 = 7.6) and the 8th is held."""
import random
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "avachain"))
import release_geometry as rg    # noqa: E402
import model_params as mp        # noqa: E402

S = 5.0                       # cluster spacing, m
SOUTH, EAST = (0.0, -1.0), (1.0, 0.0)


def _grid(slopes, ncol=3, axis='row'):
    """Rows (axis='row', fall south) or columns (axis='col', fall east) of
    clusters with the given slopes, top first; 4-neighbour graph."""
    nrow = len(slopes) if axis == 'row' else ncol
    ncl = ncol if axis == 'row' else len(slopes)
    cid = lambda r, c: r * ncl + c + 1
    cent, nb, slope, elev = {}, {}, {}, {}
    for r in range(nrow):
        for c in range(ncl):
            k = cid(r, c)
            cent[k] = (c * S, -r * S)
            nb[k] = [cid(r + dr, c + dc) for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                     if 0 <= r + dr < nrow and 0 <= c + dc < ncl]
            i = r if axis == 'row' else c
            slope[k] = slopes[i]
    # elevation falls along the axis by S tan(slope) per step
    for k in cent:
        r, c = (k - 1) // ncl, (k - 1) % ncl
        i = r if axis == 'row' else c
        elev[k] = -sum(S * np.tan(np.radians(slopes[j])) for j in range(i))
    props = {k: {'slab_density': 300.0, 'slab_thickness': 1.0} for k in cent}
    area = {k: S * S for k in cent}
    return cid, cent, nb, slope, elev, props, area


def _held(slopes, axis='row', fall=None, props_fn=None, fall_fn=None, ncol=3, **kw):
    cid, cent, nb, slope, elev, props, area = _grid(slopes, ncol, axis)
    if props_fn:
        props_fn(props, cid)
    f = fall or (SOUTH if axis == 'row' else EAST)
    fall_of = {k: f for k in cent}
    if fall_fn:
        fall_fn(fall_of, cid)
    held, st = rg.downslope_held(set(cent), nb, cent, props, slope.get, elev, area,
                                 fall_of, **kw)
    return held, st, cid


def _rows(held, cid, ncol=3):
    return sorted({(k - 1) // ncol for k in held})


def test_defaults_and_cli():
    assert mp.USE_DOWNSLOPE_SLIDE is False
    assert mp.DOWNSLOPE_ALIGN_COS == 0.707
    import analysis_pipeline as ap
    import inspect
    src = inspect.getsource(ap)
    assert "--downslope-slide" in src
    assert "mp.USE_DOWNSLOPE_SLIDE = args.downslope_slide" in src


def test_steep_ground_is_never_held():
    held, st, _ = _held([38.0] * 6 + [33.0] * 6)
    assert held == set() and st['gentle'] == 0


def test_bench_is_carried_as_far_as_the_push_reaches():
    """2 steep rows carry 7 bench rows; rows 9.. are held."""
    held, st, cid = _held([38.0, 38.0] + [30.0] * 12)
    assert _rows(held, cid) == list(range(9, 14))
    # the same by an independent running sum
    n = lambda th: (np.sin(np.radians(th)) - mp.SEVERED_SLIDE_MU * np.cos(np.radians(th))) \
        / np.cos(np.radians(th))
    p, first_held = 0.0, None
    for i, th in enumerate([38.0, 38.0] + [30.0] * 12):
        p += n(th)
        if p <= 0:
            first_held = i
            break
    assert first_held == 9


def test_gentle_ground_with_nothing_above_is_held():
    held, st, _ = _held([30.0] * 5)
    assert len(held) == 15 and st['gentle'] == 15


def test_held_cluster_passes_nothing_on():
    """Steep again below a held bench: the steep rows release on their own,
    the bench rows between stay held."""
    held, _, cid = _held([38.0] + [30.0] * 6 + [38.0] * 3)
    # 1 steep row carries 3 bench rows (0.1613 / 0.0426 = 3.8): rows 4-6 held
    assert _rows(held, cid) == [4, 5, 6]


def test_push_follows_the_fall_line_in_the_map_frame():
    """Fall east (elevation drops with the column): the push goes east, so the
    result equals the row case transposed. Routing it south instead (the
    array-row direction, wrong here) finds no lower neighbour and holds every
    gentle cluster."""
    slopes = [38.0, 38.0] + [30.0] * 12
    held, _, cid = _held(slopes, axis='col', ncol=3)
    ncl = len(slopes)
    assert sorted({(k - 1) % ncl for k in held}) == list(range(9, 14))
    held_wrong, _, _ = _held(slopes, axis='col', ncol=3, fall=SOUTH)
    assert sorted({(k - 1) % ncl for k in held_wrong}) == list(range(2, 14))


def test_cluster_without_slab_data_passes_its_inflow_on():
    def no_props(props, cid):
        for c in range(3):
            props[cid(1, c)] = {}
    held, _, cid = _held([38.0, 38.0] + [30.0] * 12, props_fn=no_props)
    # row 0 alone pushes (0.1613), row 1 adds nothing: bench rows 2-4 carried
    # (0.1613 - 3 x 0.0426 > 0), row 5 held
    assert _rows(held, cid) == list(range(5, 14))


def test_cluster_without_fall_line_passes_nothing_on():
    def no_fall(fall_of, cid):
        for c in range(3):
            del fall_of[cid(1, c)]
    held, _, cid = _held([38.0, 38.0] + [30.0] * 12, fall_fn=no_fall)
    assert _rows(held, cid) == list(range(2, 14))


def test_result_does_not_depend_on_order():
    cid, cent, nb, slope, elev, props, area = _grid([38.0, 38.0] + [30.0] * 12, ncol=4)
    fall_of = {k: SOUTH for k in cent}
    want = rg.downslope_held(set(cent), nb, cent, props, slope.get, elev, area, fall_of)
    rnd = random.Random(0)
    for _ in range(5):
        nb2 = {k: rnd.sample(v, len(v)) for k, v in rnd.sample(list(nb.items()), len(nb))}
        reg = set(rnd.sample(sorted(cent), len(cent)))
        assert rg.downslope_held(reg, nb2, cent, props, slope.get, elev, area, fall_of) == want


def test_mu_reads_config_at_call_time(monkeypatch):
    slopes = [38.0, 38.0] + [30.0] * 12
    monkeypatch.setattr(mp, 'SEVERED_SLIDE_MU', 0.5)     # atan 0.5 = 26.6 deg
    held, _, _ = _held(slopes)
    assert held == set()

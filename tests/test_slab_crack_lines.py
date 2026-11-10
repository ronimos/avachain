"""Running slab cracks (release_geometry.USE_SLAB_CRACK_LINES, Ron 2026-11-10).

A crack started by the tensile gate runs as a crown (cross-slope) or flank
(fall line) line through the admitted region while the slab is loaded, and the
area it severs from the trigger is removed; the line clusters stay. Pinned on a
synthetic grid: Jan 18 cannot show order independence or the loaded test."""
import random
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "avachain"))
import release_geometry as rg    # noqa: E402

N, S = 12, 3            # N x N clusters, each S x S pixels of 1 m
FALL = (0.0, -1.0)      # due south, map frame (east, north)


def _grid():
    cmap = np.zeros((N * S, N * S), dtype=np.int32)
    cid = lambda r, c: r * N + c + 1
    cent, nb = {}, {}
    for r in range(N):
        for c in range(N):
            cmap[r * S:(r + 1) * S, c * S:(c + 1) * S] = cid(r, c)
            # row 0 is the NORTH edge: y decreases with row
            cent[cid(r, c)] = (c * S + S / 2, -(r * S + S / 2))
            nb[cid(r, c)] = [cid(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                             if (dr or dc) and 0 <= r + dr < N and 0 <= c + dc < N]
    props = {k: {'slab_density': 300.0, 'slab_thickness': 1.0} for k in cent}
    return cmap, cid, cent, nb, props


def _run(cracks, trigger, reached=None, slope=38.0, nb=None, **kw):
    cmap, cid, cent, nb0, props = _grid()
    reached = set(cent) if reached is None else reached
    return rg.slab_crack_lines(reached, cracks, trigger, nb or nb0, cent, FALL, props,
                               lambda c: slope, cmap, **kw)


def test_orientation_in_map_frame():
    o = rg.slab_crack_orientation
    assert o((0, 0), (0, 5), FALL) == 'crown'     # north = upslope
    assert o((0, 0), (5, 0), FALL) == 'flank'     # east = cross-slope
    assert o((0, 0), (-5, 0), FALL) == 'flank'
    assert o((0, 0), (0, -5), FALL) is None       # downslope: Stauchwall side
    assert o((0, 0), (0, 0), FALL) is None


def test_flank_line_severs_the_far_side_and_keeps_the_line():
    _, cid, cent, _, _ = _grid()
    x = cid(6, 6)                                  # tensile-gate failure, interior
    reached = set(cent) - {x}
    severed, st = _run({x: 'flank'}, cid(6, 2), reached)
    east = {cid(r, c) for r in range(N) for c in range(7, N)}
    assert severed == east
    assert st['line_flank'] == N - 1               # column 6 above and below x
    assert not severed & {cid(r, 6) for r in range(N)}


def test_crown_line_severs_upslope_of_a_downslope_trigger():
    _, cid, cent, _, _ = _grid()
    x = cid(6, 6)
    reached = set(cent) - {x}
    severed, st = _run({x: 'crown'}, cid(10, 6), reached)
    assert severed == {cid(r, c) for r in range(6) for c in range(N)}
    assert st['line_crown'] > 0 and st['line_flank'] == 0


def test_no_net_load_no_line():
    _, cid, cent, _, _ = _grid()
    x = cid(6, 6)
    severed, st = _run({x: 'flank'}, cid(6, 2), set(cent) - {x}, slope=rg.FRICTION_DEG - 1)
    assert severed == set() and st['line_clusters'] == 0


def test_line_stops_at_the_trigger():
    _, cid, cent, _, _ = _grid()
    x = cid(6, 6)
    severed, st = _run({x: 'flank'}, cid(3, 6), set(cent) - {x})
    # runs rows 4-5 and 7-11 of column 6, stops at the trigger in row 3; east and
    # west stay joined across rows 0-2, so nothing is severed
    assert st['line_flank'] == 7
    assert severed == set()


def test_result_does_not_depend_on_order():
    cmap, cid, cent, nb, props = _grid()
    holes = {cid(6, 6), cid(3, 9), cid(8, 3)}
    cracks = {cid(6, 6): 'flank', cid(3, 9): 'crown', cid(8, 3): 'flank'}
    reached = set(cent) - holes
    want = _run(cracks, cid(10, 1), reached)
    rnd = random.Random(0)
    for _ in range(5):
        nb2 = {k: rnd.sample(v, len(v)) for k, v in rnd.sample(list(nb.items()), len(nb))}
        cr2 = dict(rnd.sample(list(cracks.items()), len(cracks)))
        assert _run(cr2, cid(10, 1), set(rnd.sample(sorted(reached), len(reached))), nb=nb2) == want


def test_defaults_and_cli():
    assert rg.USE_SLAB_CRACK_LINES is True
    assert rg.SLAB_CRACK_BAND_M == 5.0 and rg.SLAB_CRACK_ALIGN_COS == 0.707
    import analysis_pipeline as ap
    import inspect
    assert "--slab-crack-lines" in inspect.getsource(ap)

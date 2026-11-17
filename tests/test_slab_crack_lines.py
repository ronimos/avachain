"""Running slab cracks (model_params.USE_SLAB_CRACK_LINES, Ron 2026-11-10;
SLAB_CRACK_MODE 2026-11-16; same as ~/avalanche-release-areas
tests/test_slab_crack_lines.py).

A crack started by the tensile gate runs as a crown (cross-slope) or flank
(fall line) line through the admitted region while the slab is loaded, and the
area it severs from the trigger is removed; the line clusters stay. Pinned on a
synthetic grid: Jan 18 cannot show order independence or the loaded test."""
import inspect
import random
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "avachain"))
import release_geometry as rg    # noqa: E402
import model_params as mp        # noqa: E402

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


def _run(cracks, trigger, reached=None, slope=38.0, nb=None, mode='shear', **kw):
    """The tests above the fracture-mode section pin 'shear' (every release
    before 2026-11-16); 'opening' is the default since then."""
    cmap, cid, cent, nb0, props = _grid()
    reached = set(cent) if reached is None else reached
    return rg.slab_crack_lines(reached, cracks, trigger, nb or nb0, cent, FALL, props,
                               lambda c: slope, cmap, mode=mode, **kw)


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
    severed, st = _run({x: 'flank'}, cid(6, 2), set(cent) - {x}, slope=mp.PHI_DEG - 1)
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
    assert mp.USE_SLAB_CRACK_LINES is True
    assert mp.SLAB_CRACK_BAND_M == 5.0 and mp.SLAB_CRACK_ALIGN_COS == 0.707
    import analysis_pipeline as ap
    import inspect
    assert "--slab-crack-lines" in inspect.getsource(ap)


# --- Fracture mode (mp.SLAB_CRACK_MODE, Ron 2026-11-16) ------------------

def _fall(bearing_deg):
    b = np.radians(bearing_deg)
    return (float(np.sin(b)), float(np.cos(b)))     # map frame (east, north)


def _ridge(tilt_deg):
    """Fall lines per column: west of column 6 turned tilt_deg toward the west,
    east of it toward the east (tilt > 0: a convex roll along column 6; < 0: a
    gully). Column 6 falls due south."""
    _, cid, cent, _, _ = _grid()
    return {cid(r, c): _fall(180.0 + np.sign(6 - c) * tilt_deg) if c != 6 else FALL
            for r in range(N) for c in range(N)}


def test_opening_default_and_cli():
    assert mp.SLAB_CRACK_MODE == 'opening'
    assert mp.SLAB_CRACK_MODES == ('opening', 'shear')
    import analysis_pipeline as ap
    import inspect
    src = inspect.getsource(ap)
    assert "--slab-crack-mode" in src
    assert "mp.SLAB_CRACK_MODE = args.slab_crack_mode" in src


def test_mode_default_reads_config_at_call_time(monkeypatch):
    _, cid, cent, _, _ = _grid()
    x = cid(6, 6)
    args = ({x: 'flank'}, cid(6, 2), set(cent) - {x})
    monkeypatch.setattr(mp, 'SLAB_CRACK_MODE', 'shear')
    assert _run(*args, mode=None) == _run(*args, mode='shear')
    monkeypatch.setattr(mp, 'SLAB_CRACK_MODE', 'opening')
    assert _run(*args, mode=None) == _run(*args, mode='opening')
    with pytest.raises(ValueError):
        _run(*args, mode='tearing')


def test_opening_planar_slope_has_no_flank_line():
    """No cross-slope tension on a plane, so a flank crack does not run; the
    same crack runs and severs the east side in 'shear'."""
    _, cid, cent, _, _ = _grid()
    x = cid(6, 6)
    reached = set(cent) - {x}
    sev, st = _run({x: 'flank'}, cid(6, 2), reached, mode='opening')
    assert sev == set() and st['line_clusters'] == 0
    sev_sh, _ = _run({x: 'flank'}, cid(6, 2), reached, mode='shear')
    assert sev_sh  # the shear line does cut


def test_opening_flank_runs_along_a_convex_roll():
    _, cid, cent, _, _ = _grid()
    x = cid(6, 6)
    reached = set(cent) - {x}
    sev, st = _run({x: 'flank'}, cid(6, 2), reached, mode='opening', fall_of=_ridge(15.0))
    assert sev == {cid(r, c) for r in range(N) for c in range(7, N)}
    assert st['line_flank'] == N - 1


def test_opening_flank_does_not_run_in_a_gully():
    _, cid, cent, _, _ = _grid()
    x = cid(6, 6)
    sev, st = _run({x: 'flank'}, cid(6, 2), set(cent) - {x}, mode='opening',
                   fall_of=_ridge(-15.0))
    assert sev == set() and st['line_clusters'] == 0


def test_opening_flank_still_needs_net_load():
    """Diverging fall lines on ground below phi: nothing pulls, no line."""
    _, cid, cent, _, _ = _grid()
    x = cid(6, 6)
    sev, st = _run({x: 'flank'}, cid(6, 2), set(cent) - {x}, mode='opening',
                   fall_of=_ridge(15.0), slope=mp.PHI_DEG - 1)
    assert sev == set() and st['line_clusters'] == 0


def _edge(side):
    """Region = column 6 plus the columns on one side of it, the crack at
    (6, 6): a flank line down column 6, on the region's edge. The other side
    is attached slab (2026-11-16: it does not move, F = 0)."""
    _, cid, cent, _, _ = _grid()
    cols = range(7) if side == 'west' else range(6, N)
    x = cid(6, 6)
    return x, {cid(r, c) for r in range(N) for c in cols} - {x}, cid


@pytest.mark.parametrize('side', ['west', 'east'])
def test_opening_edge_flank_opens_where_the_detached_side_moves_away(side):
    """Convex roll: the detached slab on either side falls away from the line,
    and the attached side stays put, so the line runs (it severs nothing:
    there is nothing beyond it)."""
    x, region, cid = _edge(side)
    trig = cid(6, 2) if side == 'west' else cid(6, 10)
    sev, st = _run({x: 'flank'}, trig, region, mode='opening', fall_of=_ridge(15.0))
    assert st['line_flank'] == N - 1 and sev == set()


@pytest.mark.parametrize('tilt', [0.0, -15.0])
def test_opening_edge_flank_does_not_open_on_a_plane_or_in_a_gully(tilt):
    """Plane: the detached slab moves along the line. Gully: it moves toward
    the attached slab. Neither opens the line."""
    x, region, cid = _edge('west')
    sev, st = _run({x: 'flank'}, cid(6, 2), region, mode='opening', fall_of=_ridge(tilt))
    assert st['line_clusters'] == 0


def test_opening_crown_follows_the_local_fall_line():
    """Trigger bearing due south, but the slope falls SE (135 deg) everywhere:
    the 'opening' crown runs NE-SW across the local fall line (cells r + c = 12)
    and severs the upslope NW triangle; 'shear' runs it E-W along row 6."""
    _, cid, cent, _, _ = _grid()
    x = cid(6, 6)
    reached = set(cent) - {x}
    se = {k: _fall(135.0) for k in cent}
    sev, st = _run({x: 'crown'}, cid(10, 10), reached, mode='opening', fall_of=se)
    # (1,11)..(11,1) without the hole (6,6), plus one 45 deg step at each grid
    # edge into the corner cells (0,11) and (11,0)
    assert st['line_crown'] == 12
    assert sev == {cid(r, c) for r in range(N) for c in range(N)
                   if r + c < 12} - {cid(0, 11), cid(11, 0)}
    sev_sh, st_sh = _run({x: 'crown'}, cid(10, 10), reached, mode='shear', fall_of=se)
    assert st_sh['line_crown'] == N - 1
    assert sev_sh == {cid(r, c) for r in range(6) for c in range(N)}


def test_shear_ignores_local_fall_lines():
    _, cid, cent, _, _ = _grid()
    holes = {cid(6, 6), cid(3, 9)}
    cracks = {cid(6, 6): 'flank', cid(3, 9): 'crown'}
    reached = set(cent) - holes
    assert (_run(cracks, cid(10, 1), reached, mode='shear', fall_of=_ridge(20.0))
            == _run(cracks, cid(10, 1), reached, mode='shear'))


def test_opening_result_does_not_depend_on_order():
    cmap, cid, cent, nb, props = _grid()
    holes = {cid(6, 6), cid(3, 9), cid(8, 3)}
    cracks = {cid(6, 6): 'flank', cid(3, 9): 'crown', cid(8, 3): 'flank'}
    reached = set(cent) - holes
    kw = dict(mode='opening', fall_of=_ridge(15.0))
    want = _run(cracks, cid(10, 1), reached, **kw)
    rnd = random.Random(1)
    for _ in range(5):
        nb2 = {k: rnd.sample(v, len(v)) for k, v in rnd.sample(list(nb.items()), len(nb))}
        cr2 = dict(rnd.sample(list(cracks.items()), len(cracks)))
        assert _run(cr2, cid(10, 1), set(rnd.sample(sorted(reached), len(reached))),
                    nb=nb2, **kw) == want


@pytest.mark.parametrize('bearing', [30.0, 120.0, 200.0, 315.0])
def test_cluster_fall_vector_in_map_frame(bearing):
    """A plane falling toward `bearing` (degrees from north, clockwise) on a
    north-up raster: row increases southward, so y = -row * px."""
    px, n = 1.0, 40
    rows, cols = np.mgrid[0:n, 0:n]
    x, y = cols * px, -rows * px
    b = np.radians(bearing)
    dem = -(x * np.sin(b) + y * np.cos(b)) * 0.7
    slope, aspect = rg.compute_slope_aspect(dem, px)
    sl = (slice(2, n - 2), slice(2, n - 2))
    v = rg.cluster_fall_vector(rows[sl].ravel(), cols[sl].ravel(), slope, aspect)
    assert v == pytest.approx(_fall(bearing), abs=1e-6)
    assert rg.cluster_fall_vector(np.array([5]), np.array([5]),
                                  np.zeros((n, n)), np.zeros((n, n))) is None


# --- Toughness gate on opening flanks (USE_OPENING_TOUGHNESS, 2026-11-16) ----

def _tough(kic, mode='opening', **kw):
    """Convex roll (+-15 deg), flank crack at (6, 6), every cell's K_Ic = kic.
    Returns (severed, stats, intact, cid)."""
    cmap, cid, cent, nb, props = _grid()
    for p in props.values():
        if kic is not None:
            p[mp.TOUGHNESS_K_COL] = kic
    x = cid(6, 6)
    intact = set()
    sev, st = rg.slab_crack_lines(set(cent) - {x}, {x: 'flank'}, cid(6, 2), nb, cent, FALL,
                                  props, lambda c: 38.0, cmap, mode=mode,
                                  fall_of=_ridge(15.0), intact_out=intact, **kw)
    return sev, st, intact, cid


def _k_at(a):
    """K at a step of that line: L = 33 m (column extent), 38 deg, spread
    2 sin 15, rho 300, h 1."""
    th = np.radians(38.0)
    drive = np.sin(th) - np.tan(np.radians(mp.PHI_DEG)) * np.cos(th)
    sig = 300.0 * 9.81 * 33.0 * drive * (2 * np.sin(np.radians(15.0))) / 2.0
    return mp.TOUGHNESS_EDGE_F * sig * np.sqrt(np.pi * a)


def test_opening_toughness_default_and_cli():
    assert mp.USE_OPENING_TOUGHNESS is True
    assert "--opening-toughness" in inspect.getsource(__import__("analysis_pipeline"))


def test_opening_toughness_marks_short_line_cells_and_changes_no_line():
    """K grows with the line's length a. K_Ic between K(3 m) and K(6 m): only
    the first cell each way (a = 3 m) cannot break. Lines and severed set are
    those without the gate."""
    base_sev, base_st, _, cid = _tough(None)
    sev, st, intact, _ = _tough((_k_at(3.0) + _k_at(6.0)) / 2)
    assert intact == {cid(5, 6), cid(7, 6)}
    assert sev == base_sev and st['line_flank'] == base_st['line_flank'] == N - 1
    assert st['intact'] == 2


def test_opening_toughness_extremes_and_missing_column():
    _, st, intact, cid = _tough(1e12)
    assert intact == {cid(r, 6) for r in range(N)} - {cid(6, 6)}
    assert _tough(1.0)[2] == set()
    assert _tough(None)[2] == set()           # no K_Ic column: no gate


def test_opening_toughness_off_and_shear_unaffected(monkeypatch):
    assert _tough(1e12, mode='shear')[2] == set()
    monkeypatch.setattr(mp, 'USE_OPENING_TOUGHNESS', False)
    assert _tough(1e12)[2] == set()

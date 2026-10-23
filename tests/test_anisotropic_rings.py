"""Tests for anisotropic BFS ring growth in propagate_release.

Ported from ~/avalanche-release-areas/tests/test_anisotropic_rings.py so the
production path is not the one without coverage — that is how the two
terrain-direction bugs survived here (see tests/test_terrain_direction.py).

The hypothesis under test: if one ring of the flood-fill stands for one
increment of crack-front advance in *time*, a front running supershear
along-slope (~1.6 c_s) and capped at c_s cross-slope should already be ~1.6x
further along-slope when it arrives, so the anisotropy belongs in the growth
rather than only in a terminal distance cap (USE_ANISOTROPIC_RINGS).

Two invariants matter:

  1. Default OFF must be bit-identical to the isotropic flood-fill.
  2. The cross-slope budget must be driven by the ALONG-SLOPE clock. Driving it
     from the global ring counter is self-defeating: flank growth keeps the
     flood-fill alive, which raises the flanks' own budget.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from affine import Affine

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "avachain"))

import release_geometry as rg                                   # noqa: E402
from release_geometry import (                                  # noqa: E402
    _hop_length_m,
    cluster_pixel_index,
    compute_slope_aspect,
    propagate_release,
)

T = Affine(1.0, 0.0, 400000.0, 0.0, -1.0, 4000000.0)
BLOCK = 4          # cluster side in pixels -> 4 m hop, as on Little Professor
N = 161            # grid side in pixels; trigger at the centre


@pytest.fixture
def scene():
    """Uniform 35 deg south-facing plane tiled with 4x4 m clusters.

    Due south means the fall line runs along +row and the cross-slope axis
    along +col. Every cluster carries identical properties, so no continuity or
    floor gate ever fires and only the distance caps and the ring budget can
    stop the flood-fill.
    """
    r, c = np.mgrid[0:N, 0:N]
    cluster_map = ((r // BLOCK) * (N // BLOCK + 1) + (c // BLOCK) + 1).astype(float)
    dem = -np.tan(np.radians(35.0)) * (-r.astype(float))   # falls toward +row
    slope_grid, _ = compute_slope_aspect(dem, 1.0)

    pixel_index = cluster_pixel_index(cluster_map)
    cids = sorted(pixel_index)
    trigger = int(cluster_map[N // 2, N // 2])

    meloche_df = pd.DataFrame({
        'Pi1_elastic': 50.0, 'Lambda': 2.0, 'Lambda_cross': 2.0,
        'tau_g': 800.0, 'L_t': 30.0, 'A_ca_brittle': 60.0,
    }, index=cids)
    snap_features = pd.DataFrame({
        'slab_thickness': 0.8, 'slab_density': 330.0,
        'wl_shear_strength': 2400.0, 'sigma_t': 4000.0,
    }, index=cids)

    return dict(trigger=trigger, cluster_map=cluster_map, dem=dem,
                slope_grid=slope_grid, meloche_df=meloche_df,
                snap_features=snap_features, pixel_index=pixel_index)


def _run(scene, **over):
    # A_ca is deliberately larger than the domain so neither the upslope cap
    # nor the lateral Gaume width (which falls back to A_ca without a theta
    # column) binds: the isotropic run is then bounded by the grid in every
    # direction and any anisotropy is the ring budget's doing.
    kw = dict(trigger_cluster_id=scene['trigger'], meloche_df=scene['meloche_df'],
              cluster_map=scene['cluster_map'], dem=scene['dem'],
              slope_grid=scene['slope_grid'], transform=T,
              A_ca=200.0, snap_features=scene['snap_features'],
              max_clusters=20000)
    kw.update(over)
    return propagate_release(**kw)


def _extent(scene, reached):
    """Max along-slope reach and flank width at the crown, both in metres.

    The flank width is measured only within a band of the trigger's own row:
    the ±45° sector test classifies a corner cluster at (80, 80) as
    *downslope*, so a plain max |Δcol| is set by corner leakage rather than by
    the cross-slope budget.
    """
    px = scene['pixel_index']
    t_rows, t_cols = px[scene['trigger']]
    t_r, t_c = t_rows.mean(), t_cols.mean()
    along = cross = 0.0
    for cid in reached:
        rows, cols = px[cid]
        d_along = abs(rows.mean() - t_r)
        along = max(along, d_along)
        if d_along <= 2 * BLOCK:
            cross = max(cross, abs(cols.mean() - t_c))
    return along, cross


@pytest.fixture(autouse=True)
def no_speed_cap(monkeypatch):
    """Isolate the rings: the mode III distance cap would otherwise bind first."""
    monkeypatch.setattr(rg, 'USE_MODE3_SPEED_CAP', False)
    monkeypatch.setattr(rg, 'USE_ANISOTROPIC_RINGS', False)


class TestDefaults:
    def test_rings_are_off_by_default(self):
        """Checked on the module source, which the autouse fixture overrides."""
        src = (Path(rg.__file__).read_text()
               .split('USE_ANISOTROPIC_RINGS = ')[1].split('\n')[0])
        assert src.strip() == 'False'

    def test_ring_step_is_derived_not_hardcoded(self):
        assert rg.RING_STEP_M is None

    def test_speed_ratio_is_broberg_not_a_free_parameter(self):
        """1.6 / 1.0 comes from Broberg (1989) via arrest_indices, and the ring
        budget must reuse it rather than introduce a second constant."""
        assert rg.MODE2_SPEED_RATIO == 1.6
        assert rg.MODE3_SPEED_RATIO == 1.0
        assert rg.MODE3_SPEED_RATIO / rg.MODE2_SPEED_RATIO == 0.625


class TestHopLength:
    def test_median_edge_separation(self):
        centroids = {1: (0.0, 0.0), 2: (0.0, 3.0), 3: (0.0, 10.0)}
        neighbours = {1: [2, 3], 2: [1], 3: [1]}
        # edges: 3, 10, 3, 10 -> median 6.5
        assert _hop_length_m(neighbours, centroids) == pytest.approx(6.5)

    def test_missing_centroids_are_skipped(self):
        centroids = {1: (0.0, 0.0), 2: (0.0, 4.0)}
        neighbours = {1: [2, 99], 2: [1], 99: [1]}
        assert _hop_length_m(neighbours, centroids) == pytest.approx(4.0)

    def test_empty_graph_gives_zero(self):
        assert _hop_length_m({}, {}) == 0.0


class TestAnisotropicGrowth:
    def test_rings_shorten_the_flanks_but_not_the_fall_line(self, scene, monkeypatch):
        """The whole point: cross-slope reach shrinks, along-slope reach does not."""
        monkeypatch.setattr(rg, 'USE_ANISOTROPIC_RINGS', False)
        _, iso = _run(scene)
        monkeypatch.setattr(rg, 'USE_ANISOTROPIC_RINGS', True)
        _, ani = _run(scene)

        iso_along, iso_cross = _extent(scene, iso)
        ani_along, ani_cross = _extent(scene, ani)

        assert ani_along == pytest.approx(iso_along, abs=BLOCK)
        assert ani_cross < iso_cross - 2 * BLOCK
        assert len(ani) < len(iso)

    def test_released_aspect_ratio_moves_toward_the_speed_ratio(self, scene, monkeypatch):
        monkeypatch.setattr(rg, 'USE_ANISOTROPIC_RINGS', True)
        _, ani = _run(scene)
        along, cross = _extent(scene, ani)
        ratio = rg.MODE2_SPEED_RATIO / rg.MODE3_SPEED_RATIO
        assert along / cross == pytest.approx(ratio, rel=0.25)

    def test_isotropic_region_is_not_elongated(self, scene):
        """Control: with no cap binding the same scene grows round, so the
        elongation above is the mechanism and not the fixture."""
        _, iso = _run(scene)
        along, cross = _extent(scene, iso)
        assert along / cross == pytest.approx(1.0, abs=0.25)

    def test_flanks_stay_within_the_reported_budget(self, scene, monkeypatch, capsys):
        """The budget the run reports must actually bound the flank reach, to
        within one cluster width of granularity."""
        monkeypatch.setattr(rg, 'USE_ANISOTROPIC_RINGS', True)
        _, ani = _run(scene)
        budget = float(capsys.readouterr().out
                       .split('cross-slope budget (')[1].split(' m)')[0])
        _, cross = _extent(scene, ani)
        assert cross <= budget + BLOCK

    def test_cross_budget_is_not_inflated_by_flank_growth(self, scene, monkeypatch):
        """Regression on the along-slope clock. Driving the budget from the
        global ring counter lets the flanks buy themselves time: measured in
        release-areas on Jan 18 with the lateral cap removed, that version
        dropped 1 cluster at a 132 m budget against 43 at 62 m once the clock
        was used.

        The clock is still not a pure fall-line measure: the ±45° sector counts
        a diagonal cluster as along-slope, so corner growth keeps it ticking
        after the fall-line front has stopped. Bounded here at 2x the ideal,
        which the self-inflating version blew through by a factor of ~2.5.
        """
        monkeypatch.setattr(rg, 'USE_ANISOTROPIC_RINGS', True)
        _, ani = _run(scene)
        along, cross = _extent(scene, ani)
        ratio = rg.MODE3_SPEED_RATIO / rg.MODE2_SPEED_RATIO
        assert cross < along
        assert cross <= 2.0 * ratio * along

    def test_a_zero_budget_is_refused_rather_than_hanging(self, scene, monkeypatch):
        """A zero step or ratio would defer every flank cluster forever."""
        monkeypatch.setattr(rg, 'USE_ANISOTROPIC_RINGS', True)
        monkeypatch.setattr(rg, 'RING_STEP_M', 0.0)
        monkeypatch.setattr(rg, 'MODE3_SPEED_RATIO', 0.0)
        with pytest.raises(ValueError, match='positive ring_step'):
            _run(scene)


class TestMelocheArrestPrefactor:
    """Eq. 20 carries C = 0.045. The inline copy in _qualifies dropped it, so
    the arrest length came out 22x too long and the test could not fire."""

    def test_arrest_length_is_the_shared_eq20_with_c(self):
        import arrest_indices as ai
        a, X = ai.arrest_length(800.0, 300.0, 2.0, 4000.0, delta=1.0, L_t=30.0)
        assert a == pytest.approx(
            ai.C_FIT * 30.0 * X * np.sqrt(4000.0 / 800.0))

    def test_stronger_wl_downslope_arrests_at_the_first_hop(self, scene, monkeypatch):
        """WL strength rises 300 Pa/m down the fall line. With C the arrest
        length is ~2.9 m, under the 4 m hop, so the crack must not enter any
        cluster downslope of the trigger. Without C it is ~63 m and runs on."""
        monkeypatch.setattr(rg, 'USE_MELOCHE_ARREST', True)
        px = scene['pixel_index']
        snap = scene['snap_features'].copy()
        for cid, (rows, _cols) in px.items():
            snap.loc[cid, 'wl_shear_strength'] = 2400.0 + 300.0 * rows.mean()
        sc = dict(scene, snap_features=snap)
        _poly, reached = _run(sc)
        t_rows, _ = px[scene['trigger']]
        downslope = [c for c in reached
                     if px[c][0].mean() > t_rows.mean() + BLOCK]
        assert downslope == []
        assert len(reached) > 1            # it still grew upslope and sideways

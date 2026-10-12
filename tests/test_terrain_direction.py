"""Tests for the terrain direction helpers in release_geometry.

Ported from ~/avalanche-release-areas/tests/test_terrain_direction.py, which
covers the two direction bugs fixed there on 2026-10-09. Neither had coverage
in this repo before.

Two invariants matter here, and both are about *direction*, not magnitude:

  1. compute_slope_aspect() must return the true compass bearing of steepest
     descent on a north-up raster, where row increases southward.
  2. find_stauchwall() must walk along that bearing. An integer
     dr=sign(-cos), dc=sign(sin) step reaches only 45/135/225/315 deg, so on
     an ESE slope it veers ~20 deg clockwise toward south and the release
     polygon rotates with it.

The lesson generalises: this package converts between pixel (row, col) and map
(east, north) constantly, and row increases SOUTHWARD. Any new code that
derives a direction must be checked against the map frame, never assumed from
the array axes.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from affine import Affine

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "avachain"))

from release_geometry import (  # noqa: E402
    GAUME_ASPECT_CAP,
    compute_slope_aspect,
    estimate_cross_slope_width,
    find_stauchwall,
    pixel_to_utm,
    project_along_aspect,
)

# North-up 1 m transform: col -> +easting, row -> -northing.
T = Affine(1.0, 0.0, 400000.0, 0.0, -1.0, 4000000.0)


def planar_dem(bearing_deg, slope_deg, n=201):
    """DEM of a perfect plane whose steepest descent runs along bearing_deg."""
    r, c = np.mgrid[0:n, 0:n]
    x = c * 1.0                      # easting, +col
    y = -r * 1.0                     # northing, -row
    b = np.radians(bearing_deg)
    # unit downslope direction in map frame
    e_hat, n_hat = np.sin(b), np.cos(b)
    # descend along (e_hat, n_hat)
    return -np.tan(np.radians(slope_deg)) * (x * e_hat + y * n_hat)


def ang_diff(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


class TestComputeSlopeAspect:
    @pytest.mark.parametrize("bearing", [0, 45, 90, 114, 135, 180, 225, 270, 315])
    def test_recovers_plane_bearing(self, bearing):
        dem = planar_dem(bearing, 30.0)
        slope, aspect = compute_slope_aspect(dem, 1.0)
        mid = (slice(5, -5), slice(5, -5))
        assert abs(ang_diff(float(np.mean(aspect[mid])), bearing)) < 1e-6
        assert float(np.mean(slope[mid])) == pytest.approx(30.0, abs=1e-6)

    def test_east_facing_slope_is_90_deg(self):
        # elevation falls toward +easting only
        r, c = np.mgrid[0:50, 0:50]
        _, aspect = compute_slope_aspect(-0.5 * c.astype(float), 1.0)
        assert float(aspect[10, 10]) == pytest.approx(90.0)

    def test_south_facing_slope_is_180_deg(self):
        # elevation falls toward -northing, i.e. toward +row
        r, c = np.mgrid[0:50, 0:50]
        _, aspect = compute_slope_aspect(-0.5 * r.astype(float), 1.0)
        assert float(aspect[10, 10]) == pytest.approx(180.0)


class TestFindStauchwall:
    @pytest.mark.parametrize("bearing", [114.0, 100.0, 120.0, 160.0, 20.0, 290.0])
    def test_walk_follows_aspect_not_the_diagonals(self, bearing):
        """The realised trigger->stauchwall bearing must match the aspect.

        The old sign()-quantised step could only produce 45/135/225/315, so
        this fails by up to 45 deg on any non-diagonal slope.
        """
        n = 201
        dem = planar_dem(bearing, 35.0, n)
        slope, aspect = compute_slope_aspect(dem, 1.0)
        # uniform 35 deg plane: never drops below threshold, so the walk runs
        # until it leaves the grid -> pure direction test
        r0 = c0 = n // 2
        r1, c1 = find_stauchwall(r0, c0, slope, aspect,
                                 threshold_deg=10.0, max_steps=60)
        x0, y0 = pixel_to_utm(r0, c0, T)
        x1, y1 = pixel_to_utm(r1, c1, T)
        realised = np.degrees(np.arctan2(x1 - x0, y1 - y0)) % 360.0
        assert abs(ang_diff(realised, bearing)) < 2.0

    def test_ese_slope_does_not_veer_south(self):
        """Regression: the Little Professor case. Aspect 114 deg (ESE) used to
        walk at exactly 135 deg (SE), a +21 deg clockwise error."""
        n = 201
        dem = planar_dem(114.0, 35.0, n)
        slope, aspect = compute_slope_aspect(dem, 1.0)
        r0 = c0 = n // 2
        r1, c1 = find_stauchwall(r0, c0, slope, aspect,
                                 threshold_deg=10.0, max_steps=60)
        x0, y0 = pixel_to_utm(r0, c0, T)
        x1, y1 = pixel_to_utm(r1, c1, T)
        realised = np.degrees(np.arctan2(x1 - x0, y1 - y0)) % 360.0
        assert abs(ang_diff(realised, 135.0)) > 15.0, "still snapped to the SE diagonal"
        assert abs(ang_diff(realised, 114.0)) < 2.0

    def test_stops_at_slope_threshold(self):
        slope = np.full((50, 50), 40.0)
        slope[:, 30:] = 10.0              # flat beyond col 30
        aspect = np.full((50, 50), 90.0)  # due east
        r, c = find_stauchwall(10, 10, slope, aspect, threshold_deg=28.0)
        assert (r, c) == (10, 30)

    def test_returns_immediately_when_already_below_threshold(self):
        slope = np.full((20, 20), 5.0)
        aspect = np.full((20, 20), 135.0)
        assert find_stauchwall(7, 8, slope, aspect, threshold_deg=28.0) == (7, 8)

    def test_stops_at_grid_edge_without_error(self):
        slope = np.full((20, 20), 40.0)
        aspect = np.full((20, 20), 90.0)   # east, will run off the right edge
        r, c = find_stauchwall(10, 10, slope, aspect, threshold_deg=28.0)
        assert 0 <= r < 20 and 0 <= c < 20

    def test_stops_at_nodata(self):
        slope = np.full((20, 20), 40.0)
        slope[:, 15:] = np.nan
        aspect = np.full((20, 20), 90.0)
        r, c = find_stauchwall(10, 10, slope, aspect, threshold_deg=28.0)
        assert (r, c) == (10, 14)

    def test_transform_argument_is_accepted_and_ignored(self):
        """Call sites in this repo pass `transform` positionally; the walk is in
        pixel units, so it must make no difference."""
        slope = np.full((50, 50), 40.0)
        slope[:, 30:] = 10.0
        aspect = np.full((50, 50), 90.0)
        assert (find_stauchwall(10, 10, slope, aspect, T, threshold_deg=28.0)
                == find_stauchwall(10, 10, slope, aspect, threshold_deg=28.0))


class TestProjectAlongAspect:
    def test_downslope_matches_aspect_bearing(self):
        x, y = project_along_aspect(0.0, 0.0, 100.0, 90.0, upslope=False)
        assert (x, y) == pytest.approx((100.0, 0.0), abs=1e-9)

    def test_upslope_is_opposite(self):
        x, y = project_along_aspect(0.0, 0.0, 100.0, 90.0, upslope=True)
        assert (x, y) == pytest.approx((-100.0, 0.0), abs=1e-9)

    def test_round_trip_on_a_plane_loses_elevation_downslope(self):
        dem = planar_dem(114.0, 30.0)
        _, aspect = compute_slope_aspect(dem, 1.0)
        r0 = c0 = 100
        a = float(aspect[r0, c0])
        # step 20 m downslope in pixel space and confirm the DEM drops
        dr = -np.cos(np.radians(a)) * 20.0
        dc = np.sin(np.radians(a)) * 20.0
        assert dem[int(r0 + dr), int(c0 + dc)] < dem[r0, c0]


# -----------------------------------------------------------------------
# Cross-slope theta sampling
# -----------------------------------------------------------------------

ASPECT = 114.0   # ESE, as on the Little Professor start zone


def _cross_slope_scene(tau_of_offset, aspect_deg=ASPECT, theta_down=3.0):
    """Build a cluster map with the trigger at centre, six neighbours on the
    cross-slope axis and six on the fall line, all 10/20/30 m out.

    tau_of_offset(de, dn) -> tau_p, so a test can make tau_p vary along
    exactly one of the two axes.
    """
    a = np.radians(aspect_deg)
    fall  = np.array([np.sin(a),  np.cos(a)])      # (east, north)
    cross = np.array([-fall[1], fall[0]])

    n = 121
    cluster_map = np.zeros((n, n), float)
    r0 = c0 = n // 2
    cluster_map[r0, c0] = 1
    taus = {1: tau_of_offset(0.0, 0.0)}

    cid = 2
    for axis in (cross, fall):
        for s in (-30.0, -20.0, -10.0, 10.0, 20.0, 30.0):
            de, dn = axis * s
            # east -> +col, north -> -row
            r, c = int(round(r0 - dn)), int(round(c0 + de))
            cluster_map[r, c] = cid
            taus[cid] = tau_of_offset(de, dn)
            cid += 1

    df = pd.DataFrame({
        'theta': theta_down,
        'tau_p': [taus[k] for k in sorted(taus)],
    }, index=sorted(taus))
    df.index.name = 'cluster_id'
    T_ = Affine(1.0, 0.0, 400000.0, 0.0, -1.0, 4000000.0)
    return cluster_map, df, T_, fall, cross


class TestEstimateCrossSlopeWidth:
    def test_fall_line_neighbours_are_excluded(self):
        """tau_p varying only along the fall line must leave theta_cross at 0.

        The row/col selection used before picked neighbours a mean 30 deg off
        the cross axis -- some within 1.2 deg of the fall line -- so a purely
        along-slope gradient leaked into theta_cross.
        """
        a = np.radians(ASPECT)
        fall = np.array([np.sin(a), np.cos(a)])
        cmap, df, T_, fall, cross = _cross_slope_scene(
            lambda de, dn: 1000.0 + 50.0 * (np.array([de, dn]) @ fall))
        got = estimate_cross_slope_width(
            1, A_ca=40.0, meloche_df=df, cluster_map=cmap, transform=T_,
            aspect_deg=ASPECT)
        # theta_cross == 0 -> fall back to the aspect cap, not a leaked gradient
        assert got == pytest.approx(40.0 * GAUME_ASPECT_CAP)

    def test_cross_slope_gradient_is_recovered_exactly(self):
        g = 2.0          # Pa/m along the cross axis
        theta_down = 3.0
        a = np.radians(ASPECT)
        fall = np.array([np.sin(a), np.cos(a)])
        cross = np.array([-fall[1], fall[0]])
        cmap, df, T_, _, _ = _cross_slope_scene(
            lambda de, dn: 1000.0 + g * (np.array([de, dn]) @ cross),
            theta_down=theta_down)
        got = estimate_cross_slope_width(
            1, A_ca=40.0, meloche_df=df, cluster_map=cmap, transform=T_,
            aspect_deg=ASPECT)
        # |dtau|/d == g on every cross-axis pair, so width_factor = theta_down/g
        assert got == pytest.approx(40.0 * (theta_down / g), rel=0.02)

    def test_no_aspect_means_no_theta_widening(self):
        cmap, df, T_, _, _ = _cross_slope_scene(lambda de, dn: 1000.0)
        got = estimate_cross_slope_width(
            1, A_ca=40.0, meloche_df=df, cluster_map=cmap, transform=T_,
            aspect_deg=None)
        assert got == pytest.approx(40.0)

    def test_width_factor_is_capped(self):
        cmap, df, T_, _, _ = _cross_slope_scene(
            lambda de, dn: 1000.0, theta_down=1e6)
        got = estimate_cross_slope_width(
            1, A_ca=40.0, meloche_df=df, cluster_map=cmap, transform=T_,
            aspect_deg=ASPECT)
        assert got == pytest.approx(40.0 * GAUME_ASPECT_CAP)

    def test_both_flanks_contribute(self):
        """The sector test uses |d.cross|, so neighbours on either flank count."""
        a = np.radians(ASPECT)
        fall = np.array([np.sin(a), np.cos(a)])
        cross = np.array([-fall[1], fall[0]])
        # antisymmetric field: +g one flank, -g the other, |gradient| = g both ways
        cmap, df, T_, _, _ = _cross_slope_scene(
            lambda de, dn: 1000.0 + 2.0 * abs(np.array([de, dn]) @ cross),
            theta_down=3.0)
        got = estimate_cross_slope_width(
            1, A_ca=40.0, meloche_df=df, cluster_map=cmap, transform=T_,
            aspect_deg=ASPECT)
        assert got == pytest.approx(40.0 * (3.0 / 2.0), rel=0.02)

    def test_numerator_is_measured_along_slope_not_read_from_the_csv(self):
        """theta_along must come from the fall-line sector, so a wildly wrong
        CSV theta cannot affect the width."""
        a = np.radians(ASPECT)
        fall = np.array([np.sin(a), np.cos(a)])
        cross = np.array([-fall[1], fall[0]])
        # 4 Pa/m along the fall line, 2 Pa/m across it
        cmap, df, T_, _, _ = _cross_slope_scene(
            lambda de, dn: (1000.0
                            + 4.0 * (np.array([de, dn]) @ fall)
                            + 2.0 * (np.array([de, dn]) @ cross)),
            theta_down=1e4)          # absurd CSV value; must be ignored
        got = estimate_cross_slope_width(
            1, A_ca=40.0, meloche_df=df, cluster_map=cmap, transform=T_,
            aspect_deg=ASPECT)
        assert got == pytest.approx(40.0 * (4.0 / 2.0), rel=0.02)

    def test_ratio_is_invariant_to_a_uniform_tau_p_rescale(self):
        """Both halves of the ratio are the same estimator, so scaling tau_p
        scales numerator and denominator alike."""
        a = np.radians(ASPECT)
        fall = np.array([np.sin(a), np.cos(a)])
        cross = np.array([-fall[1], fall[0]])

        def scene(k):
            return _cross_slope_scene(
                lambda de, dn: 1000.0 + k * (3.0 * (np.array([de, dn]) @ fall)
                                             + 2.0 * (np.array([de, dn]) @ cross)),
                theta_down=1e4)

        outs = []
        for k in (1.0, 7.0):
            cmap, df, T_, _, _ = scene(k)
            outs.append(estimate_cross_slope_width(
                1, A_ca=40.0, meloche_df=df, cluster_map=cmap, transform=T_,
                aspect_deg=ASPECT))
        assert outs[0] == pytest.approx(outs[1], rel=1e-9)
        assert outs[0] == pytest.approx(40.0 * 1.5, rel=0.02)

    def test_falls_back_to_csv_theta_when_no_along_slope_neighbour(self):
        """tau_p constant along the fall line leaves theta_along at 0, so the
        CSV value is the documented last resort."""
        a = np.radians(ASPECT)
        fall = np.array([np.sin(a), np.cos(a)])
        cross = np.array([-fall[1], fall[0]])
        cmap, df, T_, _, _ = _cross_slope_scene(
            lambda de, dn: 1000.0 + 2.0 * (np.array([de, dn]) @ cross),
            theta_down=3.0)
        got = estimate_cross_slope_width(
            1, A_ca=40.0, meloche_df=df, cluster_map=cmap, transform=T_,
            aspect_deg=ASPECT)
        assert got == pytest.approx(40.0 * (3.0 / 2.0), rel=0.02)

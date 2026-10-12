"""Tests for the two spatial shear-gradient estimators.

Ported from ~/avalanche-release-areas/tests/test_theta_estimators.py.

The invariant that matters: θ is a function of the lag it is measured at. A
short-lag difference estimator ('knn') reports mostly local noise; a plane fit
over a neighbourhood of radius ~L_ss reports the slope-scale ramp, which is
what Meloche et al.'s θ represents.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "avachain"))

import snowpack_analysis as sa                              # noqa: E402
from snowpack_analysis import theta_knn, theta_plane_fit     # noqa: E402


def _grid(step=4.0, extent=60.0):
    e, n = np.meshgrid(np.arange(0.0, extent, step), np.arange(0.0, extent, step))
    return e.ravel(), n.ravel()


class TestThetaKnn:
    def test_matches_mean_abs_difference_over_distance(self):
        rng = np.random.default_rng(1)
        n = 40
        tau = rng.normal(2400.0, 200.0, n)
        cents = rng.uniform(0.0, 60.0, (n, 2))
        nb = NearestNeighbors(n_neighbors=7, algorithm='ball_tree').fit(cents)
        dist, idx = nb.kneighbors(cents)

        got = theta_knn(tau, idx, dist, 1.0)
        exp = [np.mean([abs(tau[i] - tau[j]) / d
                        for j, d in zip(idx[i][1:], dist[i][1:]) if d >= 1e-6])
               for i in range(n)]
        assert got == pytest.approx(exp, nan_ok=True)

    def test_px_m_scales_theta_inversely(self):
        """The bug this fixes: without px_m, theta is Pa per *pixel*, which only
        coincides with Pa/m on a 1 m grid."""
        rng = np.random.default_rng(2)
        tau = rng.normal(2400.0, 200.0, 30)
        cents = rng.uniform(0.0, 60.0, (30, 2))
        nb = NearestNeighbors(n_neighbors=7, algorithm='ball_tree').fit(cents)
        dist, idx = nb.kneighbors(cents)
        a = theta_knn(tau, idx, dist, 1.0)
        b = theta_knn(tau, idx, dist, 2.0)
        assert b == pytest.approx(np.asarray(a) / 2.0, nan_ok=True)

    def test_nan_tau_gives_nan_theta(self):
        tau = np.array([100.0, np.nan, 300.0, 400.0])
        cents = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        nb = NearestNeighbors(n_neighbors=4, algorithm='ball_tree').fit(cents)
        dist, idx = nb.kneighbors(cents)
        out = theta_knn(tau, idx, dist, 1.0)
        assert np.isnan(out[1])
        assert not np.isnan(out[0])

    def test_nan_neighbour_is_skipped_not_propagated(self):
        tau = np.array([100.0, np.nan, 120.0])
        cents = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        nb = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(cents)
        dist, idx = nb.kneighbors(cents)
        out = theta_knn(tau, idx, dist, 1.0)
        assert out[0] == pytest.approx(10.0)   # |100-120|/2, NaN pair dropped

    def test_uniform_field_gives_zero(self):
        cents = np.random.default_rng(3).uniform(0, 50, (20, 2))
        nb = NearestNeighbors(n_neighbors=7, algorithm='ball_tree').fit(cents)
        dist, idx = nb.kneighbors(cents)
        assert theta_knn(np.full(20, 2400.0), idx, dist, 1.0) == pytest.approx(0.0)


class TestThetaPlaneFit:
    def test_recovers_an_exact_planar_ramp(self):
        e, n = _grid()
        theta, g_e, g_n = theta_plane_fit(e, n, 1000.0 + 3.0 * e - 4.0 * n,
                                          radius_m=20.0, min_neighbours=8)
        m = ~np.isnan(theta)
        assert m.sum() > 50
        assert theta[m] == pytest.approx(5.0)        # hypot(3, 4)
        assert g_e[m] == pytest.approx(3.0)
        assert g_n[m] == pytest.approx(-4.0)

    def test_uniform_field_gives_zero_gradient(self):
        e, n = _grid()
        theta, _, _ = theta_plane_fit(e, n, np.full(e.size, 2400.0),
                                      radius_m=20.0, min_neighbours=8)
        m = ~np.isnan(theta)
        assert theta[m] == pytest.approx(0.0, abs=1e-9)

    def test_suppresses_noise_that_knn_reports(self):
        """The point of the estimator: on ramp + noise, the plane fit returns
        the ramp while a 3 m-lag difference estimator returns mostly noise."""
        rng = np.random.default_rng(4)
        e, n = _grid(step=3.0, extent=60.0)
        ramp = 5.0
        tau = 2400.0 + ramp * e + rng.normal(0.0, 70.0, e.size)

        theta_pf, _, _ = theta_plane_fit(e, n, tau, radius_m=20.0,
                                         min_neighbours=8)
        cents = np.column_stack([n, e])          # (row, col)-like for knn
        nb = NearestNeighbors(n_neighbors=7, algorithm='ball_tree').fit(cents)
        dist, idx = nb.kneighbors(cents)
        theta_kn = theta_knn(tau, idx, dist, 1.0)

        pf = np.nanmedian(theta_pf)
        kn = np.nanmedian(theta_kn)
        assert pf == pytest.approx(ramp, rel=0.25), f"plane fit {pf}"
        assert kn > 2.0 * pf, f"knn {kn} should be noise-inflated above {pf}"

    def test_too_few_neighbours_gives_nan(self):
        e = np.array([0.0, 1.0, 2.0])
        n = np.array([0.0, 0.0, 0.0])
        theta, _, _ = theta_plane_fit(e, n, np.array([1.0, 2.0, 3.0]),
                                      radius_m=20.0, min_neighbours=8)
        assert np.all(np.isnan(theta))

    def test_collinear_neighbourhood_gives_nan(self):
        # all points on one line: rank 2, no plane
        e = np.arange(12.0)
        n = np.zeros(12)
        theta, _, _ = theta_plane_fit(e, n, 3.0 * e, radius_m=50.0,
                                      min_neighbours=8)
        assert np.all(np.isnan(theta))

    def test_nan_tau_excluded_but_others_still_fit(self):
        e, n = _grid()
        tau = 1000.0 + 3.0 * e - 4.0 * n
        tau[:5] = np.nan
        theta, _, _ = theta_plane_fit(e, n, tau, radius_m=20.0,
                                      min_neighbours=8)
        assert np.all(np.isnan(theta[:5]))
        m = ~np.isnan(theta)
        assert theta[m] == pytest.approx(5.0)

    def test_radius_defaults_come_from_the_module_constants(self):
        e, n = _grid()
        tau = 1000.0 + 3.0 * e - 4.0 * n
        a, _, _ = theta_plane_fit(e, n, tau)
        b, _, _ = theta_plane_fit(e, n, tau,
                                  radius_m=sa.THETA_FIT_RADIUS_M,
                                  min_neighbours=sa.THETA_FIT_MIN_NEIGHBOURS)
        assert a == pytest.approx(b, nan_ok=True)


class TestEstimatorDefault:
    def test_plane_fit_is_the_default(self):
        """Default since 2026-10-09, argued from the L_ss calibration scale.
        'knn' is retained to reproduce every pre-change CSV."""
        assert sa.THETA_ESTIMATOR == 'plane_fit'

    def test_fit_radius_is_the_propagation_length(self):
        """The whole argument for 'plane_fit' is that the neighbourhood matches
        the scale Meloche's ramp was calibrated over."""
        assert sa.THETA_FIT_RADIUS_M == sa.L_SS

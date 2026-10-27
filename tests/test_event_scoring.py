"""Tests for event_scoring: the release-probability map and its scores.

Synthetic grids only; none of this needs the Jan 18 data, and the
date-matching trap (a February day scored against Jan 18's polygon) is
exactly the kind of thing Jan 18 cannot see.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from rasterio.transform import from_origin
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "avachain"))

import event_scoring as es  # noqa: E402

T = from_origin(0.0, 10.0, 1.0, 1.0)      # 10 x 10 grid, 1 m pixels, north-up
SHAPE = (10, 10)


def _scen(rows):
    return pd.DataFrame(rows, columns=['trigger_cluster', 'size_factor', 'depth_percentile',
                                       'weight', 'geometry'])


class TestReleaseProbability:
    def test_overlapping_polygons_sum_their_weights(self):
        scen = _scen([(1, 1.0, 50, 0.6, box(0, 0, 5, 10)),
                      (2, 1.0, 50, 0.4, box(3, 0, 10, 10))])
        p = es.release_probability(scen, SHAPE, T)
        assert p[0, 0] == pytest.approx(0.6)
        assert p[0, 4] == pytest.approx(1.0)       # covered by both
        assert p[0, 9] == pytest.approx(0.4)

    def test_depth_percentiles_share_one_polygon(self):
        g = box(0, 0, 5, 10)
        scen = _scen([(1, 1.0, d, w, g) for d, w in ((10, .2), (50, .6), (90, .2))])
        p = es.release_probability(scen, SHAPE, T)
        assert p.max() == pytest.approx(1.0)
        assert p.sum() == pytest.approx(50.0)

    def test_empty_polygon_contributes_nothing(self):
        from shapely.geometry import Polygon
        scen = _scen([(1, 1.0, 50, 1.0, Polygon())])
        assert es.release_probability(scen, SHAPE, T).sum() == 0.0


class TestAuc:
    def test_perfect_inverted_and_uninformative(self):
        o = np.array([0, 0, 1, 1])
        assert es.auc(np.array([.1, .2, .8, .9]), o) == pytest.approx(1.0)
        assert es.auc(np.array([.9, .8, .2, .1]), o) == pytest.approx(0.0)
        assert es.auc(np.full(4, .5), o) == pytest.approx(0.5)

    def test_constant_truth_is_nan(self):
        assert np.isnan(es.auc(np.array([.1, .9]), np.array([0, 0])))


class TestScore:
    sz = np.zeros(SHAPE, bool)
    sz[:, :8] = True

    def test_no_event_brier_is_mean_square_probability(self):
        p = np.zeros(SHAPE)
        p[:, :2] = 0.5
        out = es.score(p, self.sz)
        assert out['brier'] == pytest.approx(np.mean(p[self.sz] ** 2))
        assert out['expected_area_m2'] == pytest.approx(10.0)
        assert out['any_prob_area_m2'] == pytest.approx(20.0)
        assert 'auc' not in out

    def test_perfect_forecast(self):
        obs = np.zeros(SHAPE, bool)
        obs[:5, :4] = True
        out = es.score(obs.astype(float), self.sz, obs)
        assert out['brier'] == pytest.approx(0.0)
        assert out['brier_skill'] == pytest.approx(1.0)
        assert out['auc'] == pytest.approx(1.0)
        assert out['observed_with_any_prob'] == pytest.approx(1.0)
        assert out['expected_false_m2'] == pytest.approx(0.0)

    def test_observed_outside_start_zone_counts_as_missed(self):
        obs = np.zeros(SHAPE, bool)
        obs[:5, 6:] = True                     # cols 8-9 lie outside the start zone
        out = es.score(obs & self.sz, self.sz, obs)
        assert out['observed_outside_start_zone_m2'] == pytest.approx(10.0)
        assert out['observed_with_any_prob'] == pytest.approx(0.5)
        assert out['brier'] > 0

    def test_trigger_weight_inside(self):
        obs = np.zeros(SHAPE, bool)
        obs[:2] = True
        out = es.score(np.zeros(SHAPE), self.sz, obs,
                       trigger_weights={1: .7, 2: .3}, trigger_inside={1: True, 2: False})
        assert out['trigger_weight_inside'] == pytest.approx(0.7)


class TestObservedOn:
    """cfg.release_geojsons_for_date returns every polygon ON OR BEFORE the
    date; observed_on must not turn Jan 18's into February's."""

    class _Cfg:
        def release_geojsons_for_date(self, date):
            return [Path('data/validation/avalanche_release_area_20260118.geojson')]

    def test_later_day_is_no_event(self):
        assert es.observed_on(self._Cfg(), '2026-02-15') is None
        assert es.observed_on(self._Cfg(), '2026-01-19') is None

"""
test_road_approach.py - the report's "How close the flow comes to the
road" curve (cycle_products.road_approach_curve).

It replaced the runout exceedance plot on 2026-11-02. That plot put each
scenario's runout (from its own release, in any direction) against one
release-to-road line, so on Jan 18 the curve passed the road while no
scenario reached it.

    cd runout && .venv/bin/python -m unittest tests.test_road_approach -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts import cycle_products as cp  # noqa: E402


class Curve(unittest.TestCase):

    def test_nothing_reaches_means_zero_at_the_road(self):
        df = pd.DataFrame({'distance_to_road_m': [37.3, 120.0, 400.0],
                           'reached_road': [False, False, False],
                           'scenario_probability': [0.1, 0.3, 0.6]})
        xs, ys = cp.road_approach_curve(df)
        self.assertEqual(ys[0], 0.0)
        self.assertEqual(list(xs), [0.0, 37.3, 120.0, 400.0])
        np.testing.assert_allclose(ys[1:], [0.1, 0.4, 1.0])

    def test_reached_counts_at_zero_whatever_the_distance_column_says(self):
        df = pd.DataFrame({'distance_to_road_m': [15.0, 200.0],
                           'reached_road': [True, False],
                           'scenario_probability': [0.25, 0.75]})
        xs, ys = cp.road_approach_curve(df)
        self.assertAlmostEqual(ys[0], 0.25)

    def test_distances_to_the_crossing_replace_the_any_road_column(self):
        # 2026-11-02: report distances are to the path's road crossing
        df = pd.DataFrame({'scenario_id': ['a', 'b'],
                           'distance_to_road_m': [400.0, 40.0],
                           'reached_road': [False, False],
                           'scenario_probability': [0.5, 0.5]})
        xs, ys = cp.road_approach_curve(df, {'a': 620.0, 'b': 41.0})
        self.assertEqual(list(xs), [0.0, 41.0, 620.0])

    def test_nearest_miss_by_crossing_distance(self):
        df = pd.DataFrame({'scenario_id': ['a', 'b', 'c'],
                           'reached_road': [False, False, True],
                           'scenario_probability': [0.2, 0.3, 0.5]})
        hz = {'df': df, 'road': {'nearest_miss': {'scenario': 'a', 'distance_short_m': 10.0}}}
        nm = cp.nearest_miss_with_rank(hz, {'a': 300.0, 'b': 45.0, 'c': 0.0})
        self.assertEqual(nm['scenario'], 'b')          # c reached, so not a miss
        self.assertEqual(nm['distance_short_m'], 45.0)
        self.assertEqual(nm['rank'], 2)
        # without distances: the stack's any-road nearest miss, as before
        self.assertEqual(cp.nearest_miss_with_rank(hz)['scenario'], 'a')

    def test_no_table(self):
        self.assertIsNone(cp.road_approach_curve(pd.DataFrame()))


if __name__ == '__main__':
    unittest.main()

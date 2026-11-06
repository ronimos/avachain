"""
test_absolute_weights.py - forced explosive-shot sets carry absolute
weights (Ron, 2026-11-06): each shot holds 1/N, and a shot that does not
release counts as no reach. The release side writes
ensemble_weight_total = 1.0 into params.json; every weighted fraction then
divides by it instead of by the sum of the weights present. Model sets
carry none and keep the old renormalisation.

    cd runout && .venv/bin/python -m unittest tests.test_absolute_weights -v
"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
# the simulation modules import their siblings flat (they run from scripts/)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
import avalanche_stack as st  # noqa: E402
import avalanche_helpers as h  # noqa: E402
from scripts import cycle_products as cp  # noqa: E402


def _summ(sid, reached, dist=50.0):
    return {'scenario_id': sid,
            'road': {'reached': reached, 'distance_to_road_m': 0.0 if reached else dist,
                     'intersection_xy': [0, 0], 'flow_depth_at_road_m': 1.0,
                     'velocity_at_road_ms': 5.0, 'chainage_m': 0.0}}


# 4 shots, one did not release: the three that did carry 0.25 each
SHOTS = [{'scenario_id': f's{i}', 'params': {'scenario_probability': 0.25,
                                             'ensemble_weight_total': 1.0}}
         for i in range(3)]
MODEL = [{'scenario_id': f's{i}', 'params': {'scenario_probability': 0.25}}
         for i in range(3)]


class Total(unittest.TestCase):
    def test_present_and_absent(self):
        self.assertEqual(st.ensemble_weight_total(s['params'] for s in SHOTS), 1.0)
        self.assertIsNone(st.ensemble_weight_total(s['params'] for s in MODEL))


class RoadReach(unittest.TestCase):
    summaries = [_summ('s0', True), _summ('s1', False), _summ('s2', False)]

    def test_shots_divide_by_the_fixed_total(self):
        with tempfile.TemporaryDirectory() as d:
            r = st.build_road_impact_summary(self.summaries, Path(d) / 'r.json',
                                             scenarios=SHOTS, use_weights=True,
                                             weight_total=1.0)
        self.assertAlmostEqual(r['fraction_reached_weighted'], 0.25)

    def test_model_renormalises_as_before(self):
        with tempfile.TemporaryDirectory() as d:
            r = st.build_road_impact_summary(self.summaries, Path(d) / 'r.json',
                                             scenarios=MODEL, use_weights=True)
        self.assertAlmostEqual(r['fraction_reached_weighted'], 1 / 3, places=4)


class Raster(unittest.TestCase):
    def test_probability_raster_uses_the_fixed_total(self):
        import rasterio
        from rasterio.transform import from_origin
        with tempfile.TemporaryDirectory() as d:
            paths = []
            for i, v in enumerate([1, 1, 0]):
                p = Path(d) / f'm{i}.tif'
                with rasterio.open(p, 'w', driver='GTiff', height=2, width=2, count=1,
                                   dtype='uint8', crs='EPSG:6342',
                                   transform=from_origin(0, 2, 1, 1)) as dst:
                    dst.write(np.full((1, 2, 2), v, dtype='uint8'))
                paths.append(p)
            out = Path(d) / 'p.tif'
            st.stack_reach_masks(paths, out, weights=[0.25] * 3, total=1.0)
            prob, _ = h.read_raster(out)
            np.testing.assert_allclose(prob, 0.5)
            st.stack_reach_masks(paths, out, weights=[0.25] * 3)
            prob, _ = h.read_raster(out)
            np.testing.assert_allclose(prob, 2 / 3, rtol=1e-6)


class ReportCurves(unittest.TestCase):
    def test_road_approach_tops_out_at_p_release(self):
        df = pd.DataFrame({'distance_to_road_m': [10.0, 100.0, 300.0],
                           'reached_road': [True, False, False],
                           'scenario_probability': [0.25] * 3,
                           'ensemble_weight_total': [1.0] * 3})
        xs, ys = cp.road_approach_curve(df)
        self.assertAlmostEqual(ys[0], 0.25)
        self.assertAlmostEqual(ys[-1], 0.75)        # 0.25 is no release

    def test_model_curve_unchanged(self):
        df = pd.DataFrame({'distance_to_road_m': [10.0, 100.0],
                           'reached_road': [False, False],
                           'scenario_probability': [0.25, 0.25]})
        xs, ys = cp.road_approach_curve(df)
        self.assertAlmostEqual(ys[-1], 1.0)


if __name__ == '__main__':
    unittest.main()

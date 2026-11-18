"""
test_merged_scenarios.py - avachain's runout_step runs scenarios with
identical inputs once, with the group's summed scenario_probability
(Ron, 2026-11-18). com1DFA gives identical output for identical input, so
each run's reach mask stands for the whole group. These tests pin, on this
package's own stacking code, that one mask with the summed weight gives the
same weighted products as the group's masks with their own weights: the
P(reach) raster, the weighted road and reference-point fractions, for model
sets (sum of weights) and shot sets (ensemble_weight_total).

    cd runout && .venv/bin/python -m unittest tests.test_merged_scenarios -v
"""
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
import avalanche_stack as st  # noqa: E402

PROFILE = dict(driver='GTiff', height=8, width=8, count=1, dtype='uint8',
               crs='EPSG:6342', transform=from_origin(0, 8, 1, 1))


def _summ(sid, reached, dist):
    return {'scenario_id': sid,
            'road': {'reached': reached, 'distance_to_road_m': 0.0 if reached else dist,
                     'intersection_xy': [0, 0], 'flow_depth_at_road_m': 1.0,
                     'velocity_at_road_ms': 5.0, 'chainage_m': 0.0},
            'reference_points': [{'ref_name': 'tower', 'reached': reached,
                                  'distance_short_m': 0.0 if reached else dist}]}


class TestMergedGroups(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        rng = np.random.default_rng(0)
        # 4 groups of 5 identical runouts, each member its own weight
        self.full, self.merged = [], []     # (mask path, scenario dict, summary)
        for g in range(4):
            mask = rng.integers(0, 2, (8, 8), dtype='uint8')
            path = self.tmp / f'g{g}.tif'
            with rasterio.open(path, 'w', **PROFILE) as dst:
                dst.write(mask, 1)
            ws = rng.uniform(0.001, 0.05, 5)
            reached = g % 2 == 0
            for k, w in enumerate(ws):
                sid = f'scenario_{g * 5 + k + 1:03d}'
                self.full.append((path, {'scenario_id': sid,
                                         'params': {'scenario_probability': float(w)}},
                                  _summ(sid, reached, 10.0 + g)))
            rep = self.full[-5]
            self.merged.append((path, {'scenario_id': rep[1]['scenario_id'],
                                       'params': {'scenario_probability':
                                                  math.fsum(float(w) for w in ws),
                                                  'merged_scenarios':
                                                  [r[1]['scenario_id'] for r in self.full[-5:]]}},
                                rep[2]))

    def _with_total(self, rows, total):
        if total is None:
            return rows
        return [(p, {**s, 'params': {**s['params'], 'ensemble_weight_total': total}}, m)
                for p, s, m in rows]

    def _products(self, rows, tag):
        scen = [s for _, s, _ in rows]
        summ = [m for _, _, m in rows]
        tot = st.ensemble_weight_total(s['params'] for s in scen)
        out = self.tmp / f'p_{tag}.tif'
        st.stack_reach_masks([p for p, _, _ in rows], out,
                             weights=[s['params']['scenario_probability'] for s in scen],
                             total=tot)
        road = st.build_road_impact_summary(summ, self.tmp / f'r_{tag}.json', scenarios=scen,
                                            use_weights=True, weight_total=tot)
        ref = st.build_reference_point_summary(summ, self.tmp / f'f_{tag}.json',
                                               scenarios=scen, use_weights=True,
                                               weight_total=tot)
        with rasterio.open(out) as src:
            return src.read(1), road, ref

    def test_model_and_shot_products_unchanged(self):
        for total in (None, 1.0):
            with self.subTest(total=total):
                pa, ra, fa = self._products(self._with_total(self.full, total), f'a{total}')
                pb, rb, fb = self._products(self._with_total(self.merged, total), f'b{total}')
                np.testing.assert_allclose(pa, pb, rtol=0, atol=1e-6)
                self.assertAlmostEqual(ra['fraction_reached_weighted'],
                                       rb['fraction_reached_weighted'], places=12)
                for name in fa:
                    self.assertAlmostEqual(fa[name]['fraction_reached_weighted'],
                                           fb[name]['fraction_reached_weighted'], places=12)
                # the counts are of distinct runouts now
                self.assertEqual((ra['n_scenarios'], rb['n_scenarios']), (20, 4))
                self.assertEqual((ra['n_scenarios_represented'],
                                  rb['n_scenarios_represented']), (20, 20))
                self.assertEqual(ra['fraction_reached'], rb['fraction_reached'])


class TestLabels(unittest.TestCase):
    def test_runs_label(self):
        self.assertEqual(st.runs_label(15, 75), '15 distinct runouts (75 scenarios)')
        self.assertEqual(st.runs_label(75, 75), '75 scenarios')
        self.assertEqual(st.runs_label(75, None), '75 scenarios')

    def test_cycle_table_cell(self):
        from scripts import cycle_products as cp
        self.assertEqual(cp._n_label({'n_scenarios': 15, 'n_scenarios_represented': 75}),
                         '15 runouts (75)')
        self.assertEqual(cp._n_label({'n_scenarios': 75}), '75')
        self.assertEqual(cp._n_label({}), '-')


if __name__ == '__main__':
    unittest.main()

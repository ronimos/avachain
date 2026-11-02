"""
test_report_interactive.py - the data behind the interactive report map
and histogram (scripts/report_interactive.py).

    cd runout && .venv/bin/python -m unittest tests.test_report_interactive -v
"""
import base64
import gzip
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
import geopandas as gpd
from shapely.geometry import LineString

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
import report_interactive as ri  # noqa: E402

CRS = 'EPSG:6342'


def _tif(path, arr, nodata):
    with rasterio.open(path, 'w', driver='GTiff', width=arr.shape[1], height=arr.shape[0],
                       count=1, dtype='float32', crs=CRS, nodata=nodata,
                       transform=from_origin(1000.0, 2000.0, 1.0, 1.0)) as d:
        d.write(arr.astype('float32'), 1)


class Payload(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        hdir = self.tmp / 'T_0'
        hdir.mkdir()
        self.prob = np.zeros((40, 30), dtype='float32')
        self.prob[5:20, 10:15] = 0.4321
        self.prob[0, 0] = -1.0                      # nodata
        _tif(hdir / 'probability.tif', self.prob, -1.0)
        yy, xx = np.mgrid[0:40, 0:30]
        _tif(self.tmp / 'dem.tif', 3000.0 - yy * 0.7 + xx * 0.1, -10000.0)
        road = self.tmp / 'road.geojson'
        gpd.GeoDataFrame(geometry=[LineString([(990, 1965), (1040, 1965)])],
                         crs=CRS).to_file(road, driver='GeoJSON')
        df = pd.DataFrame({'scenario_id': ['s1', 's2'], 'scenario_probability': [0.7, 0.3],
                           'reached_road': [False, True], 'distance_to_road_m': [12.5, 0.0],
                           'runout_max_m': [100.0, 150.0]})
        self.hz = {'label': 'T_0', 'name': 'T+0', 'dir': hdir, 'df': df, 'dem': self.tmp / 'dem.tif',
                   'config': {'paths': {'road': str(road)}},
                   'road': {'fraction_reached_weighted': 0.3, 'n_scenarios': 2, 'n_reached': 1,
                            'impact_points': [{'xy': [1012.0, 1965.0]}]},
                   'manifest': {'valid_date': '2026-01-18'}}

    def test_three_days_even_with_one_horizon(self):
        p = ri.build_payload([self.hz], '20260118')
        self.assertEqual([s['label'] for s in p['slots']], ['T_0', 'T_1', 'T_2'])
        self.assertEqual([s['valid'] for s in p['slots']],
                         ['2026-01-18', '2026-01-19', '2026-01-20'])
        self.assertTrue(p['slots'][1]['missing'] and p['slots'][2]['missing'])
        self.assertNotIn('missing', p['slots'][0])

    def test_probability_round_trips_for_the_readout(self):
        p = ri.build_payload([self.hz], '20260118')
        raw = gzip.decompress(base64.b64decode(p['slots'][0]['prob_gz']))
        back = np.frombuffer(raw, dtype='<u2').reshape(40, 30) / p['p_scale']
        expect = np.where(self.prob > 0, self.prob, 0.0)    # nodata reads as 0
        self.assertLess(np.abs(back - expect).max(), 0.5 / p['p_scale'] + 1e-9)
        self.assertEqual(p['grid'], {'left': 1000.0, 'top': 2000.0, 'res': 1.0, 'w': 30, 'h': 40})

    def test_histogram_inputs(self):
        s = ri.build_payload([self.hz], '20260118')['slots'][0]
        by = {q['id']: q for q in s['scenarios']}
        self.assertEqual(by['s2']['d'], 0.0)           # reached -> 0
        self.assertTrue(by['s2']['reached'])
        self.assertEqual(by['s1']['d'], 12.5)
        self.assertAlmostEqual(by['s1']['w'], 0.7)

    def test_road_and_extent(self):
        p = ri.build_payload([self.hz], '20260118')
        lines = p['roads'][self.hz['config']['paths']['road']]
        self.assertEqual(lines, [[[990.0, 1965.0], [1040.0, 1965.0]]])
        # the extent takes in the road where it runs past the raster
        self.assertLessEqual(p['extent'][0], 990.0)
        self.assertGreaterEqual(p['extent'][2], 1040.0)

    def test_no_horizons(self):
        self.assertIsNone(ri.build_payload([], '20260118'))
        self.assertEqual(ri.render_parts(None)['imap_html'], '')


class AvalanchePath(unittest.TestCase):
    """The path to the road on a synthetic slope: a south-facing plane
    (grid north-up, 1 m cells), road east-west across the bottom."""

    def _run(self, ditch):
        tmp = Path(tempfile.mkdtemp())
        H, W = 200, 120
        yy, xx = np.mgrid[0:H, 0:W]
        dem = 2000.0 - yy * 0.6 + np.abs(xx - 60) * 0.05
        if ditch:
            # road cut: a trench 5-9 m above the road (y = 1825) that drains
            # east - the trap the raw steepest descent fell into at Little
            # Professor
            dem[166:170, :] -= 3.0 + np.linspace(0.0, 4.0, W)[None, :]
        _tif(tmp / 'dem.tif', dem, -10000.0)
        pmax = np.zeros((H, W)); pmax[10:60, 50:70] = 0.9
        road = [[[950.0, 1825.0], [1170.0, 1825.0]]]
        return ri.avalanche_path(tmp / 'dem.tif', pmax, road)

    def test_crosses_straight_below_the_start(self):
        # start = highest cell of the P >= 0.5 core: row 10, col 50
        r = self._run(ditch=False)
        self.assertEqual(r['how'], 'track direction projected to the road')
        self.assertAlmostEqual(r['crossing'][1], 1825.0, delta=0.01)
        self.assertAlmostEqual(r['crossing'][0], 1050.5, delta=1.0)

    def test_road_cut_ditch_does_not_drag_the_crossing(self):
        r = self._run(ditch=True)
        self.assertAlmostEqual(r['crossing'][0], 1050.5, delta=1.0)

    def test_the_ditch_is_a_real_trap(self):
        # without the APPROACH_M stop the trace follows the ditch east
        old = ri.APPROACH_M
        ri.APPROACH_M = 0.0
        try:
            r = self._run(ditch=True)
        finally:
            ri.APPROACH_M = old
        self.assertGreater(abs(r['crossing'][0] - 1050.5), 20.0)


class ScenarioFields(unittest.TestCase):

    def test_count_and_weighted_damage(self):
        tmp = Path(tempfile.mkdtemp())
        shape = (6, 5)
        rows = []
        for sid, w, p, d in [('s1', 0.75, 40.0, 1.5), ('s2', 0.25, 6.0, 0.3)]:
            out = tmp / sid / 'outputs'
            out.mkdir(parents=True)
            reach = np.zeros(shape); reach[2, 2] = 1
            ppr = np.full(shape, np.nan); ppr[2, 2] = p
            pft = np.full(shape, np.nan); pft[2, 2] = d
            _tif(out / 'reach_mask.tif', reach, 0.0)
            _tif(out / 'peak_pressure.tif', ppr, None)
            _tif(out / 'peak_flow_depth.tif', pft, None)
            rows.append({'scenario_id': sid, 'scenario_probability': w})
        hz = {'label': 'T_0', 'manifest': {'work_dir': str(tmp)}, 'df': pd.DataFrame(rows)}
        f = ri.scenario_fields(hz, shape, (5.0, 30.0), (1.0,))
        self.assertEqual(f['n'], 2)
        self.assertEqual(int(f['count'][2, 2]), 2)
        self.assertEqual(int(f['count'][0, 0]), 0)
        vals = {lab: float(a[2, 2]) for lab, a in f['impacts']}
        self.assertAlmostEqual(vals['P(pressure &ge; 5 kPa)'], 1.0)
        self.assertAlmostEqual(vals['P(pressure &ge; 30 kPa)'], 0.75)
        self.assertAlmostEqual(vals['P(depth &ge; 1 m)'], 0.75)

    def test_no_work_dir(self):
        hz = {'label': 'T_0', 'manifest': {}, 'df': pd.DataFrame({'scenario_id': ['s1']})}
        self.assertIsNone(ri.scenario_fields(hz, (2, 2), (5.0,), (1.0,)))


if __name__ == '__main__':
    unittest.main()

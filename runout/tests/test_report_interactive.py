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


if __name__ == '__main__':
    unittest.main()

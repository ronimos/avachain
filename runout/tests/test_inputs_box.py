"""
test_inputs_box.py - the report's "Inputs and caveats" box
(cycle_products.inputs_and_caveats).

    cd runout && .venv/bin/python -m unittest tests.test_inputs_box -v
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts import cycle_products as cp  # noqa: E402


def _hz(label, raster):
    return {'label': label, 'name': 'T+' + label[2:], 'df': pd.DataFrame({'a': range(3)}),
            'config': {'entrainment': {'mode': 'raster', 'raster_cap': False,
                                       'snow_height_raster': str(raster)},
                       'friction': {'tau0': 125}, 'com1dfa': {'friction_model': 'samosATSmall'},
                       'paths': {'dem': '/x/260605_dem.tif'},
                       'reach': {'flow_depth_threshold_m': 0.1}}}


class InputsBox(unittest.TestCase):

    def _release_set(self, root, date):
        d = Path(root) / date / 'tensile_gate'
        d.mkdir(parents=True)
        (d / 'manifest.json').write_text(json.dumps(
            {'producer': 'release', 'horizon': 'tensile_gate', 'snapshot_date': date,
             'created_utc': '2026-11-02T19:40:00Z', 'commit': '387938fd11'}))
        return d / 'scour_depth.tif'

    def test_one_release_row_per_day(self):
        # each horizon has its own release set under rule B
        tmp = tempfile.mkdtemp()
        hzs = [_hz('T_0', self._release_set(tmp, '2026-01-18')),
               _hz('T_1', self._release_set(tmp, '2026-01-19'))]
        box = cp.inputs_and_caveats({'display': {'caveats': ['c1']}}, hzs)
        rows = dict(box['rows'])
        self.assertIn('snowpack 2026-01-18', rows['Release scenarios T+0'])
        self.assertIn('snowpack 2026-01-19', rows['Release scenarios T+1'])
        self.assertEqual(box['snow_line'][0], 'Snowpack')
        self.assertIn('T+1 2026-01-19', box['snow_line'][1])
        self.assertEqual(rows['Entrainment'], 'scour_depth.tif as is, no cap')
        # T+2 missing -> generated caveat first, then the path profile's
        self.assertTrue(box['caveats'][0].startswith('No forecast for T+2'))
        self.assertEqual(box['caveats'][1], 'c1')

    def test_survey_raster_falls_back_to_survey_line(self):
        hz = _hz('T_0', '/nowhere/260114_HS.tif')
        box = cp.inputs_and_caveats({'survey': {'date': '2026-01-14'}}, [hz])
        self.assertEqual(box['snow_line'], ('Last snow height survey', '2026-01-14'))


if __name__ == '__main__':
    unittest.main()

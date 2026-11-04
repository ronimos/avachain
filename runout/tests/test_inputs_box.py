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

    def test_forecast_days_show_their_snapshot_time_and_the_clamp(self):
        # Ron, 2026-11-04: three days at 18 UTC; T+2 clamped to the WRF
        # outlook's last step must say so.
        tmp = Path(tempfile.mkdtemp())
        hzs = []
        for lab, ts, clamped in (('T_0', '2025-11-30T18:00', False),
                                 ('T_2', '2025-12-02T00:00', True)):
            d = Path(tmp) / lab
            d.mkdir()
            (d / 'manifest.json').write_text(json.dumps(
                {'producer': 'release', 'horizon': lab, 'snapshot_date': ts[:10],
                 'snapshot_ts': ts, 'store': 'forecast', 'clamped': clamped,
                 'created_utc': '2026-11-04T22:38:16Z', 'commit': '2fe31d6'}))
            hzs.append(_hz(lab, d / 'scour_depth.tif'))
        box = cp.inputs_and_caveats({'display': {'caveats': ['c1']}}, hzs)
        self.assertEqual(box['snow_line'][1],
                         'T+0 2025-11-30 18:00 UTC (forecast), T+2 2025-12-02 00:00 UTC (forecast)')
        self.assertNotIn('release set', box['snow_line'][1])      # each day has its own set
        self.assertTrue(any(c.startswith('T+2: the WRF outlook ends before') for c in box['caveats']))
        self.assertFalse(any(c.startswith('T+0: the WRF') for c in box['caveats']))

    def test_survey_raster_falls_back_to_survey_line(self):
        hz = _hz('T_0', '/nowhere/260114_HS.tif')
        box = cp.inputs_and_caveats({'survey': {'date': '2026-01-14'}}, [hz])
        self.assertEqual(box['snow_line'], ('Last snow height survey', '2026-01-14'))



class EventCaveats(unittest.TestCase):
    # Ron, 2026-11-03: the Jan 18 notes (debris, last season's DEM offset)
    # are last season's and must not reach a live report.
    DISPLAY = {'caveats': ['general'],
               'event_caveats': {'2026-01-18': ['jan18 a', 'jan18 b']}}

    def test_dated_caveats_only_on_their_own_cycle(self):
        self.assertEqual(cp.cycle_caveats({'display': self.DISPLAY, 'cycle_date': '20260118'}),
                         ['jan18 a', 'jan18 b', 'general'])
        self.assertEqual(cp.cycle_caveats({'display': self.DISPLAY, 'cycle_date': '20261205'}),
                         ['general'])

    def test_cycle_date_with_dashes_and_missing(self):
        self.assertEqual(cp.cycle_caveats({'display': self.DISPLAY, 'cycle_date': '2026-01-18'})[0],
                         'jan18 a')
        self.assertEqual(cp.cycle_caveats({'display': self.DISPLAY}), ['general'])

    def test_yaml_date_keys_still_match(self):
        # An unquoted 2026-01-18 key loads from YAML as a date object
        import datetime
        disp = {'event_caveats': {datetime.date(2026, 1, 18): ['jan18']}}
        self.assertEqual(cp.cycle_caveats({'display': disp, 'cycle_date': '20260118'}), ['jan18'])

    def test_little_professor_profile_keeps_jan18_notes_off_live_reports(self):
        import yaml
        prof = yaml.safe_load((Path(__file__).resolve().parent.parent / 'config' / 'paths'
                               / 'little_professor.yaml').read_text())
        live = cp.cycle_caveats({'display': prof['display'], 'cycle_date': '20261205'})
        jan18 = cp.cycle_caveats({'display': prof['display'], 'cycle_date': '20260118'})
        self.assertFalse(any('debris' in c or '251110' in c for c in live))
        self.assertTrue(any('debris' in c for c in jan18))
        self.assertTrue(any('251110' in c for c in jan18))


if __name__ == '__main__':
    unittest.main()

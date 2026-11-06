"""
test_no_release.py - a forecast day whose release set was published empty
(status "no_scenarios") is reported with the reason (Ron, 2026-11-04).

The release side publishes an empty set when no cluster qualifies as a
trigger and the shot-location releases hold too little snow, or when every
model release holds less than 0.10 m on average. The fixtures copy the
manifest/metadata shapes of the 2025-11-27 and 2025-11-30 sets from the
season-start rehearsal.

    cd runout && .venv/bin/python -m unittest tests.test_no_release -v
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts import cycle_products as cp  # noqa: E402
from scripts import discover  # noqa: E402
from scripts import manifest as mf  # noqa: E402

# 2025-11-27: no candidate; 60 shot-location releases all under the floor
SHOT = ({'producer': 'release', 'snapshot_date': '2025-11-27', 'horizon': 'T_0',
         'status': 'no_scenarios', 'n_scenarios': 0, 'trigger_source': 'explosive_fallback'},
        {'n_scenarios': 0, 'n_trigger_locations': 4, 'trigger_source': 'explosive_fallback',
         'n_below_depth_floor': 60})
# 2025-11-30: 2 model triggers, all 30 releases under the floor
MODEL = ({'producer': 'release', 'snapshot_date': '2025-11-30', 'horizon': 'T_1',
          'status': 'no_scenarios', 'n_scenarios': 0, 'trigger_source': 'model'},
         {'n_scenarios': 0, 'n_trigger_locations': 2, 'trigger_source': 'model',
          'n_below_depth_floor': 30})
# no candidate and no shot locations configured: no metadata is written
NONE = ({'producer': 'release', 'snapshot_date': '2025-11-27', 'status': 'no_scenarios',
         'n_scenarios': 0, 'trigger_source': None}, None)


def _set(d, man, meta):
    d.mkdir(parents=True)
    (d / 'manifest.json').write_text(json.dumps(man))
    if meta is not None:
        (d / 'metadata.json').write_text(json.dumps(meta))
    (d / 'READY').write_text('')
    return d


class Reasons(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_shot_locations_under_the_floor(self):
        r = discover.release_info(_set(self.tmp / 'a', *SHOT))
        self.assertEqual(r['status'], 'no_scenarios')
        self.assertIn('no snowpack cluster met the trigger criteria', r['reason'])
        self.assertIn('60 releases', r['reason'])
        self.assertIn('explosive-control shot locations', r['reason'])
        self.assertIn('0.10 m', r['reason'])
        self.assertIn('2025-11-27', r['reason'])

    def test_model_triggers_under_the_floor(self):
        r = discover.release_info(_set(self.tmp / 'b', *MODEL))
        self.assertIn('all 30 candidate releases from 2 trigger locations', r['reason'])
        self.assertIn('too little snow', r['reason'])
        self.assertNotIn('shot', r['reason'])

    def test_no_shot_locations_configured(self):
        r = discover.release_info(_set(self.tmp / 'c', *NONE))
        self.assertIn('no explosive-control shot locations are configured', r['reason'])

    def test_a_set_with_scenarios_has_no_reason(self):
        man = dict(MODEL[0], status='ok', n_scenarios=75)
        r = discover.release_info(_set(self.tmp / 'd', man, None))
        self.assertEqual(r['status'], 'ok')
        self.assertIsNone(r['reason'])

    def test_not_a_release_set(self):
        d = self.tmp / 'e'
        d.mkdir()
        self.assertIsNone(discover.release_info(d))


class Discovery(unittest.TestCase):

    def test_an_empty_release_set_is_a_horizon(self):
        tmp = Path(tempfile.mkdtemp())
        (tmp / 'T_0' / 'scenario_001').mkdir(parents=True)
        _set(tmp / 'T_1', *MODEL)
        (tmp / 'T_2').mkdir()                    # empty, no manifest: not a horizon
        self.assertEqual([l for _, l in discover.discover_horizons(tmp)], ['T_0', 'T_1'])


def _manifest(horizons):
    return {'path': 'little_professor', 'cycle_date': '20251130', 'issued': '2025-11-30T07:00:00',
            'display': {'project_name': 'Little Professor', 'caveats': ['general']},
            'horizons': horizons}


def _no_release(tmp, label, man_meta, valid):
    # the cycle-manifest entry cli.py writes for an empty release set
    rel = discover.release_info(_set(tmp / 'src' / label, *man_meta))
    m = {'horizons': {}}
    mf.add_horizon(m, label, tmp / 'src' / label, 0, valid,
                   tmp / label, tmp / 'work' / label, 'no release', release=rel)
    return m['horizons'][label]


class Report(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.cycle = self.tmp / 'cycle'
        self.cycle.mkdir()
        self.hz = {'T_0': _no_release(self.tmp, 'T_0', SHOT, '2025-11-30'),
                   'T_1': _no_release(self.tmp, 'T_1', MODEL, '2025-12-01')}

    def test_manifest_keeps_the_reason(self):
        self.assertEqual(self.hz['T_1']['status'], 'no release')
        self.assertIn('too little snow', self.hz['T_1']['release']['reason'])

    def test_caveats_say_why_per_day(self):
        m = _manifest(self.hz)
        # inputs_and_caveats needs a completed horizon; T_2 ran
        hz2 = {'label': 'T_2', 'name': 'T+2', 'df': __import__('pandas').DataFrame({'a': [1]}),
               'config': {'entrainment': {}, 'friction': {}, 'com1dfa': {'friction_model': 'x'},
                          'paths': {'dem': '/x/dem.tif'}, 'reach': {'flow_depth_threshold_m': 0.1}}}
        box_caveats = cp.inputs_and_caveats(m, [hz2])['caveats']
        self.assertTrue(box_caveats[0].startswith('T+1: No release scenarios'))
        self.assertTrue(box_caveats[1].startswith('T+0: No release scenarios'))
        self.assertFalse(any(c.startswith('No forecast for') for c in box_caveats))
        self.assertEqual(box_caveats[-1], 'general')

    def test_every_day_without_release_still_gets_a_report(self):
        (self.cycle / 'manifest.json').write_text(json.dumps(_manifest(self.hz)))
        written = cp.build_cycle_products(self.cycle, products=('report', 'viewer'))
        self.assertEqual([p.name for p in written], ['report.html'])
        html = (self.cycle / 'report.html').read_text()
        self.assertIn('No release scenarios', html)
        self.assertIn('explosive-control shot locations', html)
        self.assertIn('all 30 candidate releases from 2 trigger locations', html)
        self.assertIn('2025-12-01', html)
        self.assertIn('general', html)

    def test_no_completed_day_and_no_reason_still_fails(self):
        m = _manifest({'T_0': {'valid_date': '2025-11-30', 'status': 'FAILED (exit 1)'}})
        (self.cycle / 'manifest.json').write_text(json.dumps(m))
        with self.assertRaises(RuntimeError):
            cp.build_cycle_products(self.cycle)

    def test_publish_links_only_what_the_cycle_has(self):
        (self.cycle / 'manifest.json').write_text(json.dumps(_manifest(self.hz)))
        cp.build_cycle_products(self.cycle, products=('report',))
        pub = self.tmp / 'www'
        dest = pub / 'forecasts' / 'little_professor' / '20251130'
        dest.mkdir(parents=True)
        (dest / 'details.html').write_text('stale run')      # an earlier run of the date
        cp.publish(self.cycle, pub, 'little_professor', '20251130')
        self.assertTrue((dest / 'report.html').exists())
        self.assertFalse((dest / 'details.html').exists())
        index = (pub / 'forecasts' / 'little_professor' / 'index.html').read_text()
        self.assertIn('20251130/report.html', index)
        self.assertNotIn('20251130/details.html', index)


class TestForcedShotReason(unittest.TestCase):
    # Forced explosive shots (Ron, 2026-11-06): each shot initiates; the
    # report says, per shot, why none propagated
    def test_names_each_shot(self):
        info = {'status': 'no_scenarios', 'n_scenarios': 0,
                'trigger_source': 'explosive_forced', 'n_below_depth_floor': 0,
                'shot_outcomes': {
                    '3500': {'n_scenarios': 0, 'n_below_depth_floor': 0,
                             'n_no_release': 15, 'no_release_reason': 'slab_tensile'},
                    '2568': {'n_scenarios': 0, 'n_below_depth_floor': 15,
                             'n_no_release': 0, 'no_release_reason': None}}}
        r = discover.no_release_reason(info)
        self.assertIn('none of the explosive-control shots', r)
        self.assertIn('cluster 3500: the slab could not carry the crack', r)
        self.assertIn('cluster 2568: the release held too little snow', r)


if __name__ == '__main__':
    unittest.main()


class Progress(unittest.TestCase):
    """The report is published after each forecast day (Ron, 2026-11-04);
    days not run yet show as still running, not as missing."""

    def test_set_raster_is_the_sets_own(self):
        tmp = Path(tempfile.mkdtemp())
        s = _set(tmp / 'T_1', *MODEL)
        (s / 'scour_depth.tif').write_text('r')
        (s / 'scenarios').mkdir()
        self.assertEqual(discover.set_raster(s / 'scenarios'), s / 'scour_depth.tif')
        self.assertEqual(discover.set_raster(s), s / 'scour_depth.tif')
        self.assertIsNone(discover.set_raster(tmp))

    def test_pending_days_in_the_report(self):
        tmp = Path(tempfile.mkdtemp())
        m = _manifest({'T_0': _no_release(tmp, 'T_0', SHOT, '2025-11-30'),
                       'T_1': {'valid_date': '2025-12-01', 'status': 'pending'},
                       'T_2': {'valid_date': '2025-12-02', 'status': 'pending'}})
        self.assertEqual(set(cp.pending_days(m)), {'T_1', 'T_2'})
        cycle = tmp / 'cycle'
        cycle.mkdir()
        (cycle / 'manifest.json').write_text(json.dumps(m))
        cp.build_cycle_products(cycle, products=('report',))
        html = (cycle / 'report.html').read_text()
        self.assertIn('Still running; this page is updated when it finishes.', html)
        self.assertIn('explosive-control shot locations', html)      # T+0's reason

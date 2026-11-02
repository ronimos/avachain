"""
test_entrainment_default.py - the forecast entrainment rule (B, 2026-11-02)
and its summary label.

Stdlib unittest (the runout venv has no pytest):
    cd runout && .venv/bin/python -m unittest tests.test_entrainment_default -v
"""
import logging
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))                 # the `scripts` package
sys.path.insert(0, str(ROOT / 'scripts'))     # flat sibling imports

import avalanche_simulation as sim            # noqa: E402
from scripts import cli                       # noqa: E402
from scripts import config as cfgmod          # noqa: E402
from scripts import profiles as prof          # noqa: E402

CONFIG = ROOT / 'config'


def sections_for(profile_name, overrides=()):
    profile = prof.resolve_profile(profile_name, CONFIG / 'profiles.yaml',
                                   overrides=list(overrides))
    physics = cfgmod.load_physics(profile['physics'], CONFIG)
    return profile, physics, cfgmod.merge_physics_and_profile(physics, profile)


class ProfileDefaults(unittest.TestCase):

    def test_forecast_entrains_the_raster_as_is(self):
        _, _, s = sections_for('forecast')
        self.assertEqual(s['entrainment']['mode'], 'raster')
        self.assertIs(s['entrainment']['raster_cap'], False)

    def test_legacy_keeps_vals_capped_rule(self):
        # legacy extends forecast but reads survey snow height
        _, _, s = sections_for('legacy')
        self.assertIs(s['entrainment']['raster_cap'], True)

    def test_set_can_restore_the_cap(self):
        _, _, s = sections_for('forecast', ['entrainment.raster_cap=true'])
        self.assertIs(s['entrainment']['raster_cap'], True)


class CliRefusesSurveyRasterUncapped(unittest.TestCase):

    def _run(self, overrides):
        profile, physics, sections = sections_for('forecast', overrides)
        args = types.SimpleNamespace(
            date='20260118', run_name='t' if overrides else None,
            path=['nowhere'], horizon=None, snow_height='latest',
            survey_date=None)
        local = {'avachain_root': '/nonexistent', 'scenario_root': '/nonexistent'}
        with self.assertLogs(level='ERROR') as logs:
            rc = cli.run_forecast(args, profile, sections, physics, local, CONFIG)
        return rc, '\n'.join(logs.output)

    def test_latest_refused_when_uncapped(self):
        rc, out = self._run([])
        self.assertEqual(rc, 1)
        self.assertIn('raster_cap is false', out)

    def test_latest_allowed_with_cap(self):
        # gets past the guard and fails later on the missing path profile
        rc, out = self._run(['entrainment.raster_cap=true'])
        self.assertEqual(rc, 1)
        self.assertNotIn('raster_cap is false', out)


class Label(unittest.TestCase):

    def test_uncapped_raster_reports_corridor_mean(self):
        # 2026-11-02: the B run's CSV said 0.3 m / config defaults here
        d, src = sim.entrainment_label(
            {'depth_mode': 'dsm_capped', 'depth_mean_m': 1.098},
            raster_cap=False, cap_m=0.3,
            cap_source='config defaults (fallback - scour=0 in params)',
            raster_name='scour_depth.tif')
        self.assertAlmostEqual(d, 1.098)
        self.assertIn('no cap', src)
        self.assertIn('scour_depth.tif', src)
        self.assertNotIn('fallback', src)

    def test_capped_raster_reports_the_cap(self):
        d, src = sim.entrainment_label(
            {'depth_mode': 'dsm_capped', 'depth_mean_m': 0.29},
            raster_cap=True, cap_m=0.107, cap_source='params.json (per-scenario)',
            raster_name='hs_260114_1m.tif')
        self.assertAlmostEqual(d, 0.107)
        self.assertIn('capped at params.json', src)

    def test_constant_mode_unchanged(self):
        # no raster (missing file, auto, polygon): the cap, as before
        d, src = sim.entrainment_label(
            {'depth_mode': 'constant', 'depth_mean_m': 0.3},
            raster_cap=False, cap_m=0.3, cap_source='config defaults',
            raster_name=None)
        self.assertEqual((d, src), (0.3, 'config defaults'))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    unittest.main()

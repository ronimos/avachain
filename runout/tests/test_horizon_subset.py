"""
test_horizon_subset.py - a run of some days only keeps the cycle's other
days in its manifest (Ron, 2026-11-18). avachain's nightly chain runs T_0
alone (due 03:00) and cron runs each later day after every slope's T_0 (due
08:00). Without this, each run rewrote manifest.json with its own days only,
so the T_1 run would have dropped T_0 from the published report.

    cd runout && .venv/bin/python -m unittest tests.test_horizon_subset -v
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts import cli  # noqa: E402
from scripts import cycle_products as cp  # noqa: E402


def _manifest(hash_='h1'):
    return {'issued': '2026-12-04T22:40:00', 'git_commit': 'abc',
            'physics': {'hash': hash_}, 'horizons': {}}


class TestKeepOtherHorizons(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.mpath = self.tmp / 'manifest.json'
        self.days = [(self.tmp / 'src' / l, l) for l in ('T_0', 'T_1', 'T_2')]

    def _run(self, labels, manifest=None):
        m = manifest or _manifest()
        cli.keep_other_horizons(m, cli.read_manifest(self.mpath), self.days,
                                [d for d in self.days if d[1] in labels],
                                '20261205', self.tmp / 'out', self.tmp / 'work')
        return m

    def test_today_alone_lists_the_later_days_as_pending(self):
        m = self._run(['T_0'])
        self.assertEqual(sorted(m['horizons']), ['T_1', 'T_2'])
        self.assertEqual({h['status'] for h in m['horizons'].values()}, {'pending'})
        self.assertEqual(set(cp.pending_days(m)), {'T_1', 'T_2'})
        self.assertEqual(m['horizons']['T_2']['valid_date'], '2026-12-07')

    def test_a_later_day_keeps_what_is_done_and_what_is_still_to_come(self):
        prev = _manifest()
        prev['horizons'] = {'T_0': {'status': 'ok', 'n_scenarios': 15},
                            'T_1': {'status': 'pending'}, 'T_2': {'status': 'pending'}}
        self.mpath.write_text(json.dumps(prev))
        m = self._run(['T_1'])
        self.assertEqual(sorted(m['horizons']), ['T_0', 'T_2'])
        t0 = m['horizons']['T_0']
        self.assertEqual((t0['status'], t0['n_scenarios']), ('ok', 15))
        self.assertEqual(t0['carried_from']['issued'], '2026-12-04T22:40:00')
        self.assertEqual(m['horizons']['T_2']['status'], 'pending')

    def test_a_failed_or_no_release_day_is_carried_as_it_was(self):
        prev = _manifest()
        prev['horizons'] = {'T_0': {'status': 'no release'}, 'T_1': {'status': 'FAILED (exit 1)'}}
        self.mpath.write_text(json.dumps(prev))
        m = self._run(['T_2'])
        self.assertEqual(m['horizons']['T_0']['status'], 'no release')
        self.assertEqual(m['horizons']['T_1']['status'], 'FAILED (exit 1)')

    def test_the_earlier_manifest_is_read_before_this_run_writes_its_own(self):
        """run_forecast writes its new manifest.json straight after building
        it; the earlier run's has to be read first (it was not, on the first
        try: the T_1 run showed T_0 as pending)."""
        import inspect
        src = inspect.getsource(cli.run_forecast)
        self.assertLess(src.index('prev_manifest = read_manifest(manifest_path)'),
                        src.index('mf.write(manifest, manifest_path)'))

    def test_all_days_adds_nothing(self):
        m = self._run(['T_0', 'T_1', 'T_2'])
        self.assertEqual(m['horizons'], {})


if __name__ == '__main__':
    unittest.main()

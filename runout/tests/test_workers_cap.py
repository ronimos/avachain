"""
test_workers_cap.py - the max_workers ceiling (scripts/workers.py).

Stdlib unittest, because the runout venv has no pytest:
    cd runout && .venv/bin/python -m unittest tests.test_workers_cap -v
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
import workers  # noqa: E402


class CappedWorkers(unittest.TestCase):

    def test_request_above_cap_is_clamped(self):
        # 2026-11-02: --set execution.max_workers=12 against the forecast profile
        with self.assertLogs(level='WARNING'):
            n = workers.capped_workers({'max_workers': 12, 'max_workers_cap': 4})
        self.assertEqual(n, 4)

    def test_missing_cap_key_still_caps(self):
        # A profile without the key must not fall back to "no limit"
        n = workers.capped_workers({'max_workers': 12})
        self.assertEqual(n, workers.DEFAULT_MAX_WORKERS_CAP)

    def test_auto_is_capped(self):
        # max_workers None = one per CPU; 64 CPUs here
        n = workers.capped_workers({'max_workers': None}, cpu_count=64)
        self.assertEqual(n, workers.DEFAULT_MAX_WORKERS_CAP)

    def test_request_below_cap_unchanged(self):
        self.assertEqual(workers.capped_workers({'max_workers': 2}), 2)

    def test_cap_can_be_raised_deliberately(self):
        n = workers.capped_workers({'max_workers': 8, 'max_workers_cap': 8})
        self.assertEqual(n, 8)

    def test_set_string_values(self):
        # --set may leave strings when there is no type to copy
        self.assertEqual(workers.capped_workers({'max_workers': '12',
                                                 'max_workers_cap': '4'}), 4)

    def test_bad_cap_rejected(self):
        with self.assertRaises(ValueError):
            workers.capped_workers({'max_workers': 2, 'max_workers_cap': 0})

    def test_no_execution_section(self):
        self.assertEqual(workers.capped_workers(None, cpu_count=64),
                         workers.DEFAULT_MAX_WORKERS_CAP)


if __name__ == '__main__':
    unittest.main()

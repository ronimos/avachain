"""
tests/test_trigger_ranking.py — trigger ranking in analysis_pipeline.

Sk38 is reported to two decimals, so the top-N cut regularly falls inside a
tie (4-way at 0.79 on 2026-01-18, 10-way on 2026-03-07). Until 2026-10-20
nsmallest() settled those by CSV row order. They are now broken by the
critical cut length r_c at the same weak layer. Mirrors
~/avalanche-release-areas tests/test_trigger_ranking.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "avachain"))

from analysis_pipeline import _rank_triggers


def _cands(rows):
    return pd.DataFrame(rows, columns=['cid', 'min_sk38', 'rc_wl']).set_index('cid')


def test_sk38_still_decides_when_it_separates():
    """A lower Sk38 wins even against a far lower r_c."""
    df = _cands([(1, 0.80, 0.00), (2, 0.77, 0.50)])
    assert list(_rank_triggers(df, 2, 'min_sk38').index) == [2, 1]


def test_sk38_tie_is_broken_by_lowest_rc_not_row_order():
    # Row order puts 10 first; r_c says 30 is the most propagation-prone.
    df = _cands([(10, 0.79, 0.15), (20, 0.79, 0.09), (30, 0.79, 0.04),
                 (40, 0.77, 0.30)])
    assert list(_rank_triggers(df, 3, 'min_sk38').index) == [40, 30, 20]


def test_missing_rc_sorts_after_defined_rc_within_a_tie():
    df = _cands([(1, 0.79, np.nan), (2, 0.79, 0.20)])
    assert list(_rank_triggers(df, 2, 'min_sk38').index) == [2, 1]


def test_full_tie_falls_back_to_row_order():
    df = _cands([(5, 0.79, 0.10), (3, 0.79, 0.10), (9, 0.79, 0.10)])
    assert list(_rank_triggers(df, 3, 'min_sk38').index) == [5, 3, 9]


def test_missing_sk38_is_never_a_trigger():
    df = _cands([(1, np.nan, 0.0), (2, 0.90, 0.5)])
    assert list(_rank_triggers(df, 2, 'min_sk38').index) == [2]


def test_csv_without_rc_column_ranks_by_sk38_alone():
    df = _cands([(1, 0.80, 0.1), (2, 0.77, 0.1)]).drop(columns='rc_wl')
    assert list(_rank_triggers(df, 2, 'min_sk38').index) == [2, 1]

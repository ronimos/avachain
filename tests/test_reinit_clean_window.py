"""The reinit's clean-window check, which decides whether the post-event
survey HS sets the scour depth.

Until 2026-10-28 it read wx.iloc[:, 11] from the raw CAIC CSV, which is
snow24h, not depth, and scaled tenths of an inch to cm while the threshold
is in m. On Jan 18 the station depth was flat (302 -> 304 tenths of an inch
from the event to the 01-20 flight), yet the window came out "contaminated",
and all 420 scoured clusters took the modelled slab thickness instead of
the survey.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "avachain"))

from reinitialize_snowpack import post_event_station_dhs, is_clean_window
from smet_writer import load_and_convert

# Column order of data/weather/weather_data.csv: depth is 10, snow24h is 11.
CAIC_COLUMNS = ['swin', 'temp', 'dewp', 'rh', 'wspd', 'wdir', 'gust',
                'mxtemp24h', 'mntemp24h', 'pcpac', 'depth', 'snow24h']


def _caic_csv(tmp_path, depth, snow24h):
    """Hourly raw CAIC CSV from 2026-01-18 00:00 UTC, units as stored."""
    t = pd.date_range('2026-01-18', periods=len(depth), freq='h')
    df = pd.DataFrame({c: 0 for c in CAIC_COLUMNS}, index=t)
    df['temp'], df['rh'] = 200, 80
    df['depth'], df['snow24h'] = depth, snow24h
    df.index.name = 'time'
    p = tmp_path / 'weather_data.csv'
    df.to_csv(p)
    return load_and_convert(str(p))


def test_flat_depth_is_clean_whatever_snow24h_does(tmp_path):
    # Depth flat at 302 tenths of an inch; snow24h swings 1 -> 12 -> 2.
    n = 72
    snow = np.r_[np.full(30, 1), np.full(12, 12), np.full(n - 42, 2)]
    wx = _caic_csv(tmp_path, np.full(n, 302), snow)
    dhs = post_event_station_dhs(wx, '2026-01-18T18:00', '2026-01-20', 18)
    assert dhs == pytest.approx(0.0, abs=1e-9)
    assert is_clean_window(dhs)
    # What the old code computed from the same CSV: snow24h difference over
    # event_date..date_after, "x 0.1 x 2.54", read as metres.
    raw = pd.read_csv(tmp_path / 'weather_data.csv', parse_dates=[0], index_col=0)
    old = raw.iloc[:, 11].loc['2026-01-18':'2026-01-20']
    assert raw.columns[11] == 'snow24h'
    assert not (float((old.iloc[-1] - old.iloc[0]) * 0.1 * 2.54) < 0.05)


def test_new_snow_after_the_event_is_contaminated(tmp_path):
    # 10 cm of new snow (39.4 tenths of an inch) after the event at 18:00.
    depth = np.r_[np.full(24, 302), np.full(48, 302 + 39.4)]
    wx = _caic_csv(tmp_path, depth, np.zeros(72))
    dhs = post_event_station_dhs(wx, '2026-01-18T18:00', '2026-01-20', 18)
    assert dhs == pytest.approx(0.10, abs=1e-3)
    assert not is_clean_window(dhs)


def test_snow_before_the_event_does_not_count(tmp_path):
    # The station gains 10 cm on the event day before 18:00, then stays flat.
    depth = np.r_[np.full(12, 302), np.full(60, 302 + 39.4)]
    wx = _caic_csv(tmp_path, depth, np.zeros(72))
    dhs = post_event_station_dhs(wx, '2026-01-18T18:00', '2026-01-20', 18)
    assert dhs == pytest.approx(0.0, abs=1e-9)


def test_threshold_is_metres_and_nan_is_not_clean():
    assert is_clean_window(0.049) and not is_clean_window(0.051)
    assert not is_clean_window(np.nan)

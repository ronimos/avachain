"""Tests for the single-pass cluster SMET builder.

forcing_pipeline's `smet` step used to write each cluster's SMET three times
over the whole season: write_smet, then augment_smet_with_refill (re-parse,
add the refill fields, rewrite), then fill_smet_gaps_from_refill (re-parse,
fill gaps, rewrite). That was ~54 of the 71 minutes of a survey tick's smet
step. build_cluster_smet does the same in memory and writes once.

It has to be byte-identical to the chain, and that is not automatic: the chain
round-trips values through the file's %.3f text between steps, and the gap
fill interpolates HS across inserted rows from those *rounded* neighbours. An
in-memory version that skips the rounding writes different HS in every
inserted row. The fixture below has inserted rows, NaN fields, a refill that
only partly covers the period and HS with more than three decimals, so it
fails if any stage's rounding is dropped.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "avachain"))

from smet_writer import (                                    # noqa: E402
    StationConfig,
    augment_smet_with_refill,
    build_cluster_smet,
    fill_smet_gaps_from_refill,
    read_smet,
    write_smet,
)

CONFIG = StationConfig(
    station_id="cluster_0042", station_name="cluster_42_n7",
    latitude=39.6312345678, longitude=-105.8712345678, altitude_m=3712.34,
    slope_angle=37.123456789, slope_azi=84.98765, tz_offset=0,
)


def _station_df() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    idx = pd.date_range("2026-02-01T00:00", "2026-02-12T23:00", freq="1h")
    df = pd.DataFrame({
        "TA": rng.normal(-8, 4, len(idx)),
        "RH": rng.uniform(40, 100, len(idx)),
        "VW": rng.gamma(2, 2, len(idx)),
        "DW": rng.uniform(0, 360, len(idx)),
        "ISWR": np.clip(rng.normal(200, 200, len(idx)), 0, None),
        "ILWR": rng.normal(230, 20, len(idx)),
        "HS": 150 + np.cumsum(rng.normal(0, 0.37, len(idx))),
    }, index=idx)
    df.loc["2026-02-03T05:00":"2026-02-03T09:00", "TA"] = np.nan   # short gap
    df.loc["2026-02-04T00:00":"2026-02-06T12:00", "VW"] = np.nan   # > 48 h
    # Missing hours: a short block (filled) and a long one (left as nodata).
    df = df.drop(df.loc["2026-02-07T03:00":"2026-02-07T08:00"].index)
    df = df.drop(df.loc["2026-02-09T00:00":"2026-02-11T06:00"].index)
    return df


def _refill() -> tuple:
    idx = pd.date_range("2026-02-02T00:00", "2026-02-12T12:00", freq="6h")
    rng = np.random.default_rng(11)
    df = pd.DataFrame({
        "TA": rng.normal(-9, 3, len(idx)),
        "VW": rng.gamma(2, 2, len(idx)),
        "Qs": rng.normal(10, 30, len(idx)),
        "OLWR": rng.normal(280, 15, len(idx)),
        "HS_mod": rng.normal(1.4, 0.1, len(idx)),
        "HS_meas": np.full(len(idx), np.nan),
    }, index=pd.DatetimeIndex(idx, name="timestamp"))
    header = {
        "fields_list": list(df.columns),
        "units_offset_list": [273.15, 0, 0, 0, 0, 0],
        "units_multiplier_list": [1, 1, 1, 1, 1, 1],
    }
    return header, df


def _chain(tmp_path: Path, df, refill_header, refill_df) -> bytes:
    path = tmp_path / "chain.smet"
    write_smet(df, str(path), CONFIG)
    augment_smet_with_refill(str(path), refill_header, refill_df)
    fill_smet_gaps_from_refill(str(path), refill_header, refill_df, max_gap_hours=48)
    return path.read_bytes()


def test_single_pass_is_byte_identical_to_the_three_step_chain(tmp_path):
    df = _station_df()
    refill_header, refill_df = _refill()

    expected = _chain(tmp_path, df, refill_header, refill_df)
    out = tmp_path / "single.smet"
    stats = build_cluster_smet(df, str(out), CONFIG, refill_header, refill_df,
                               max_gap_hours=48)

    assert out.read_bytes() == expected
    assert stats["added"] == ["Qs", "OLWR"]
    assert stats["n_rows_inserted"] == 6 + 55


def test_fixture_exercises_hs_interpolation_from_rounded_values(tmp_path):
    """Guard on the fixture: HS in the inserted rows depends on the rounding.

    If interpolating from unrounded HS gave the same text, the byte-identity
    test above could not tell a dropped rounding stage from a correct one.
    """
    df = _station_df()
    refill_header, refill_df = _refill()
    out = tmp_path / "single.smet"
    build_cluster_smet(df, str(out), CONFIG, refill_header, refill_df)
    _, written = read_smet(str(out))

    inserted = pd.date_range("2026-02-07T03:00", "2026-02-07T08:00", freq="1h")
    full = df["HS"].reindex(pd.date_range(df.index[0], df.index[-1], freq="1h"))
    unrounded = full.interpolate(method="time", limit=48).loc[inserted]
    assert [f"{v:.3f}" for v in unrounded] != \
           [f"{v:.3f}" for v in written.loc[inserted, "HS"]]


def test_without_refill_it_is_write_smet(tmp_path):
    df = _station_df()
    write_smet(df, str(tmp_path / "a.smet"), CONFIG)
    build_cluster_smet(df, str(tmp_path / "b.smet"), CONFIG)
    assert (tmp_path / "a.smet").read_bytes() == (tmp_path / "b.smet").read_bytes()

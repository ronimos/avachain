"""
tests/test_survey_validity.py — biased survey patches are not trusted.

Run with:  pytest tests/test_survey_validity.py -v

Negative survey HS is clipped to exact 0, and a cluster with many zero cells
is a biased patch (median bias -45 cm at 10-25% zeros, -99 cm above 75%),
not bare ground. Such a cluster-survey is invalid: gap-fill carries the
cluster forward from its last valid survey and gives it no transport, and
the anchor and reinit skip it.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "avachain"))

from survey_validity import (carry_forward, compute_validity,          # noqa: E402
                             invalid_cells, invalid_clusters, write_validity)

# two 10-cell clusters side by side
CM = np.array([[1] * 10 + [2] * 10])


def _grid(c1: list, c2: list) -> np.ndarray:
    return np.array([c1 + c2], dtype=float)


SNOW = [0.8] * 10


class TestComputeValidity:

    def test_zero_patch_is_invalid(self):
        g = {"2026-02-21": _grid(SNOW, SNOW),
             "2026-03-02": _grid([0.0] * 5 + [0.3] * 5, SNOW)}
        assert compute_validity(g, CM) == {"2026-02-21": [], "2026-03-02": [1]}

    def test_few_zeros_still_valid(self):
        g = {"2026-02-21": _grid(SNOW, SNOW),
             "2026-03-02": _grid([0.0] + [0.8] * 9, SNOW)}       # 10%, not > 10%
        assert compute_validity(g, CM)["2026-03-02"] == []

    def test_first_survey_never_invalid(self):
        g = {"2025-11-26": _grid([0.0] * 6 + [0.05] * 4, SNOW)}
        assert compute_validity(g, CM)["2025-11-26"] == []

    def test_near_bare_before_keeps_zeros_valid(self):
        g = {"2026-03-24": _grid([0.1] * 10, SNOW),               # melting out
             "2026-03-31": _grid([0.0] * 6 + [0.05] * 4, SNOW)}
        assert compute_validity(g, CM)["2026-03-31"] == []

    def test_last_valid_is_the_reference_through_a_streak(self):
        # cluster 1 invalid twice; the near-bare test must still look back
        # to the last VALID survey (0.8 m), not the invalid one
        bad = [0.0] * 6 + [0.05] * 4
        g = {"2026-02-14": _grid(SNOW, SNOW),
             "2026-02-19": _grid(bad, SNOW),
             "2026-02-21": _grid(bad, SNOW),
             "2026-03-02": _grid(SNOW, SNOW)}
        v = compute_validity(g, CM)
        assert v["2026-02-19"] == [1] and v["2026-02-21"] == [1]
        assert v["2026-03-02"] == []

    def test_round_trip(self, tmp_path):
        write_validity(tmp_path, {"2026-03-02": [1]})
        assert invalid_clusters(tmp_path, "2026-03-02") == {1}
        assert invalid_clusters(tmp_path, "2026-03-05") == set()
        assert invalid_clusters(tmp_path / "missing", "2026-03-02") == set()


class TestCarryForward:

    def setup_method(self):
        self.hs = _grid([0.0] * 10, SNOW)                         # cluster 1 biased
        self.prev = _grid([0.9] * 10, [0.7] * 10)                 # predicted end of previous pair
        self.transport = np.full(self.hs.shape, 0.5)
        self.inv1 = invalid_cells(CM, [1])
        self.none = invalid_cells(CM, [])

    def test_invalid_start_takes_previous_prediction(self):
        hs, _ = carry_forward(self.hs, self.transport, self.inv1, self.none, self.prev)
        assert np.all(hs[CM == 1] == 0.9)
        assert np.all(hs[CM == 2] == 0.8)                         # valid: survey kept

    def test_transport_zero_where_either_end_invalid(self):
        _, tr = carry_forward(self.hs, self.transport, self.none, self.inv1, self.prev)
        assert np.all(tr[CM == 1] == 0.0) and np.all(tr[CM == 2] == 0.5)
        _, tr = carry_forward(self.hs, self.transport, self.inv1, self.none, self.prev)
        assert np.all(tr[CM == 1] == 0.0)

    def test_no_previous_pair_keeps_survey(self):
        hs, _ = carry_forward(self.hs, self.transport, self.inv1, self.none, None)
        assert np.all(hs == self.hs)

    def test_station_only_has_no_transport(self):
        _, tr = carry_forward(self.hs, None, self.inv1, self.none, self.prev)
        assert tr is None


class TestAnchorSkipsInvalid:

    def test_invalid_cluster_not_anchored(self, tmp_path):
        from reinitialize_snowpack import anchor_to_survey
        sno_text = ("SMET 1.1 ASCII\n[HEADER]\nProfileDate = 2026-03-02T18:00:00\n"
                    "HS_Last = 1.0\nSlopeAngle = 0.0\nnSnowLayerData = 1\n"
                    "ErosionLevel = 0\nfields = timestamp Layer_Thick\n[DATA]\n"
                    "2025-11-26T18:00:00     1.000000\n")
        out, smet, an = (tmp_path / d for d in ("out", "smet", "an"))
        for d in (out, smet, an):
            d.mkdir()
        np.save(an / "cluster_map.npy", CM)
        np.save(an / "hs_2026-03-02.npy", _grid([0.0] * 5 + [0.3] * 5, [0.5] * 10))
        for cid in (1, 2):
            n = f"cluster_{cid:04d}"
            (out / f"{n}_{n}.sno").write_text(sno_text)
            (smet / f"{n}.smet").write_text(
                "SMET 1.1 ASCII\n[HEADER]\nnodata = -999\nfields = timestamp HS\n"
                "[DATA]\n2026-03-02T18:00:00  100.0\n")
        write_validity(an, {"2026-03-02": [1]})
        cfg = SimpleNamespace(pro_dir=out, smet_dir=smet, analysis_dir=an,
                              cluster_map_path=an / "cluster_map.npy", resampled_dir=an)
        r = anchor_to_survey(cfg, pd.Timestamp("2026-03-02T18:00"))
        assert [s["cid"] for s in r["stats"]] == [2]
        assert "1.000000" in (out / "cluster_0001_cluster_0001.sno").read_text()


class TestFillNodata:
    """A NaN start cell stays NaN through gap_fill_period and every later pair."""

    def test_nan_propagates_without_fill(self):
        from spatial_model import gap_fill_period
        hs = np.array([[0.8, 0.8, np.nan]])
        idx = pd.date_range("2026-01-01", periods=4, freq="h")
        grids = gap_fill_period(hs, np.zeros(3), pd.Series([1.0, 1.1, 1.2, 1.3], idx),
                                pd.Series([2.0] * 4, idx), np.ones(3, bool))
        assert np.isnan(list(grids.values())[-1][0, 2])

    def test_cluster_median_removes_it(self):
        from spatial_model import gap_fill_period
        from survey_validity import fill_nodata
        cm = np.array([[1, 1, 1, 1]])
        hs = np.array([[0.6, 1.0, 5.0, np.nan]])
        out, n_med, n_near = fill_nodata(hs, cm)
        assert out[0, 3] == pytest.approx(1.0) and (n_med, n_near) == (1, 0)
        assert out[0, :3].tolist() == [0.6, 1.0, 5.0]       # valid cells untouched
        idx = pd.date_range("2026-01-01", periods=4, freq="h")
        grids = gap_fill_period(out, np.zeros(4), pd.Series([1.0, 1.1, 1.2, 1.3], idx),
                                pd.Series([2.0] * 4, idx), np.ones(4, bool))
        assert not any(np.isnan(g).any() for g in grids.values())

    def test_all_nodata_cluster_takes_nearest_valid_cell(self):
        from survey_validity import fill_nodata
        cm = np.array([[1, 1, 2, 3]])
        hs = np.array([[0.5, 0.7, np.nan, 2.0]])
        out, n_med, n_near = fill_nodata(hs, cm)
        assert (n_med, n_near) == (0, 1) and out[0, 2] == pytest.approx(0.7)

    def test_outside_clusters_stay_nan(self):
        from survey_validity import fill_nodata
        out, *_ = fill_nodata(np.array([[0.5, np.nan]]), np.array([[1, 0]]))
        assert np.isnan(out[0, 1])

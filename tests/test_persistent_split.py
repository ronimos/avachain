"""
tests/test_persistent_split.py — clusters split on persistent heterogeneity.

Run with:  pytest tests/test_persistent_split.py -v

The single-survey std test split every eligible cluster (3102 of 3102): its
median threshold (7.9 cm) sat below the median within-cluster std on clean
surveys (8.2 cm), and its max over surveys only grows. Survey noise and
biased patches do not repeat from survey to survey; terrain does. So the
split groups are formed on half the surveys and tested on the other half,
against a null of cells shuffled within each survey.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "avachain"))

from clustering import apply_split, assess_persistent_splits, persistent_split  # noqa: E402

RNG = np.random.default_rng(1)
N_SURVEYS = 12


def _cluster(n=16, offset=0.0, noise=0.08, base=1.0):
    """n cells x N_SURVEYS; the first half sits `offset` higher on every survey."""
    trend = base + np.linspace(0, 0.5, N_SURVEYS)
    X = trend + RNG.normal(0, noise, (n, N_SURVEYS))
    X[: n // 2] += offset
    return X


class TestPersistentSplit:

    def test_persistent_difference_found(self):
        r = persistent_split(_cluster(offset=0.3))
        assert r["effect_m"] == pytest.approx(0.3, abs=0.06)
        assert r["consistency"] == 1.0
        assert set(r["labels"][:8]) == {1} and set(r["labels"][8:]) == {0}

    def test_noise_alone_is_small_and_inconsistent(self):
        effects = [persistent_split(_cluster(offset=0.0))["effect_m"] for _ in range(30)]
        assert np.median(effects) < 0.03

    def test_one_biased_survey_is_not_persistent(self):
        # a -1 m patch on half the cells in ONE survey: the old max-std test
        # splits this; across held-out surveys it does not repeat
        X = _cluster(offset=0.0)
        X[:8, 5] -= 1.0
        old_std_test = np.max(np.nanstd(X, axis=0)) > 0.12
        assert old_std_test
        assert persistent_split(X)["effect_m"] < 0.05

    def test_too_few_cells_or_surveys(self):
        assert persistent_split(_cluster(n=6)) is None
        assert persistent_split(_cluster()[:, :3]) is None

    def test_no_data_cells_get_no_label(self):
        X = _cluster(offset=0.3)
        X[0, :] = np.nan
        assert persistent_split(X)["labels"][0] == -1


def _map_and_grids(structured: dict, n_noise=40, n=16):
    """A row of clusters, n cells each; `structured` maps cid -> offset."""
    n_c = n_noise + len(structured)
    cm = np.repeat(np.arange(1, n_c + 1), n)[None, :]
    grids = {}
    stacks = {c: _cluster(n=n, offset=structured.get(c, 0.0)) for c in range(1, n_c + 1)}
    for j in range(N_SURVEYS):
        g = np.concatenate([stacks[c][:, j] for c in range(1, n_c + 1)])[None, :]
        grids[f"2026-01-{j + 1:02d}"] = g
    return cm, grids


class TestAssessPersistentSplits:

    def test_only_structured_cluster_splits(self):
        cm, grids = _map_and_grids({7: 0.4})
        res, thr = assess_persistent_splits(cm, grids)
        assert [r["cid"] for r in res if r["should_split"]] == [7]
        assert thr["n_null"] == 41

    def test_invalid_surveys_are_left_out(self):
        # A patch biased on one or two surveys is already rejected by the
        # cross-validation. One biased on most surveys (52 clusters had 6+
        # invalid surveys in a row at the 10% rule) is persistent, and only
        # the validity mask keeps it from splitting the cluster.
        cm, grids = _map_and_grids({})
        biased = [f"2026-01-{j:02d}" for j in range(1, 8)]          # 7 of 12
        for d in biased:
            grids[d][0, 2 * 16: 2 * 16 + 8] -= 1.0
        res, _ = assess_persistent_splits(cm, grids)
        assert any(r["should_split"] for r in res if r["cid"] == 3)   # fooled
        res, _ = assess_persistent_splits(cm, grids, {d: [3] for d in biased})
        assert not any(r["should_split"] for r in res if r["cid"] == 3)

    def test_apply_split_gives_smaller_group_new_id(self):
        cm = np.array([[5] * 10])
        r = {"labels": np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, -1]),
             "rows": np.zeros(10, int), "cols": np.arange(10)}
        cm, child = apply_split(cm, r)
        assert child == 6
        assert cm.tolist() == [[6, 6, 6, 5, 5, 5, 5, 5, 5, 5]]

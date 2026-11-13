"""Severed slab slides (model_params.USE_SEVERED_SLIDE, Ron 2026-11-13).

A piece the slab-crack lines cut off from the trigger stays in the release when
its mean cluster slope exceeds atan(SEVERED_SLIDE_MU): its weak layer already
fractured, so only crack-face friction can hold it. Pinned on a synthetic
graph; Jan 18 cannot isolate it."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "avachain"))
import release_geometry as rg    # noqa: E402
import model_params as mp        # noqa: E402

# chain 1-2-3-4-5-6; 7 isolated
NB = {1: [2], 2: [1, 3], 3: [2, 4], 4: [3, 5], 5: [4, 6], 6: [5], 7: []}


def test_steep_piece_slides_gentle_piece_stays_dropped():
    slope = {1: 35, 2: 35, 4: 28, 5: 28, 7: 40}.get
    # severed pieces {1,2} (35 deg) and {4,5} (28 deg) and {7} (40 deg)
    out = rg.severed_slides({1, 2, 4, 5, 7}, NB, slope, mu=0.62)
    assert out == {1, 2, 7}


def test_piece_is_judged_by_its_mean_slope():
    slope = {1: 40, 2: 25}.get                     # mean 32.5 > 31.8
    assert rg.severed_slides({1, 2}, NB, slope, mu=0.62) == {1, 2}
    slope = {1: 36, 2: 25}.get                     # mean 30.5 < 31.8
    assert rg.severed_slides({1, 2}, NB, slope, mu=0.62) == set()


def test_threshold_is_atan_mu_and_order_independent():
    lim = np.degrees(np.arctan(0.62))
    assert rg.severed_slides({3}, NB, lambda c: lim + 0.01, mu=0.62) == {3}
    assert rg.severed_slides({3}, NB, lambda c: lim - 0.01, mu=0.62) == set()
    nb_rev = {k: list(reversed(v)) for k, v in reversed(list(NB.items()))}
    slope = {1: 35, 2: 35, 4: 28, 5: 28}.get
    assert rg.severed_slides([5, 4, 2, 1], nb_rev, slope, mu=0.62) == {1, 2}


def test_defaults():
    assert mp.USE_SEVERED_SLIDE is True   # adopted 2026-11-13 (Ron)
    assert mp.SEVERED_SLIDE_MU == 0.62
    import inspect
    src = inspect.getsource(rg.propagate_release)
    assert "severed_slides(" in src and "mp.USE_SEVERED_SLIDE" in src

"""
tests/test_avalanche_separation.py — the interior of a large avalanche mask.

separate_avalanche_from_wind interpolates wind transport into the avalanche
mask from its surroundings, with windows up to 50 cells. Deeper than that
into a large mask there is nothing to average, the estimate stayed NaN, and
avalanche_dhs = transport - NaN was NaN exactly where the loss is largest
(12-12..12-23: median observed dHS -1.33 m). apply_avalanche_event then put
that NaN into every hourly grid after the event.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "avachain"))

from avalanche import separate_avalanche_from_wind                     # noqa: E402


def _case():
    n = 200
    transport = np.full((n, n), 0.1)                 # wind: +10 cm everywhere
    mask = np.zeros((n, n), bool)
    mask[20:180, 20:180] = True                       # 160 cells: interior > 50 from edge
    transport[mask] = -1.3                            # observed loss minus station
    return transport, mask, np.ones((n, n), bool)


def test_deep_interior_is_finite_and_keeps_the_loss():
    transport, mask, valid = _case()
    avy, wind = separate_avalanche_from_wind(transport, mask, valid)
    assert not np.isnan(avy).any() and not np.isnan(wind).any()
    # centre is >50 cells from any wind cell: no wind estimate, whole loss is avalanche
    assert avy[100, 100] == -1.3 and wind[100, 100] == 0.0
    # near the edge the wind estimate (+0.1) is still removed
    assert abs(avy[22, 22] - (-1.3 - 0.1)) < 1e-6


def test_no_data_transport_gives_no_avalanche_dhs():
    transport, mask, valid = _case()
    transport[100, 100] = np.nan                      # survey no-data inside the mask
    avy, _ = separate_avalanche_from_wind(transport, mask, valid)
    assert avy[100, 100] == 0.0

"""plot_gap_fill_validation for any number of gap-fill periods.

With one period (a season's first survey pair) plt.subplots returned a 1-D
axes array and axes[0, j] raised IndexError, failing the 251129 survey in the
11-04 season-start rehearsal. A validation plot must not fail a survey.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "avachain"))
import plots


def _metric(i):
    rng = np.random.default_rng(i)
    obs = rng.uniform(0, 1, (20, 20))
    return {"date_b": f"2025-11-{26 + i}", "r": 0.8, "rmse": 0.3, "bias": 0.02,
            "n_hours": 73, "pair_id": f"2025-11-{25 + i}__2025-11-{26 + i}",
            "obs": obs, "pred": obs + rng.normal(0, 0.1, obs.shape),
            "valid": np.ones(obs.shape, bool)}


@pytest.mark.parametrize("n", [1, 2, 5])
def test_any_number_of_periods_plots(tmp_path, n):
    plots.plot_gap_fill_validation([_metric(i) for i in range(n)],
                                   SimpleNamespace(plots_dir=tmp_path))
    assert (tmp_path / "gap_fill_validation.png").stat().st_size > 0

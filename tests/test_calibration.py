"""Per-slope release-model calibration ([calibration], calibration.py).

CLAUDE.md "Calibration" (Ron, 2026-11-08). Two things must hold: a slope
without [calibration] runs exactly as before, and a calibrated value actually
reaches the physics. The second failed silently until 2026-11-08: C_FIT was a
default argument, so setting arrest_indices.C_FIT at runtime changed nothing.
"""
import json
from types import SimpleNamespace

import numpy as np
import pytest

import jan18_fixture  # noqa: F401  (puts src/avachain on sys.path)

import arrest_indices as _ai
import calibration
from config import ProjectConfig


@pytest.fixture(autouse=True)
def _restore_c_fit(monkeypatch):
    monkeypatch.setattr(_ai, "C_FIT", _ai.C_FIT)


def test_default_c_is_unchanged():
    assert _ai.C_FIT == 0.045
    a, _ = _ai.arrest_length(200.0, 50.0, 0.5, 2000.0, delta=1.0, L_t=3.0)
    b, _ = _ai.arrest_length(200.0, 50.0, 0.5, 2000.0, delta=1.0, L_t=3.0, C=0.045)
    assert a == b


def test_c_fit_is_read_at_call_time():
    base, _ = _ai.arrest_length(200.0, 50.0, 0.5, 2000.0, L_t=3.0)
    _ai.C_FIT = 0.09
    twice, _ = _ai.arrest_length(200.0, 50.0, 0.5, 2000.0, L_t=3.0)
    assert twice == pytest.approx(2 * base)


def test_evaluate_follows_c_fit():
    kw = dict(rho=250.0, h=0.8, psi_deg=38.0, E=5e6, sigma_t=3000.0, D_wl=0.05,
              G_wl=0.6e6, theta=40.0)
    a = _ai.evaluate(**kw)["A_ca"]
    _ai.C_FIT = 0.09
    b = _ai.evaluate(**kw)["A_ca"]
    assert np.isfinite(a) and b == pytest.approx(2 * a)


def test_apply_sets_the_module_value():
    cfg = SimpleNamespace(calibration={"version": "t1", "c_fit": 0.06, "size_scale": 1.2})
    assert calibration.apply(cfg, quiet=True) == {"c_fit": 0.06, "size_scale": 1.2}
    assert _ai.C_FIT == 0.06
    assert calibration.size_scale(cfg) == 1.2 and calibration.version(cfg) == "t1"


def test_uncalibrated_slope_changes_nothing():
    cfg = SimpleNamespace(calibration={})
    assert calibration.apply(cfg, quiet=True) == {}
    assert _ai.C_FIT == 0.045 and calibration.size_scale(cfg) == 1.0
    assert calibration.version(cfg) is None
    assert calibration.apply(SimpleNamespace(), quiet=True) == {}


@pytest.mark.parametrize("bad", [{"lambda_factor": 2.0}, {"c_fit": 0.0}, {"size_scale": -1}])
def test_bad_calibration_is_refused(bad):
    with pytest.raises(ValueError):
        calibration.apply(SimpleNamespace(calibration=bad), quiet=True)


def test_toml_section_is_loaded(tmp_path):
    t = tmp_path / "s.toml"
    t.write_text('[calibration]\nversion = "2026-11-09.1"\nc_fit = 0.05\n'
                 'events = ["2026-01-18"]\n')
    cfg = ProjectConfig.from_toml(t)
    assert cfg.calibration["c_fit"] == 0.05
    assert calibration.version(cfg) == "2026-11-09.1"


def test_every_shipped_slope_is_uncalibrated_or_valid():
    import tomllib
    from pathlib import Path
    for t in (Path(__file__).resolve().parent.parent / "slopes").glob("*/slope_config.toml"):
        with open(t, "rb") as f:
            calibration._check(tomllib.load(f).get("calibration", {}))

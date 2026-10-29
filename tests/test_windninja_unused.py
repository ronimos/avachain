"""
tests/test_windninja_unused.py — the WindNinja + RF transport path stays out of every run.

Ron, 2026-10-29: it was tried for between-survey gap fill, performed poorly,
and was never used. It still ran in `all` and run_full_pipeline.sh: `features`
loads the wind library (a fresh deployment has none, so it crashes) and
`train` fits a model only `gap_fill --use-model` reads, which nothing passes.
The code is kept for a future revisit; these keep it from being called.
"""
import importlib.util
import re
import sys
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src" / "avachain"))

import forcing_pipeline as fp                      # noqa: E402

UNUSED = re.compile(r"forcing_pipeline\.py\b.*\b(features|train|validate)\b|--use-model"
                    r"|\$PIPELINE (features|train|validate)\b")


def test_all_skips_the_windninja_steps():
    assert not set(fp.ALL_STEPS) & {"features", "train", "validate"}
    assert fp.ALL_STEPS == ["resample", "transport", "avalanche", "cluster",
                            "gap_fill", "smet"]


def _code_lines(text):
    return [l for l in text.splitlines() if not l.lstrip().startswith("#")]


@pytest.mark.parametrize("script", sorted(p.name for p in REPO.glob("run_*.sh")))
def test_no_run_script_calls_it(script):
    hits = [l for l in _code_lines((REPO / script).read_text()) if UNUSED.search(l)]
    assert not hits, hits


def test_no_generated_script_calls_it():
    spec = importlib.util.spec_from_file_location("install_wn", REPO / "install.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with open(REPO / "slopes" / "Professor_PTC" / "slope_config.toml", "rb") as fh:
        cfg = tomllib.load(fh)
    for render in (mod._render_run_operational, mod._render_run_daily, mod._render_run_snowpack):
        hits = [l for l in _code_lines(render(cfg)) if UNUSED.search(l)]
        assert not hits, (render.__name__, hits)


def test_gap_fill_defaults_to_observed_transport():
    import inspect
    assert inspect.signature(fp.step_gap_fill).parameters["use_model"].default is False

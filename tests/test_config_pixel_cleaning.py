"""
tests/test_config_pixel_cleaning.py — survey pixel cleaning is a config switch.

step_resample cleans survey pixels (spatial_model.load_and_resample_survey)
unless [processing] clean_survey_pixels = false, which resamples with the plain
mean and clip as before 1e71ebb. The switch exists for ablations; the default
must stay on.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "avachain"))

from config import ProjectConfig                                       # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def test_default_is_on():
    assert ProjectConfig().clean_survey_pixels is True
    assert ProjectConfig.from_toml(
        ROOT / "slopes/little_prof/slope_config.toml").clean_survey_pixels is True


def test_toml_switches_it_off(tmp_path):
    t = tmp_path / "s.toml"
    t.write_text('[slope]\nname = "x"\n[processing]\nclean_survey_pixels = false\n')
    assert ProjectConfig.from_toml(t).clean_survey_pixels is False

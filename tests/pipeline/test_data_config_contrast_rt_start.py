"""data_config.py's thesis is that every value it exposes derives from a file on disk, not from a
constant that can drift from it. A hand-typed `CONTRAST_RT_START = 2011` with an adjacent comment
claiming it was "taken from the exhibit's own column name rather than retyped" would contradict
that outright, since nothing would actually parse the column name. So the value comes from
contrast_rt_start(), which parses the real `pooled_RT_<start>-<end>` column name from
contrast_exhibit.csv at runtime.
"""
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "code" / "pipeline"))

import data_config as DC  # noqa: E402


def test_parses_the_real_column_name():
    """The real contrast_exhibit.csv column is 'pooled_RT_2011-25' -- must parse to 2011."""
    assert DC.contrast_rt_start(["pooled_RT_2011-25", "pooled_1989-25", "pooled_1989-25_exCOVID"]) == 2011


def test_parses_a_different_window_correctly():
    """THE CONFIRMED REGRESSION shape: if the exhibit were ever regenerated on a different window,
    the derived value must track it, not silently stay at a stale hardcoded 2011."""
    assert DC.contrast_rt_start(["pooled_RT_2015-25", "pooled_1989-25"]) == 2015
    assert DC.contrast_rt_start(["pooled_1989-25", "pooled_RT_1999-25"]) == 1999


def test_missing_column_raises_not_silently_defaults():
    """If the exhibit's column is renamed or missing entirely, this must fail loudly -- not
    silently fall back to some default that could disagree with the real file."""
    with pytest.raises(ValueError, match="pooled_RT_"):
        DC.contrast_rt_start(["pooled_1989-25", "pooled_1989-25_exCOVID"])


def test_real_contrast_exhibit_parses_to_2011():
    """Direct check against the real, currently-committed contrast_exhibit.csv: must still parse
    to 2011, matching the value the old hardcoded constant asserted (this is a regression guard
    for the fix itself, not a new claim about the data)."""
    contrast_path = ROOT / "results" / "contrast_exhibit.csv"
    if not contrast_path.exists():
        pytest.skip("contrast_exhibit.csv not available")
    import pandas as pd
    contrast = pd.read_csv(contrast_path, index_col=0)
    assert DC.contrast_rt_start(contrast.columns) == 2011

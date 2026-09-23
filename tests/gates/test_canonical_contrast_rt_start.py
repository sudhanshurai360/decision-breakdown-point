"""Regression test: canonical.py's independent
re-derivation for flagship.ar1_rank_short_window_revised used to hard-code the window cutoff as a
bare `2011` literal, instead of parsing it from contrast_exhibit.csv's own `pooled_RT_<start>-<end>`
column name the way the producer (code/pipeline/data_config.py::contrast_rt_start()) already does.
Fixed by duplicating that same parse as canonical.py::_contrast_rt_start() -- mirrors
tests/pipeline/test_data_config_contrast_rt_start.py's own cases so the two copies cannot silently
drift apart from each other.
"""
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "code" / "gates"))

import canonical as CG  # noqa: E402


def test_parses_the_real_column_name():
    assert CG._contrast_rt_start(["pooled_RT_2011-25", "pooled_1989-25", "pooled_1989-25_exCOVID"]) == 2011


def test_parses_a_different_window_correctly():
    """THE CONFIRMED REGRESSION shape: if the exhibit were ever regenerated on a different window,
    the derived value must track it, not silently stay at a stale hardcoded 2011."""
    assert CG._contrast_rt_start(["pooled_RT_2015-25", "pooled_1989-25"]) == 2015
    assert CG._contrast_rt_start(["pooled_1989-25", "pooled_RT_1999-25"]) == 1999


def test_missing_column_raises_not_silently_defaults():
    with pytest.raises(ValueError, match="pooled_RT_"):
        CG._contrast_rt_start(["pooled_1989-25", "pooled_1989-25_exCOVID"])


def test_real_contrast_exhibit_parses_to_2011():
    contrast_path = ROOT / "results" / "contrast_exhibit.csv"
    if not contrast_path.exists():
        pytest.skip("contrast_exhibit.csv not available")
    import pandas as pd
    contrast = pd.read_csv(contrast_path, index_col=0)
    assert CG._contrast_rt_start(contrast.columns) == 2011


def test_matches_data_configs_own_contrast_rt_start_on_the_same_input():
    """The two copies (data_config.py's producer-side parse, canonical.py's independent
    re-derivation) must agree on every input, not just the one real committed file -- otherwise a
    future edit to one could silently diverge from the other without either test catching it."""
    sys.path.insert(0, str(ROOT / "code" / "pipeline"))
    import data_config as DC
    cases = [
        ["pooled_RT_2011-25", "pooled_1989-25"],
        ["pooled_RT_2015-25", "pooled_1989-25"],
        ["pooled_1989-25", "pooled_RT_1999-25"],
    ]
    for cols in cases:
        assert CG._contrast_rt_start(cols) == DC.contrast_rt_start(cols)


def test_matches_data_configs_own_error_path_too():
    """The success-path cross-check above
    doesn't prove the two copies fail the same way -- a duplicated implementation that silently
    diverged only on the missing-column path would pass every case above and still be wrong. Both
    must raise the same exception type on the same malformed input, not just agree when a valid
    column exists."""
    sys.path.insert(0, str(ROOT / "code" / "pipeline"))
    import data_config as DC
    bad_cols = ["pooled_1989-25", "pooled_1989-25_exCOVID"]
    with pytest.raises(ValueError):
        CG._contrast_rt_start(bad_cols)
    with pytest.raises(ValueError):
        DC.contrast_rt_start(bad_cols)

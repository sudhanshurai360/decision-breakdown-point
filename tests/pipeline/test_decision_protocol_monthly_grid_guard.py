"""Regression test: `_monthly_grid()` used `pd.date_range(s.index.min(), s.index.max(), freq="MS")` with no guard
against a non-day=1-normalized index -- if the caller's index isn't already month-start aligned
(e.g. day=15), the grid silently contains NONE of the caller's real timestamps (verified live: an
index of 2025-01-15/02-15/03-15 produces a grid of only 2025-02-01/03-01, dropping Jan-15
entirely), and every downstream `.reindex()` in this module would then silently return NaN/
misaligned results with no error -- exactly the failure mode this module exists to prevent.
Currently inert (multiseries/data/state_ur.parquet and unrate_vintages.parquet, the only two real
inputs, are both 100% day=1 aligned) but worth failing loudly instead of silently misaligning.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "code" / "pipeline"))

import decision_protocol as DP  # noqa: E402


def test_monthly_grid_raises_on_a_non_day1_normalized_index():
    s = pd.Series([1.0, 2.0, 3.0], index=pd.to_datetime(["2025-01-15", "2025-02-15", "2025-03-15"]))
    with pytest.raises(ValueError, match="day=1-normalized"):
        DP._monthly_grid(s)


def test_monthly_grid_still_works_on_a_real_day1_normalized_index():
    s = pd.Series([1.0, 2.0, 3.0], index=pd.to_datetime(["2025-01-01", "2025-02-01", "2025-03-01"]))
    grid = DP._monthly_grid(s)
    assert list(grid) == list(s.index)


def test_monthly_grid_demonstrates_the_silent_misalignment_this_guard_prevents():
    """Direct proof of the failure mode the guard closes: WITHOUT the guard, a non-normalized
    index produces a grid missing every one of the caller's real timestamps."""
    s = pd.Series([1.0, 2.0, 3.0], index=pd.to_datetime(["2025-01-15", "2025-02-15", "2025-03-15"]))
    unguarded_grid = pd.date_range(s.index.min(), s.index.max(), freq="MS")
    assert not any(ts in unguarded_grid for ts in s.index), (
        "sanity check: confirms the unguarded date_range() genuinely drops every real timestamp, "
        "proving the guard is protecting against a real (if currently inert) failure mode"
    )

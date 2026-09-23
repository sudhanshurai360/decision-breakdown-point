"""Regression tests: walkforward.load_vintages() used to drop NaN-valued vintage rows up front and panel_as_of()/
realized() used groupby.last() (which skips NaN), so an observation that a later vintage REMOVED
from the series (ALFRED records the removal as a "." row with the new realtime_start) was silently
replaced by the last vintage that still carried it. On the real curated panel this resurrected
DGORDER's pre-1992 SIC-basis history (408 months), BUSINV 1980-12..1991-12 (133) and PI
1946-01..1958-12 (156) onto the live series -- data that exists in no vintage -- into every model
fit, every MASE scale and the scoring truth. Removed observations must stay removed as of the
vintage that removed them, while an as-of date BEFORE the removal must still see the value.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "code" / "pipeline"))

import walkforward as WF  # noqa: E402


def _vintages():
    """Month 1990-01 published at 100.0 in vintage 1990-02, removed ('.') in vintage 2001-05;
    month 1995-01 published at 200.0 in 1995-02 and revised to 210.0 in 2001-05."""
    return pd.DataFrame({
        "date": pd.to_datetime(["1990-01-01", "1990-01-01", "1995-01-01", "1995-01-01"]),
        "realtime_start": pd.to_datetime(["1990-02-15", "2001-05-25", "1995-02-15", "2001-05-25"]),
        "value": [100.0, float("nan"), 200.0, 210.0],
    })


def test_removed_observation_is_absent_from_the_latest_vintage_truth():
    r = WF.realized(_vintages())
    assert r.date.tolist() == [pd.Timestamp("1995-01-01")]
    assert r.value.tolist() == [210.0]


def test_as_of_before_the_removal_still_sees_the_observation():
    p = WF.panel_as_of(_vintages(), "2000-12-31")
    assert p.date.tolist() == [pd.Timestamp("1990-01-01"), pd.Timestamp("1995-01-01")]
    assert p.value.tolist() == [100.0, 200.0]


def test_as_of_after_the_removal_does_not_resurrect_it():
    """THE CORE REGRESSION: the old code returned 100.0 for 1990-01 here (fell back to the
    1990-02 vintage), a value the 2001-05 vintage had explicitly dropped."""
    p = WF.panel_as_of(_vintages(), "2001-06-01")
    assert p.date.tolist() == [pd.Timestamp("1995-01-01")]
    assert p.value.tolist() == [210.0]


def test_load_vintages_keeps_the_removal_rows_for_the_selectors_to_see():
    """The three real curated series with a discontinued history: their latest-vintage truth must
    start where the live (post-redefinition) series starts, not at the obsolete history's start."""
    expected_start = {"DGORDER": "1992-02-01", "BUSINV": "1992-01-01", "PI": "1959-01-01"}
    for sid, start in expected_start.items():
        path = WF.RAW / f"{sid}_alfred_vintages.parquet"
        if not path.exists():
            pytest.skip(f"{path.name} not available")
        v = WF.load_vintages(sid)
        assert v.value.isna().any(), f"{sid}: the removal rows must survive load_vintages()"
        r = WF.realized(v, sid)
        assert str(r.date.min().date()) == start, f"{sid}: truth starts {r.date.min().date()}, expected {start}"

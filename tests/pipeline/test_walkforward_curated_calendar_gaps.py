"""Regression test for realized()'s calendar-gap guard (code/pipeline/walkforward.py) across the
12-series curated panel.

UNEMPLOY_alfred_vintages.parquet has a real Sept-2025->Nov-2025 calendar gap (an ALFRED data
fact, not a code defect), and realized() raises ValueError(...calendar gap...) for it. This test
doesn't assert a specific pass/fail split (that would just re-encode the current data's state as
a hardcoded expectation and fail loudly -- appropriately -- the next time the data changes);
instead it fails loudly with a clear message when the ACTUAL state diverges from what's
documented here, so a future re-run of this test after a data refresh tells a maintainer to go
re-verify and update KNOWN_GAP_SERIES below, rather than this file silently drifting from reality.
"""
from pathlib import Path

import pandas as pd
import pytest

import walkforward as WF

CURATED = ["BOPGEXP", "BOPGIMP", "BUSINV", "DGORDER", "HOUST", "M2SL",
           "PAYEMS", "PCE", "PI", "RSAFS", "TCU", "UNEMPLOY"]

# The known-current state at the time this test was written -- kept here, not in a bare
# assertion, precisely so a future divergence names itself instead of just failing opaquely.
KNOWN_GAP_SERIES = {"UNEMPLOY"}


def _curated_series_present():
    pred = WF.RAW.parent.parent / "results" / "breadth_distinct" / "breadth_distinct_predictions.csv"
    if not Path(pred).exists():
        pytest.skip("breadth_distinct_predictions.csv not present in this environment")
    return set(pd.read_csv(pred)["series"].unique())


def test_curated_panel_membership_matches_expected_12():
    present = _curated_series_present()
    assert present == set(CURATED), (
        f"the curated panel's series list has changed ({sorted(present)}); update CURATED above "
        f"and re-verify which of them currently hit a calendar-gap crash in realized()."
    )


def test_curated_calendar_gap_state_matches_documented_state():
    """Locks in the CURRENT reality of the UNCONDITIONAL scan (realized() with guard_before=None):
    11 of 12 curated series pass; UNEMPLOY raises on its permanent Oct-2025 absence (a real ALFRED
    data fact, not a code defect). run() now passes guard_before=f"{eval_end}-01-01", so
    UNEMPLOY DOES regenerate (see the test below); this test pins the raw scan. If it starts failing, the data
    (or the fix) has changed and code/pipeline/walkforward.py's guard-comment claim needs a fresh
    re-verification, not a silent edit."""
    actually_broken = set()
    for sid in CURATED:
        try:
            v = WF.load_vintages(sid)
            WF.realized(v, sid)
        except ValueError as e:
            if "calendar gap" in str(e):
                actually_broken.add(sid)
            else:
                raise
    assert actually_broken == KNOWN_GAP_SERIES, (
        f"realized()'s calendar-gap failures ({sorted(actually_broken)}) no longer match the "
        f"documented state ({sorted(KNOWN_GAP_SERIES)}) -- re-verify and update KNOWN_GAP_SERIES "
        f"above, and re-check code/pipeline/walkforward.py's realized() guard behavior."
    )


# realized()'s guard is SCOPED to the truth months the positional consumer (year_scales(),
# year < eval_end) can reach, so run() does not abort on UNEMPLOY's permanent Oct-2025 absence that
# no scored row touches. The unconditional scan (guard_before=None) must keep raising -- that is
# the conservative default any new caller gets.
def test_scoped_guard_lets_run_read_unemploy_but_unconditional_scan_still_raises():
    v = WF.load_vintages("UNEMPLOY")
    with pytest.raises(ValueError, match="calendar gap"):
        WF.realized(v, "UNEMPLOY")
    with pytest.raises(ValueError, match="calendar gap"):
        WF.realized(v, "UNEMPLOY", guard_before="2026-01-01")   # the gap lies inside the guarded range
    truth = WF.realized(v, "UNEMPLOY", guard_before="2025-01-01")   # what run(eval_end=2025) passes
    assert pd.Timestamp("2025-10-01") not in set(truth["date"])     # the absence is honoured, not filled


def test_scoped_guard_is_a_strict_prefix_of_the_unconditional_scan():
    """Synthetic: a gap AT the guard boundary is inside the scan iff it lies strictly before it."""
    months = [f"{y}-{m:02d}-01" for y in (2020, 2021) for m in range(1, 13)]
    months.remove("2021-06-01")   # one-month hole inside 2021
    v = pd.DataFrame({"date": pd.to_datetime(months), "realtime_start": pd.Timestamp("2022-01-01"),
                      "value": 1.0})
    with pytest.raises(ValueError, match="calendar gap"):
        WF.realized(v, "SYN")
    with pytest.raises(ValueError, match="calendar gap"):
        WF.realized(v, "SYN", guard_before="2022-01-01")
    # 23 = the WHOLE synthetic series (24 months minus the removed 2021-06), not just 2020's rows:
    # guard_before stops the gap CHECK at that date, it does not truncate realized()'s output.
    assert len(WF.realized(v, "SYN", guard_before="2021-01-01")) == 23

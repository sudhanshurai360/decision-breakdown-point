"""sahm_evans_realtime.py's onsets() NaN propagation -- the same defect class as
state_eb_decision.py's eb_on() (see tests/pipeline/test_eb_on_undetermined.py for that function's
own tests).

THE HAZARD: a plain `hot = sig >= thr` silently reads NaN (from the real Oct-2025 UNRATE gap,
propagated correctly by decision_protocol.py's calendar-aware helpers) as "not hot," which can
fabricate a spurious onset immediately after the gap or silently miss a genuine onset whose date
fell inside it.

THE RULE: `hot` is nullable ("boolean" dtype), pd.NA wherever `sig` is NaN. A transition is
reported as a genuine onset only when both the month itself and the immediately preceding determined
state are known -- conservative by design: an onset touching an undetermined boundary is excluded
from the returned list, not guessed at either way. This can undercount a real onset immediately
following a gap (a state that went off-then-on-again during it would not be detected) -- a
deliberate tradeoff, since fabricating a date is worse than omitting an uncertain one.

A trap this arrangement creates, guarded below: `prev is False` tracks the previous determined
state, but pandas' nullable "boolean" Series yields non-Python-bool scalars on iteration, so
`is False` never matches unless each value is coerced with `bool(h)` first. Without that coercion
onsets() returns nothing at all, even in the ordinary no-gap case -- and silently, with no
traceback, which is why the most basic possible case is pinned explicitly below.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import sahm_evans_realtime as SER


def _idx(n=12, start="2020-01-01"):
    return pd.date_range(start, periods=n, freq="MS")


def test_onsets_no_gap_matches_original_transition_detection():
    """Baseline: with no NaN anywhere, behavior is unchanged from the original plain comparison --
    this fix changes NaN handling only."""
    sig = pd.Series([0.2, 0.2, 0.6, 0.6, 0.6, 0.6], index=_idx(6))
    assert [d.strftime("%Y-%m") for d in SER.onsets(sig, 0.5)] == ["2020-03"]


def test_onsets_two_genuine_onsets_no_gap():
    sig = pd.Series([0.2, 0.6, 0.2, 0.6, 0.2, 0.2], index=_idx(6))
    assert [d.strftime("%Y-%m") for d in SER.onsets(sig, 0.5)] == ["2020-02", "2020-04"]


def test_onsets_gap_far_from_any_transition_does_not_affect_detection():
    """A gap mid-way through an already-hot run must not disturb a genuine, earlier, unrelated
    onset detection."""
    sig = pd.Series([0.2, 0.2, 0.6, 0.6, 0.6, np.nan, 0.6, 0.6], index=_idx(8))
    assert [d.strftime("%Y-%m") for d in SER.onsets(sig, 0.5)] == ["2020-03"]


def test_onsets_genuine_onset_right_after_gap_is_conservatively_excluded():
    """The case the old code got wrong: a genuine onset immediately following an undetermined
    month. Old behavior (NaN silently read as not-hot) would have fabricated an onset here. Fixed
    behavior: excluded, since we cannot confirm the state was not already hot going into the gap."""
    sig = pd.Series([0.2, 0.2, 0.2, 0.2, 0.2, np.nan, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6], index=_idx())
    assert SER.onsets(sig, 0.5) == []


def test_onsets_already_hot_across_gap_produces_no_spurious_extra_onset():
    """Already hot before AND after a gap: must report exactly the one genuine onset (from the
    Python-bool-identity `prev` tracking's initial False), never a spurious second one manufactured
    at the gap boundary itself."""
    sig = pd.Series([0.6, 0.6, 0.6, 0.6, 0.6, np.nan, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6], index=_idx())
    assert [d.strftime("%Y-%m") for d in SER.onsets(sig, 0.5)] == ["2020-01"]


def test_onsets_returns_python_bool_comparable_prev_not_numpy_scalar_identity_bug():
    """Regression lock for the bug found while writing this fix: iterating a pandas nullable
    "boolean" Series does not yield values with the SAME identity as Python's `False`, so
    `prev is False` silently never matches unless `h` is coerced with `bool()` first. This test
    would have caught that immediately (it asserts onsets() finds ANY onset at all in the most
    basic possible case)."""
    sig = pd.Series([0.2, 0.6], index=_idx(2))
    assert len(SER.onsets(sig, 0.5)) == 1


def test_hot_dtype_is_nullable_boolean():
    """onsets()'s `hot` construction lives inline in the function, so its source line is lifted out
    of sahm_evans_realtime.py and executed here: a NaN month must stay pd.NA in a nullable "boolean"
    Series, not collapse to a determined False. Replace the `.mask(sig.isna())` in the module and
    the pd.isna() assertion below fails."""
    src = Path(SER.__file__).read_text(encoding="utf-8")
    hits = [ln.strip() for ln in src.splitlines() if ln.strip().startswith("hot = ")]
    assert len(hits) == 1, f"expected exactly one `hot = ...` line in sahm_evans_realtime.py, found {hits}"

    sig = pd.Series([0.2, np.nan, 0.6])
    ns = {"sig": sig, "thr": 0.5, "pd": pd}
    exec(hits[0], ns)
    hot = ns["hot"]

    assert hot.dtype == "boolean"
    assert hot.iloc[0] == False   # noqa: E712
    assert pd.isna(hot.iloc[1])
    assert hot.iloc[2] == True    # noqa: E712

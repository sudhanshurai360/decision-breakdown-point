"""state_eb_vintage.py's own logic (`initial_release_panel`'s archive-start/dropna-ordering
filter, `_scored_window`'s panel_start/win_start arithmetic, `_window_restrict`'s window
restriction, and the per-state vintage-revision-SD accumulation it feeds), following the same
pattern as tests/pipeline/test_social_choice_power.py and tests/pipeline/test_null_sensitivity.py.
`canonical.py`'s `vintage.*` CHECKS only re-aggregate this file's own already-computed CSV columns
(`_vint().episodes_initial.sum()` etc.) -- nothing independently re-derives the underlying
panel-construction, window, or revision-SD math against a known-correct hand-computed answer. These
tests use small, hand-built synthetic inputs, independent of the real ALFRED vintage archive and
real state UR panel.

Also pins the window restriction: `_window_restrict` is the single shared restriction that both
`vintage.revision_sd`'s diff and episodes_initial/episodes_revised's `E` go through, so neither is
computed over the full, unrestricted panel.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "code" / "pipeline"))
sys.path.insert(0, str(ROOT / "code" / "instrument"))

import state_eb_vintage as SEV  # noqa: E402


# ------------------------------------------------------------------------------------------------
# initial_release_panel(): archive-start exclusion + the dropna-before-groupby ordering fix
# ------------------------------------------------------------------------------------------------

def test_initial_release_panel_excludes_rows_at_the_archive_start_snapshot(tmp_path, monkeypatch):
    """A row whose realtime_start EQUALS its state's archive_start is the initial ALFRED snapshot,
    not a genuine release -- must be excluded. A row strictly after archive_start is a genuine
    release -- must be included with its own real value."""
    df = pd.DataFrame([
        # DD's initial snapshot: realtime_start == archive_start(DD) == 2007-06-01 -> excluded
        {"state": "DD", "date": "2007-01-01", "realtime_start": "2007-06-01", "value": 3.0},
        # DD's genuine subsequent releases -> included
        {"state": "DD", "date": "2007-06-01", "realtime_start": "2007-07-01", "value": 3.1},
        {"state": "DD", "date": "2007-07-01", "realtime_start": "2007-08-01", "value": 3.2},
    ])
    df["date"] = pd.to_datetime(df["date"]); df["realtime_start"] = pd.to_datetime(df["realtime_start"])
    vint_path = tmp_path / "synthetic_vintages.parquet"
    df.to_parquet(vint_path)
    monkeypatch.setattr(SEV, "VINT", vint_path)

    ir, archive_start = SEV.initial_release_panel()

    assert archive_start["DD"] == pd.Timestamp("2007-06-01")
    assert pd.Timestamp("2007-01-01") not in ir.index  # the snapshot row must not appear at all
    assert ir.loc[pd.Timestamp("2007-06-01"), "DD"] == pytest.approx(3.1)
    assert ir.loc[pd.Timestamp("2007-07-01"), "DD"] == pytest.approx(3.2)


def test_initial_release_panel_dropna_before_groupby_prevents_mispaired_realtime_start(
    tmp_path, monkeypatch
):
    """The reason `initial_release_panel()` drops NaN values BEFORE its `groupby(...).first()`
    (state_eb_vintage.py:36-38), reproduced directly here: if
    dropna(subset=["value"]) happened AFTER (or never), pandas' GroupBy.first(skipna=True) picks
    each output COLUMN's first non-null value independently -- for a (state, date) group whose
    EARLIEST realtime_start row has value=NaN, that would pair the EARLIEST (snapshot-equal)
    realtime_start with a LATER row's real value, and since that mispaired realtime_start then
    equals archive_start exactly, the genuine later release would be wrongly EXCLUDED by the
    `realtime_start > archive_start` filter -- even though its own true realtime_start is well
    after archive_start.

    Constructed so the two behaviors diverge observably: state 'CC', reference month 2008-01-01
    has an archive-snapshot row (realtime_start == archive_start(CC) == 2007-06-01) whose value is
    NaN, plus a genuine subsequent release (realtime_start=2007-07-01, value=5.0). The dropna-first
    fix must include (2008-01-01, 'CC') = 5.0; the pre-fix ordering would have excluded it
    entirely."""
    df = pd.DataFrame([
        # CC's archive_start marker (a genuine, non-NaN snapshot row, so archive_start survives
        # the dropna even though the OTHER snapshot row below does not).
        {"state": "CC", "date": "2006-01-01", "realtime_start": "2007-06-01", "value": 2.0},
        # THE mispaired-value regression case: snapshot row for 2008-01-01 has a NaN value ...
        {"state": "CC", "date": "2008-01-01", "realtime_start": "2007-06-01", "value": np.nan},
        # ... and a genuine subsequent release for the SAME reference month, well after archive_start.
        {"state": "CC", "date": "2008-01-01", "realtime_start": "2007-07-01", "value": 5.0},
        # A couple of ordinary genuine releases so CC has normal multi-month coverage too.
        {"state": "CC", "date": "2007-06-01", "realtime_start": "2007-07-01", "value": 4.5},
        {"state": "CC", "date": "2007-07-01", "realtime_start": "2007-08-01", "value": 4.6},
    ])
    df["date"] = pd.to_datetime(df["date"]); df["realtime_start"] = pd.to_datetime(df["realtime_start"])
    vint_path = tmp_path / "synthetic_vintages_nan.parquet"
    df.to_parquet(vint_path)
    monkeypatch.setattr(SEV, "VINT", vint_path)

    ir, archive_start = SEV.initial_release_panel()

    assert archive_start["CC"] == pd.Timestamp("2007-06-01")
    # The genuine release must be present, correctly valued -- NOT dropped by a mispaired
    # realtime_start that would have equaled archive_start.
    assert pd.Timestamp("2008-01-01") in ir.index
    assert ir.loc[pd.Timestamp("2008-01-01"), "CC"] == pytest.approx(5.0)
    # The 2006-01-01 snapshot-equal row must still be excluded (ordinary archive-start filtering).
    assert pd.Timestamp("2006-01-01") not in ir.index


# ------------------------------------------------------------------------------------------------
# _scored_window(): panel_start / win_start arithmetic, hand-computed independently of DateOffset
# ------------------------------------------------------------------------------------------------

def _hand_add_months(ts: pd.Timestamp, months: int) -> pd.Timestamp:
    """Independent month-addition, deliberately NOT using pd.DateOffset (what the real
    `_scored_window` uses) -- plain year/month integer arithmetic instead, a genuinely separate
    code path for the regression check below."""
    total = (ts.year * 12 + (ts.month - 1)) + months
    y, m = divmod(total, 12)
    return pd.Timestamp(year=y, month=m + 1, day=1)


def test_scored_window_panel_start_is_earliest_all_states_covered_month():
    """panel_start must be the earliest month at which EVERY column (state) is non-NaN -- not the
    earliest month ANY state has data, and not skewed by a state with a later start."""
    idx = pd.date_range("2005-01-01", periods=12, freq="MS")
    ir = pd.DataFrame(index=idx, columns=["AA", "BB", "CC"], dtype=float)
    ir["AA"] = 1.0                                    # AA covered from the very start
    ir.loc[idx[0]:idx[2], "BB"] = np.nan               # BB starts 3 months late (idx[3] onward)
    ir.loc[idx[3]:, "BB"] = 2.0
    ir.loc[idx[0]:idx[5], "CC"] = np.nan               # CC starts latest, 6 months in (idx[6])
    ir.loc[idx[6]:, "CC"] = 3.0

    panel_start, win_start = SEV._scored_window(ir, lookback_months=26)

    assert panel_start == idx[6]                       # the LATEST of the three individual starts
    assert win_start == _hand_add_months(idx[6], 26)    # independent hand-computed cross-check


def test_scored_window_win_start_respects_a_different_lookback():
    idx = pd.date_range("2010-03-01", periods=3, freq="MS")
    ir = pd.DataFrame(1.0, index=idx, columns=["X"])
    panel_start, win_start = SEV._scored_window(ir, lookback_months=5)
    assert panel_start == idx[0]
    assert win_start == _hand_add_months(idx[0], 5) == pd.Timestamp("2010-08-01")


# ------------------------------------------------------------------------------------------------
# _window_restrict(): the shared restriction that applies to BOTH
# vintage.revision_sd's diff and episodes_initial/episodes_revised's E
# ------------------------------------------------------------------------------------------------

def test_window_restrict_matches_a_hand_written_row_by_row_loop_on_a_series():
    idx = pd.date_range("2000-01-01", periods=10, freq="MS")
    common = idx[2:8]                                  # emulate the state's own warm-up cutoff
    win_start = idx[4]
    x = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], index=idx)

    out = SEV._window_restrict(x, common, win_start)

    # Independent hand-written loop, not reusing pandas boolean-mask reindex/dropna machinery.
    expected = {}
    for d in common:
        if d < win_start:
            continue
        v = x.get(d, np.nan)
        if pd.isna(v):
            continue
        expected[d] = v
    assert dict(out) == expected
    assert list(out.index) == sorted(expected.keys())   # order preserved (already ascending)


def test_window_restrict_matches_a_hand_written_loop_on_a_dataframe_any_nan_row_dropped():
    """A DataFrame row must be dropped if ANY column is NaN (matches E.dropna()'s default
    row-wise behavior, used for the multi-model trigger-status frame in main())."""
    idx = pd.date_range("2000-01-01", periods=6, freq="MS")
    common = idx  # no warm-up trimming here, isolate the win_start + NaN-row behavior
    win_start = idx[1]
    df = pd.DataFrame({
        "a": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0],
        "b": [1.0, 2.0, 3.0, np.nan, 5.0, 6.0],
    }, index=idx)

    out = SEV._window_restrict(df, common, win_start)

    expected_rows = []
    for i, d in enumerate(idx):
        if d < win_start:
            continue
        row = df.iloc[i]
        if row.isna().any():
            continue
        expected_rows.append(d)
    assert list(out.index) == expected_rows
    # idx[0] excluded by win_start, idx[2] excluded by NaN in 'a', idx[3] excluded by NaN in 'b'
    assert expected_rows == [idx[1], idx[4], idx[5]]


# ------------------------------------------------------------------------------------------------
# The flip ("tied models disagree") definition, which lives inline in state_eb_vintage.py's main()
# and so cannot be imported: its source line is lifted out of the module and executed here, then
# independently re-derived via a hand-written per-row loop rather than pandas' vectorized any()/all().
# ------------------------------------------------------------------------------------------------

def _production_flip_source() -> str:
    """The module's own `flip = ...` line, verbatim. Change the definition in state_eb_vintage.py --
    to `any` alone, to `~all` alone, to a majority rule -- and the executed line below stops
    matching the hand-written loop."""
    src = (ROOT / "code" / "pipeline" / "state_eb_vintage.py").read_text(encoding="utf-8")
    hits = [ln.strip() for ln in src.splitlines() if ln.strip().startswith("flip = ")]
    assert len(hits) == 1, f"expected exactly one `flip = ...` definition in state_eb_vintage.py, found {hits}"
    return hits[0]


def test_flip_definition_matches_a_hand_written_loop_over_a_synthetic_boolean_frame():
    E = pd.DataFrame({
        "ar1":   [True,  True,  False, True,  False],
        "ets":   [True,  False, False, True,  False],
        "theta": [True,  False, False, False, True],
    })
    ns = {"E": E}
    exec(_production_flip_source(), ns)
    flip = ns["flip"]

    hand_flip = []
    for _, row in E.iterrows():
        vals = list(row)
        hand_flip.append(any(vals) and not all(vals))
    assert list(flip) == hand_flip
    # sanity on the individual rows: row0 all-True (not a flip), row1/row3 mixed (flip),
    # row2 all-False (not a flip), row4 mixed (flip)
    assert hand_flip == [False, True, False, True, True]


# ------------------------------------------------------------------------------------------------
# vintage.revision_sd's actual accumulation logic: per-state _window_restrict'd diffs, concatenated
# across states, then mean/std -- independently re-derived here via hand-rolled Python arithmetic
# (not pandas' .mean()/.std()) over a small multi-state synthetic scenario, exercising the REAL
# `_window_restrict` function (the fixed code) end to end at the aggregation level.
# ------------------------------------------------------------------------------------------------

def test_vintage_revision_sd_accumulation_matches_hand_rolled_mean_and_std():
    idx = pd.date_range("2009-01-01", periods=8, freq="MS")
    win_start = idx[2]
    # Two synthetic "states," each with its own initial-release (u_ir) vs revised (u) UR path and
    # its own `common` warm-up window (mirroring main()'s per-state loop, minus the heavy
    # nowcast/MCS machinery which this test intentionally does not exercise -- those are covered
    # by state_eb_decision.py's own extensive existing test suite).
    states = {
        "S1": dict(
            u=pd.Series([5.0, 5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6], index=idx),
            u_ir=pd.Series([5.0, 5.0, 5.4, 5.4, 5.9, 5.4, 5.5, 6.1], index=idx),
            common=idx[1:7],   # excludes the first and last month, like a state's own warm-up cut
        ),
        "S2": dict(
            u=pd.Series([4.0, 4.0, 4.0, 4.1, 4.2, np.nan, 4.4, 4.5], index=idx),
            u_ir=pd.Series([4.0, 4.0, 4.3, 4.3, 4.7, 4.0, 4.1, 4.9], index=idx),
            common=idx[2:8],
        ),
    }

    diff_parts = []
    for s, d in states.items():
        sdiff = SEV._window_restrict(d["u_ir"] - d["u"], d["common"], win_start)
        diff_parts.append(sdiff)
    diff = pd.concat(diff_parts)

    # Independent hand-written re-derivation: recompute per-state, per-month diffs directly from
    # the same raw inputs via a plain Python loop (no pandas reindex/dropna machinery at all),
    # applying the identical win_start/common/NaN restriction by hand, then hand-roll mean/std
    # (ddof=1, matching pandas' default, used by the module's `diff.std()`).
    hand_vals = []
    for s, d in states.items():
        for dt in d["common"]:
            if dt < win_start:
                continue
            uv = d["u"].get(dt, np.nan)
            iv = d["u_ir"].get(dt, np.nan)
            if pd.isna(uv) or pd.isna(iv):
                continue
            hand_vals.append(iv - uv)

    assert len(diff) == len(hand_vals)
    assert sorted(diff.tolist()) == pytest.approx(sorted(hand_vals))

    n = len(hand_vals)
    hand_mean = sum(hand_vals) / n
    hand_var = sum((v - hand_mean) ** 2 for v in hand_vals) / (n - 1)
    hand_sd = math.sqrt(hand_var)

    assert float(diff.mean()) == pytest.approx(hand_mean)
    assert float(diff.std()) == pytest.approx(hand_sd)


def test_vintage_revision_sd_excludes_months_before_win_start_and_outside_common():
    """Negative-space check: a state-month with a genuinely large (initial - revised) gap must be
    excluded from the accumulation entirely if it falls before win_start or outside `common`, even
    though it is a perfectly valid, non-NaN observation -- pinning that the window restriction (not
    just NaN-dropping) is what sets vintage.revision_sd's population."""
    idx = pd.date_range("2009-01-01", periods=5, freq="MS")
    win_start = idx[3]
    common = idx[1:5]   # idx[0] excluded by `common` itself (a pre-warm-up month)
    u = pd.Series([5.0, 5.0, 5.0, 5.0, 5.0], index=idx)
    # A LARGE gap planted at idx[0] (pre-common) and idx[2] (in common, but before win_start) --
    # neither should survive the restriction; only idx[3]/idx[4] (a small, genuine 0.1pp gap) should.
    u_ir = pd.Series([50.0, 5.0, 50.0, 5.1, 5.1], index=idx)

    out = SEV._window_restrict(u_ir - u, common, win_start)

    assert list(out.index) == [idx[3], idx[4]]
    assert out.tolist() == pytest.approx([0.1, 0.1])


def test_scored_window_default_lookback_is_zero():
    """The reported arms are determinable from the first
    all-state initial release, so the default window starts AT panel_start. The 26-month look-back
    (SEV.LOOKBACK_MONTHS) applies only to the superseded each-leg construct. Pins the default so a
    silent return to `lookback_months=LOOKBACK_MONTHS` (which moved the Section 7 multipliers from
    3-4x to 7-9x by dropping the 2008-09 trigger-on wave) cannot recur unnoticed."""
    idx = pd.date_range("2005-01-01", periods=60, freq="MS")
    ir = pd.DataFrame({"A": [float("nan")] * 6 + [5.0] * 54, "B": [5.0] * 60}, index=idx)
    panel_start, win_start = SEV._scored_window(ir)
    assert panel_start == idx[6]
    assert win_start == panel_start
    assert SEV.LOOKBACK_MONTHS == 26

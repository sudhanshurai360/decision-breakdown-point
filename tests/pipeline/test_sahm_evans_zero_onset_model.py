"""_run_protocol()'s zero-onset-model handling.

THE HAZARD: `_run_protocol()` (sahm_evans_realtime.py, `def _run_protocol(...)` through its
`return summary_rows, detail, onset_rows` -- lines 115-168 as of this writing) computes, for each real onset `ro`,
the nearest onset date each tied model called (`calls = {m: nearest(m_on[m]) for m in tied}`). If a
tied model has ZERO onsets across its ENTIRE history for a rule, `nearest()` returns None for that
model (there is nothing to find the nearest of). Building `vals = [c for c in calls.values() if
c is not None]` would silently drop the never-firing model from the spread computation instead of
treating it as a miss. With only one OTHER model contributing a real date, `len(vals) <= 1`, so
`spread = 0` (the "no data to compare" fallback) -- labeling a divergence driven entirely by one
never-firing model as "genuine" (spread=0, i.e. essentially no divergence) instead of a MISS
(spread > 12, i.e. `misses += 1`). This is inert on the real published-protocol data (n_missed=0
on both rules, since every real tied model fires at least once in practice), but it is a live,
otherwise-untested code path.

THE RULE: a call of None (i.e. `len(vals) < len(calls)`, meaning at least one tied model never
fired this rule at all) forces `spread = float("inf")`, landing in the same ">12" miss bucket every
other genuine long-timing-gap miss lands in, rather than being silently excluded from the
comparison. This test constructs a model that provably never fires (EVANS rule, whose realized/
model 3-month-average path is analytically controlled below) and checks that behavior directly
against the real `_run_protocol()` function -- not a reimplementation.
"""
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "code" / "pipeline"))
sys.path.insert(0, str(ROOT / "code" / "instrument"))

import sahm_evans_realtime as SER  # noqa: E402
import decision_protocol as DP  # noqa: E402


def _build_scenario():
    """Constructs a 40-month synthetic UNRATE-like panel with ONE single-month spike, and two
    'tied' models:
      - 'a': a perfect forecaster (fc == realized) -- fires the EVANS onset at exactly the same
        month as the real onset (spread 0).
      - 'b': a model that ALWAYS forecasts a constant 4.0, regardless of the true series -- chosen
        (via the algebra in the module docstring above) so that under the "published" protocol's
        3-month average (current=forecast, t-1/t-2=realized), model 'b's avg3 NEVER reaches the
        6.5 EVANS threshold anywhere in its history, even at the two months whose lag legs include
        the real spike. Verified empirically below before using it in the real assertion.
    """
    idx = pd.date_range("2000-01-01", periods=40, freq="MS")
    B, R = 5.0, 10.0
    realized = pd.Series(B, index=idx)
    spike_i = 20
    realized.iloc[spike_i] = R
    fc_a = realized.copy()
    fc_b = pd.Series(4.0, index=idx)
    now = {"a": fc_a, "b": fc_b}
    tied = ["a", "b"]
    return realized, now, tied


def test_scenario_model_b_genuinely_never_fires_the_evans_rule():
    """Precondition check, independent of _run_protocol(): model 'b's avg3 (published protocol)
    never crosses the 6.5 EVANS threshold anywhere across its 40-month history."""
    realized, now, tied = _build_scenario()
    avg3_b = DP.three_month_average(now["b"], realized, "published")
    assert SER.onsets(SER.evans(avg3_b), 6.5) == [], (
        "scenario construction error: model 'b' was supposed to never fire the EVANS rule"
    )
    # Sanity: the real (ground-truth) path DOES cross, and model 'a' (perfect forecaster) matches
    # it exactly, so the scenario genuinely exercises a real divergence, not a no-op.
    avg3_real = DP.three_month_average(realized, realized, "published")
    avg3_a = DP.three_month_average(now["a"], realized, "published")
    real_on = SER.onsets(SER.evans(avg3_real), 6.5)
    assert len(real_on) == 1
    assert SER.onsets(SER.evans(avg3_a), 6.5) == real_on


def test_zero_onset_model_counts_as_a_miss_not_a_genuine_zero_spread():
    """The rule itself, checked against the real _run_protocol(): the EVANS onset on which model
    'b' never fires must be counted in n_missed, not silently treated as a spread=0 'genuine'
    divergence. Excluding 'b's None call from the spread computation would report n_missed=0
    (spread=0) despite a real divergent onset."""
    realized, now, tied = _build_scenario()
    rows, detail, onset_rows = SER._run_protocol("published", now, realized, tied)

    assert detail["EVANS"]["n_onsets"] == 1
    assert detail["EVANS"]["n_divergent"] == 1  # 'a' fires, 'b' never does -> distinct calls

    evans_row = next(r for r in rows if r["rule"] == "EVANS")
    assert evans_row["n_missed"] == 1, (
        "a tied model with zero onsets ever must be counted as a MISS on every real onset, "
        f"got row={evans_row}"
    )
    # median_spread_mo is computed only over "genuine" (spread<=12) divergences; the one
    # divergence here is entirely the zero-onset-model miss, so genuine is empty -> median 0.0.
    assert evans_row["median_spread_mo"] == 0.0

    # onset_rows is the per-onset detail behind S7's illustrative dates -- lock its shape here too.
    evans_onset = next(r for r in onset_rows if r["rule"] == "EVANS")
    assert evans_onset["divergent"] is True
    assert evans_onset["missed"] is True
    assert evans_onset["spread_mo"] is None   # the zero-onset-model branch: spread is +inf, stored as None


def test_a_model_that_fires_late_but_at_least_once_is_still_a_genuine_spread_not_a_miss():
    """Negative-space check: the fix must not turn EVERY divergence into a miss -- a model that
    fires at some point in its history, just far from this particular real onset, is still
    governed by the pre-existing '>12 months = miss' logic, not the new zero-onset branch."""
    idx = pd.date_range("2000-01-01", periods=40, freq="MS")
    B, R = 5.0, 10.0
    realized = pd.Series(B, index=idx)
    spike_i = 20
    realized.iloc[spike_i] = R
    fc_a = realized.copy()
    # model 'c' fires its OWN, unrelated onset late in the series (month 38) but never anywhere
    # near the real onset at month ~20/21 -- it does have at least one onset, so it must NOT hit
    # the new zero-onset branch; it should fall through to the existing large-spread miss logic.
    fc_c = pd.Series(4.0, index=idx)
    fc_c.iloc[36:] = 10.0
    now = {"a": fc_a, "c": fc_c}
    tied = ["a", "c"]
    rows, detail, onset_rows = SER._run_protocol("published", now, realized, tied)
    evans_row = next(r for r in rows if r["rule"] == "EVANS")
    assert detail["EVANS"]["n_onsets"] == 1
    assert detail["EVANS"]["n_divergent"] == 1
    # far apart in time (>12mo) -> still a miss, but via the ordinary days-based spread, not the
    # zero-onset branch (model 'c' genuinely has its own onset, just a distant one)
    assert evans_row["n_missed"] == 1


def test_all_tied_models_zero_onsets_still_counts_as_a_missed_divergent_onset():
    """The unanimous-miss case (currently inert on real data). When EVERY tied model has zero
    onsets ever for a rule -- not just one, as in the scenario above -- `months` is
    {m: "NONE" for all m}, i.e. only ONE distinct value ("NONE"). A bare `distinct > 1` gate would
    read that as NOT divergent and skip n_div/n_missed entirely, even though `spread` is correctly
    `inf` and onset_rows' own `missed` column independently reads True for the same onset -- so the
    two persisted CSVs (sahm_evans_summary.csv's aggregate, sahm_evans_onsets.csv's per-onset
    detail) would silently disagree. A UNANIMOUS miss is the most severe divergence this file
    tracks, not the absence of one."""
    idx = pd.date_range("2000-01-01", periods=40, freq="MS")
    B, R = 5.0, 10.0
    realized = pd.Series(B, index=idx)
    realized.iloc[20] = R
    # Both tied models forecast a constant that never crosses the EVANS threshold anywhere.
    fc_b = pd.Series(4.0, index=idx)
    fc_c = pd.Series(4.2, index=idx)
    now = {"b": fc_b, "c": fc_c}
    tied = ["b", "c"]
    avg3_real = DP.three_month_average(realized, realized, "published")
    assert len(SER.onsets(SER.evans(avg3_real), 6.5)) == 1, "scenario error: expected one real onset"
    for m, fc in now.items():
        avg3 = DP.three_month_average(fc, realized, "published")
        assert SER.onsets(SER.evans(avg3), 6.5) == [], f"scenario error: model {m!r} was supposed to never fire"

    rows, detail, onset_rows = SER._run_protocol("published", now, realized, tied)
    evans_row = next(r for r in rows if r["rule"] == "EVANS")
    assert detail["EVANS"]["n_onsets"] == 1
    assert detail["EVANS"]["n_divergent"] == 1, (
        "a real onset that every tied model unanimously misses must still count as divergent"
    )
    assert evans_row["n_missed"] == 1, (
        f"a unanimous zero-onset miss must be counted in n_missed, got row={evans_row}"
    )
    evans_onset = next(r for r in onset_rows if r["rule"] == "EVANS")
    assert evans_onset["missed"] is True
    # the two CSVs must now agree: onset_rows.missed True <=> counted in the aggregate n_missed
    assert evans_row["n_missed"] == sum(1 for r in onset_rows if r["rule"] == "EVANS" and r["missed"])

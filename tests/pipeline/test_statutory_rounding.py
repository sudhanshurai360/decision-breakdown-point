"""decision_protocol.py's (e)(3)/(e)(1)(ii) statutory rounding cascade (`_round_half_away_from_zero`,
`statutory_round_tur`, `statutory_lookback_ok`, `eb_on_statutory_rounding`). This two-stage cascade
makes a specific, load-bearing claim: it is BIDIRECTIONAL (can move the EB trigger's on/off call
either way relative to the raw, unrounded ratio), unlike a one-directional rule that could only
ever push values ON. These tests establish that claim in the repo itself, with hand-verified
numeric cases, rather than leaving it to an external validation nobody can re-run.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "code" / "pipeline"))

import decision_protocol as DP  # noqa: E402


def test_round_half_away_from_zero_boundary_cases_both_signs():
    """Half-away-from-zero, not Python/numpy's default half-to-even: x.x5 always rounds AWAY from
    zero, for both positive and negative x, at multiple decimal places."""
    assert DP._round_half_away_from_zero(2.5, 0) == 3.0
    assert DP._round_half_away_from_zero(-2.5, 0) == -3.0
    assert DP._round_half_away_from_zero(0.05, 1) == pytest.approx(0.1)
    assert DP._round_half_away_from_zero(-0.05, 1) == pytest.approx(-0.1)
    assert DP._round_half_away_from_zero(1.125, 2) == pytest.approx(1.13)
    assert DP._round_half_away_from_zero(-1.125, 2) == pytest.approx(-1.13)


def test_round_half_away_from_zero_non_boundary_values():
    """Ordinary (non-.5-boundary) values round the way everyone expects, at 1 and 2 decimals."""
    assert DP._round_half_away_from_zero(6.44, 1) == pytest.approx(6.4)
    assert DP._round_half_away_from_zero(6.46, 1) == pytest.approx(6.5)
    assert DP._round_half_away_from_zero(1.144, 2) == pytest.approx(1.14)
    assert DP._round_half_away_from_zero(1.146, 2) == pytest.approx(1.15)


def test_round_half_away_from_zero_is_nan_safe():
    """NaN must propagate as NaN, not raise -- decision_protocol.py's real callers rely on this for
    the state UR panel's genuine calendar gap."""
    out = DP._round_half_away_from_zero(np.array([1.25, np.nan, -1.25]), 1)
    assert np.isnan(out[1])
    assert out[0] == pytest.approx(1.3)
    assert out[2] == pytest.approx(-1.3)


def test_statutory_round_tur_rounds_to_tenths():
    """20 CFR 615.12(e)(3): rounded to the tenths place."""
    s = pd.Series([6.44, 6.46, 6.45, -6.45])
    out = DP.statutory_round_tur(s)
    assert list(out) == [6.4, 6.5, 6.5, -6.5]


def test_statutory_lookback_ok_ordinary_case():
    """A clear, non-boundary case: 8.0/7.0 = 114.3% >= 110 -> True; 7.0/8.0 = 87.5% -> False."""
    assert bool(DP.statutory_lookback_ok(pd.Series([8.0]), pd.Series([7.0])).iloc[0]) is True
    assert bool(DP.statutory_lookback_ok(pd.Series([7.0]), pd.Series([8.0])).iloc[0]) is False


def test_statutory_lookback_ok_propagates_nan_as_pdNA():
    """Undetermined (pd.NA), never silently False, when either input is NaN -- the same property
    eb_on()'s sibling comparison has (see tests/pipeline/test_eb_on_undetermined.py), checked here
    for this function directly."""
    out = DP.statutory_lookback_ok(pd.Series([np.nan]), pd.Series([100.0]))
    assert out.iloc[0] is pd.NA
    out2 = DP.statutory_lookback_ok(pd.Series([100.0]), pd.Series([np.nan]))
    assert out2.iloc[0] is pd.NA


def test_rounding_cascade_is_bidirectional_not_only_ever_on():
    """THE CORE CLAIM this whole module section exists to establish, and the property a
    one-directional rounding rule lacks (it could only ever push a state-month ON,
    never OFF). Two hand-verified cases, computed independently of decision_protocol.py's own code
    (see this test file's own docstring derivation), where rounding EACH already-(e)(3)-rounded
    Trigger Value before forming their ratio moves the 110%-threshold call in OPPOSITE directions
    relative to the raw, unrounded ratio:

      Case 1 (raw OFF -> cascade ON):  current=6.15, prior=5.60
        raw ratio    = 6.15/5.60*100    = 109.821...%  (OFF, < 110)
        (e)(3)-round = round(6.15,1)/round(5.60,1) = 6.2/5.6 = 110.714...%  (ON, >= 110)

      Case 2 (raw ON -> cascade OFF):  current=6.04, prior=5.49
        raw ratio    = 6.04/5.49*100    = 110.018...%  (ON, >= 110)
        (e)(3)-round = round(6.04,1)/round(5.49,1) = 6.0/5.5 = 109.090...%  (OFF, < 110)

    A rounding rule that could only ever push values ON could never produce Case 2 -- that is
    exactly the property being pinned here.
    """
    # Case 1: raw OFF, cascade ON
    current1, prior1 = 6.15, 5.60
    raw_ratio1 = current1 / prior1 * 100
    assert raw_ratio1 < 110, f"test setup error: raw case 1 should be OFF, got {raw_ratio1:.3f}"
    current1_r = DP.statutory_round_tur(pd.Series([current1]))
    prior1_r = DP.statutory_round_tur(pd.Series([prior1]))
    cascade1_on = bool(DP.statutory_lookback_ok(current1_r, prior1_r).iloc[0])
    assert cascade1_on is True, "expected the rounding cascade to flip case 1 OFF -> ON"

    # Case 2: raw ON, cascade OFF
    current2, prior2 = 6.04, 5.49
    raw_ratio2 = current2 / prior2 * 100
    assert raw_ratio2 >= 110, f"test setup error: raw case 2 should be ON, got {raw_ratio2:.3f}"
    current2_r = DP.statutory_round_tur(pd.Series([current2]))
    prior2_r = DP.statutory_round_tur(pd.Series([prior2]))
    cascade2_on = bool(DP.statutory_lookback_ok(current2_r, prior2_r).iloc[0])
    assert cascade2_on is False, "expected the rounding cascade to flip case 2 ON -> OFF"


def test_eb_on_statutory_rounding_end_to_end_on_a_synthetic_series():
    """Integration check: eb_on_statutory_rounding() wires date_rolling_mean/date_shift + the
    rounding cascade together correctly on a small, hand-constructed monthly series, producing a
    determinate boolean where enough history exists and pd.NA before the lookback window opens."""
    idx = pd.date_range("2020-01-01", periods=30, freq="MS")
    # A flat 5.0% series for 24 months, then a jump to 7.0% for the last 6 months -- the jump
    # should eventually trigger both the >=6.5% flat leg and the >=110%-of-prior-year lookback leg.
    values = [5.0] * 24 + [7.0] * 6
    s = pd.Series(values, index=idx)
    out = DP.eb_on_statutory_rounding(s)
    assert out.iloc[:26].isna().all() or (out.iloc[:26] == False).all(), (
        "expected no trigger while the series is flat at 5.0% (below the 6.5% floor)"
    )
    assert bool(out.iloc[-1]) is True, "expected the sustained jump to 7.0% to trigger by month-end"

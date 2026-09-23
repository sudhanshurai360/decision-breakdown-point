"""m4_submissions_audit.py's score() feeds S9's headline Pareto-dominance/margin numbers, so it
needs a test with a hand-verified known-correct answer (as m4_perseries_kstar.py's
decision_breakdown()/winner_stability() call in the same file group has). This locks in the exact
formulas on a hand-computed 3-point example, and separately locks in `_leading_run()`: it must
TRUNCATE at the first interior NaN, where a plain `row.dropna()` would silently compact (shift)
the array instead.
"""
import numpy as np
import pytest

import m4_submissions_audit as MSA


def test_score_matches_hand_computed_metrics():
    """y_pred=[2,3,5], y_true=[1,3,4] -> e=[-1,0,-1], ae=[1,0,1]. Hand-computed by hand outside the
    module: sMAPE=0.2963, MAPE=0.4167, MASE=0.3333, RMSSE=0.4082 (scale_abs=2, scale_sq=4)."""
    y_pred = np.array([2.0, 3.0, 5.0])
    y_true = np.array([1.0, 3.0, 4.0])
    out = MSA.score(y_pred, y_true, scale_abs=2.0, scale_sq=4.0)
    assert out["sMAPE"] == pytest.approx(0.2962962962962963)
    assert out["MAPE"] == pytest.approx(0.4166666666666667)
    assert out["MASE"] == pytest.approx(0.3333333333333333)
    assert out["RMSSE"] == pytest.approx(0.408248290463863)


def test_score_truncates_to_the_shorter_of_pred_and_true_not_error():
    """A mismatched-length pair is truncated to n=min(len), not padded or errored -- documents the
    existing, deliberate `n = min(len(y_pred), len(y_true))` behavior rather than leaving it
    unverified."""
    y_pred = np.array([2.0, 3.0, 5.0, 9.0])   # one extra, trailing point
    y_true = np.array([1.0, 3.0, 4.0])
    out = MSA.score(y_pred, y_true, scale_abs=2.0, scale_sq=4.0)
    assert out["MASE"] == pytest.approx(0.3333333333333333), (
        "the 4th prediction must be dropped by the min(len) truncation, not scored against nothing "
        "or raising -- same numeric answer as the exact-length case above"
    )


def test_score_returns_none_on_empty_input():
    out = MSA.score(np.array([]), np.array([]), scale_abs=1.0, scale_sq=1.0)
    assert out is None


def test_leading_run_matches_dropna_for_right_padding_only():
    """M4's own Test/Train CSVs are right-padded only -- _leading_run() must agree with dropna()
    exactly in that shape, since it is a drop-in replacement for it in load_data()."""
    import pandas as pd
    row = pd.Series([1.0, 2.0, 3.0, np.nan, np.nan])
    got = MSA._leading_run(row)
    want = row.dropna().to_numpy(float)
    np.testing.assert_array_equal(got, want)


def test_leading_run_truncates_positionally_at_an_interior_gap_instead_of_compacting():
    """The bug this replaces: row.dropna() on [1, 2, NaN, 4, 5] silently COMPACTS to [1, 2, 4, 5] --
    shifting horizon steps 4 and 5 two positions to the LEFT, which would misalign forecast-to-actual
    pairing (h=4's value scored against h=2's slot). _leading_run() instead stops at the first gap:
    [1, 2] -- fewer points, but never misaligned."""
    import pandas as pd
    row = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0])
    got = MSA._leading_run(row)
    np.testing.assert_array_equal(got, np.array([1.0, 2.0]))
    # the bug this guards against, made explicit:
    compacted_by_dropna = row.dropna().to_numpy(float)
    assert not np.array_equal(got, compacted_by_dropna), (
        "if this ever matches dropna()'s compacted output again, the misalignment bug is back"
    )


def test_leading_run_returns_full_array_when_no_nan_present():
    import pandas as pd
    row = pd.Series([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(MSA._leading_run(row), np.array([1.0, 2.0, 3.0]))

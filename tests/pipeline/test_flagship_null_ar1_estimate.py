"""flagship_null_ar1.py applies the AR(1) recalibration to the flagship exhibit's OWN calibrated
null (not just the curated 24-cell panel) -- multi-seed/AR(1) null calibration is an area where a
silent error moves headline manuscript numbers. `main()` itself does file I/O (reads real CSVs,
calls emit()) and is exercised end-to-end by a full producer rerun; this file unit-tests the one
pure, independently-testable computation -- the lag-1 OLS AR(1) coefficient estimator -- against
hand-derived expected values and its documented edge-case guards (NaN/inf skip, too-short skip,
zero-variance skip).

The calibration estimator `estimate_flagship_ar1()` runs on pairwise loss DIFFERENTIALS (see that
function's docstring); the per-model hand-computed tests below target
`estimate_flagship_ar1_levels()`, the identical per-model computation kept for disclosure, and
further tests cover the pairwise estimator.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "code" / "pipeline"))

import numpy as np
import pandas as pd
import pytest

from flagship_null_ar1 import estimate_flagship_ar1, estimate_flagship_ar1_levels


def test_ar1_estimate_matches_hand_computed_value_on_a_simple_ramp():
    """x = [0,1,2,3,4], demeaned -> [-2,-1,0,1,2]. phi = sum(x[1:]*x[:-1]) / sum(x[:-1]**2)
    = ((-1*-2)+(0*-1)+(1*0)+(2*1)) / (4+1+0+1) = 4/6 = 0.6667 -- computed independently by hand,
    not by re-deriving the function's own formula."""
    sb = pd.DataFrame({"modelA": [0.0, 1.0, 2.0, 3.0, 4.0]})
    out = estimate_flagship_ar1_levels(sb)
    assert len(out) == 1
    assert out.iloc[0]["model"] == "modelA"
    assert out.iloc[0]["ar1_estimate"] == pytest.approx(4 / 6, abs=1e-9)


def test_ar1_estimate_is_minus_one_for_a_perfectly_alternating_series():
    """A hand-built alternating series (sign flips every step, magnitude-matched) is perfectly
    NEGATIVELY autocorrelated: phi = -1 exactly, by construction -- not a statistical near-minus-one,
    an exact one."""
    x = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
    sb = pd.DataFrame({"m": x})
    out = estimate_flagship_ar1_levels(sb)
    # demeaned x is already [1,-1,1,-1,1,-1] (mean 0); x[:-1]=[1,-1,1,-1,1], x[1:]=[-1,1,-1,1,-1]
    # num = sum(x[1:]*x[:-1]) = -1-1-1-1-1 = -5, den = sum(x[:-1]**2) = 5 -> phi = -1.0
    assert out.iloc[0]["ar1_estimate"] == pytest.approx(-1.0, abs=1e-9)


def test_ar1_estimate_skips_a_model_with_nan_or_inf():
    """A model with a NaN/inf value must be explicitly skipped with a [warn] print, not silently
    propagated into a NaN phi that pandas' median() would then quietly drop. Verify the skip
    actually happens."""
    sb = pd.DataFrame({
        "good": [0.0, 1.0, 2.0, 3.0, 4.0],
        "has_nan": [1.0, np.nan, 3.0, 4.0, 5.0],
        "has_inf": [1.0, np.inf, 3.0, 4.0, 5.0],
    })
    out = estimate_flagship_ar1_levels(sb)
    assert set(out["model"]) == {"good"}


def test_ar1_estimate_skips_a_model_with_fewer_than_5_observations():
    sb = pd.DataFrame({"too_short": [1.0, 2.0, 3.0, 4.0]})  # len 4 < 5
    out = estimate_flagship_ar1_levels(sb)
    assert len(out) == 0


def test_ar1_estimate_skips_a_constant_model_zero_variance():
    """A constant series has std()==0 -- explicitly skipped (den<=0 would otherwise divide by
    zero / the std()==0 guard catches it first)."""
    sb = pd.DataFrame({"constant": [7.0, 7.0, 7.0, 7.0, 7.0, 7.0]})
    out = estimate_flagship_ar1_levels(sb)
    assert len(out) == 0


def test_ar1_estimate_processes_each_model_column_independently():
    """Multiple models in one call -- each gets its own independently-computed estimate, not
    cross-contaminated by another column's values."""
    sb = pd.DataFrame({
        "ramp": [0.0, 1.0, 2.0, 3.0, 4.0],
        "constant": [5.0] * 5,
        "alternating": [1.0, -1.0, 1.0, -1.0, 1.0],
    })
    out = estimate_flagship_ar1_levels(sb)
    assert set(out["model"]) == {"ramp", "alternating"}
    ramp_row = out[out.model == "ramp"].iloc[0]
    alt_row = out[out.model == "alternating"].iloc[0]
    assert ramp_row["ar1_estimate"] == pytest.approx(4 / 6, abs=1e-9)
    # demeaned alternating (mean 0.2) has a nonzero but still computable phi -- just confirm it's
    # finite and independent of "ramp"'s own value, not that a "constant" column ever appears.
    assert np.isfinite(alt_row["ar1_estimate"])
    assert alt_row["ar1_estimate"] != pytest.approx(ramp_row["ar1_estimate"])


# --- pairwise (differential) estimator: the calibration basis -----------------------------------

def test_pairwise_estimate_matches_hand_computed_value_on_a_ramp_minus_constant():
    """ramp [0..4] minus constant [7]*5 = [-7,-6,-5,-4,-3]; demeaned -> [-2,-1,0,1,2]; phi = 4/6."""
    sb = pd.DataFrame({"ramp": [0.0, 1.0, 2.0, 3.0, 4.0], "const": [7.0] * 5})
    out = estimate_flagship_ar1(sb)
    assert len(out) == 1 and out.iloc[0]["pair"] == "ramp-const"
    assert out.iloc[0]["ar1_estimate"] == pytest.approx(4 / 6, abs=1e-9)


def test_pairwise_estimate_covers_every_pair_and_skips_nan_columns():
    sb = pd.DataFrame({
        "a": [0.0, 1.0, 2.0, 3.0, 4.0],
        "b": [1.0, -1.0, 1.0, -1.0, 1.0],
        "c": [2.0, 3.0, 5.0, 7.0, 11.0],
        "bad": [1.0, np.nan, 3.0, 4.0, 5.0],
    })
    out = estimate_flagship_ar1(sb)
    assert set(out["pair"]) == {"a-b", "a-c", "b-c"}


def test_pairwise_estimate_is_invariant_to_a_common_component():
    """A series added to EVERY column cancels exactly in every pairwise differential."""
    rng = np.random.default_rng(0)
    base = pd.DataFrame({m: rng.normal(size=40) for m in ("m1", "m2", "m3")})
    common = np.cumsum(rng.normal(size=40))
    shifted = base.add(common, axis=0)
    a = estimate_flagship_ar1(base).set_index("pair")["ar1_estimate"]
    b = estimate_flagship_ar1(shifted).set_index("pair")["ar1_estimate"]
    assert np.allclose(a.sort_index(), b.sort_index(), atol=1e-12)

"""best_arm.py's studentized-gap computation must not fall back on a bare `se > 0 else np.inf`
guard. A GENUINE tie between arm i and the best arm -- both years' losses numerically identical, so
dd.mean()==0 and se==0 -- would then be recorded as t=np.inf, the maximally-SEPARATED value, when
it is actually the maximally NON-IDENTIFIED case:
best_arm.py's own sample-complexity proxy H = sum(1/max(t,1e-6)**2) then contributes 0 for that arm
(i.e. "needs zero more samples to resolve"), exactly backwards for an exact tie, which should need
the MOST samples (Delta_i=0 => 1/Delta_i^2 -> infinity, correctly capped by H's own 1e-6 floor).

Same defect class, same fix shape as fragility.py/mcs_test.py's own degenerate-floor fixes:
a scale-relative floor (matching breakdown_number()/mcs()'s own per-pair convention: raw loss
magnitude of the two arms, not the differential's own scale, which is exactly zero for a tie and
would degenerate its own floor to zero too) distinguishes a genuine tie (mean AND se both
noise-floor-zero -> t=0) from a real deterministic advantage (se collapsed to noise but the mean gap
is a real nonzero value -> t=np.inf is still correct there).

Verified genuinely latent against the real 12-series curated panel: 0/real pairs have se==0 today
(direct instrumentation of every pairwise studentized_gap computation in best_arm.py's main()).
"""
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "code" / "pipeline"))
sys.path.insert(0, str(ROOT / "code" / "instrument"))

import best_arm as BA  # noqa: E402


def test_genuine_tie_gives_zero_not_infinity():
    """THE CONFIRMED REGRESSION: arm i identical to the best arm every year (dd.mean()==0,
    se==0 by construction). Before this fix, t was np.inf (maximally separated); it must now be
    0.0 (maximally non-identified) so H's own 1/max(t,1e-6)^2 registers this as needing the most
    evidence, not the least."""
    li = np.array([1.0, 2.0, 3.0, 1.5])
    lbest = li.copy()
    t = BA.studentized_gap(li, lbest)
    assert t == 0.0, f"expected t=0.0 for a genuine tie, got {t}"


def test_deterministic_nonzero_advantage_still_gives_infinity():
    """Positive control: se collapses to (numerically) zero but the mean gap is a REAL, nonzero
    value -- arm i is worse than the best arm by exactly the same nonzero amount every single
    year (deterministic, noise-free advantage). This is genuinely maximal evidence; np.inf is
    still the correct value here, the fix must not break this case."""
    lbest = np.array([1.0, 2.0, 3.0, 1.5])
    li = lbest + 0.5  # constant nonzero gap every year, zero variance in the differential
    t = BA.studentized_gap(li, lbest)
    assert np.isinf(t) and t > 0, f"expected +inf for a deterministic nonzero advantage, got {t}"


def test_ordinary_noisy_gap_unaffected():
    """Sanity: an ordinary case with genuine sampling variance (se well above the floor) must
    still compute the plain t-statistic, unaffected by the new floor logic."""
    rng = np.random.default_rng(0)
    lbest = rng.normal(1.0, 0.05, 20)
    li = lbest + rng.normal(0.3, 0.05, 20)
    t = BA.studentized_gap(li, lbest)
    dd = li - lbest
    se = dd.std(ddof=1) / np.sqrt(len(dd))
    expected = dd.mean() / se
    assert t == pytest.approx(expected), f"expected plain t-stat {expected}, got {t}"


def test_scale_relative_floor_not_absolute():
    """The fix's core mathematical property: the tie/advantage distinction is RELATIVE to the
    arms' own loss magnitude, not a fixed absolute threshold -- identical relative structure at
    very different loss levels must classify the same way."""
    for level in (1.0, 1e-6, 1e6):
        lbest = np.array([1.0, 2.0, 3.0, 1.5]) * level
        li_tie = lbest.copy()
        li_adv = lbest + 0.5 * level
        assert BA.studentized_gap(li_tie, lbest) == 0.0, f"level={level}: tie case"
        t_adv = BA.studentized_gap(li_adv, lbest)
        assert np.isinf(t_adv) and t_adv > 0, f"level={level}: advantage case, got {t_adv}"


def test_no_real_series_currently_have_a_degenerate_pair():
    """Confirms the fix is genuinely latent against the real curated panel: instruments every
    pairwise studentized_gap computation best_arm.py's own main() performs and asserts none of
    them hit the new floor branches (se==0 for every real pair today)."""
    pred = ROOT / "multiseries" / "results" / "breadth_distinct" / "breadth_distinct_predictions.csv"
    if not pred.exists():
        pytest.skip("frozen loss matrix not available")
    import pandas as pd
    d = pd.read_csv(pred)
    degenerate_pairs = 0
    for sid, g in d.groupby("series"):
        models = sorted(g.model.unique())
        years = sorted(g.year.unique())
        L = np.array([[g[(g.model == m) & (g.year == y)]["sae"].mean() for m in models] for y in years])
        mean = L.mean(0)
        best = np.argsort(mean)[0]
        for i in range(len(models)):
            if i == best:
                continue
            dd = L[:, i] - L[:, best]
            se = dd.std(ddof=1) / np.sqrt(len(dd))
            if se == 0:
                degenerate_pairs += 1
    assert degenerate_pairs == 0, (
        f"{degenerate_pairs} real pair(s) now hit the degenerate-se floor -- the fix is no longer "
        f"purely latent, verify best_arm.csv/canonical values before trusting them unchanged"
    )

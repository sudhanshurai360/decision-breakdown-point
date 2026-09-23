"""mcs_test.py's mcs()
"raw" elimination path (the default, used by every call site) must not rely on a bare
`se <= 0` degenerate-standard-error guard. A
bootstrap differential that is constant in EXACT arithmetic does not reliably compute to
std()==0: summing/dividing block means leaves floating-point residue whose size tracks the loss
LEVEL (the tool package's own investigation found identical-models cases where se came out exactly
0.0 at one loss magnitude and 2.6e-23 at another). The exact-zero test is magnitude-dependent dead
code: a tiny POSITIVE se sails past it and feeds `t_obs = abs(obs_diff) / se`, inflating t_obs to
an astronomically large but FINITE value from pure floating-point noise, not real evidence -- an
unsafe, over-rejecting spurious elimination.

Also: mcs_test.py's `mcs()` needed a `MIN_T_FOR_ZERO_SE` floor but had none defined anywhere in the
file -- fixed by porting it verbatim from the tool package's own selection_fragility.mcs
(MIN_T_FOR_ZERO_SE = 8), which this file's mcs() otherwise mirrors. Without it, a T=1 panel (one
period of evidence) with two models that merely
happen to differ in that single period gets a "certain rejection" (se=0 by construction when every
bootstrap resample is the same lone observation) -- a tool whose central claim is that small
samples cannot identify a winner must not report point-identification from one period.

Both ported verbatim from selection_fragility.mcs (already independently reviewed there) into
mcs_test.py's "raw" path, which previously had neither.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "code" / "pipeline"))

import mcs_test as MT  # noqa: E402


def test_min_t_for_zero_se_is_defined_and_matches_the_tool_package():
    assert MT.MIN_T_FOR_ZERO_SE == 8


def test_one_period_of_evidence_does_not_certainly_reject():
    """THE CONFIRMED REGRESSION (T-gate half): T=1, two models differ in their one shared period.
    Every bootstrap resample of a single observation IS that observation, so se=0 by construction
    -- exactly the case MIN_T_FOR_ZERO_SE exists to decline certain rejection on. Both models must
    survive; before this fix, model 1 (the "worse" one) was wrongly eliminated from one period."""
    L = np.array([[1.0, 2.0]])  # T=1, K=2
    surv, p = MT.mcs(L, alpha=0.10, B=200, seed=0)
    assert sorted(surv) == [0, 1], (
        f"expected both models to survive one period of evidence, got survivors={surv}, p={p}"
    )


def test_enough_periods_of_genuine_dominance_still_rejects():
    """Positive control for the T-gate: with T well above MIN_T_FOR_ZERO_SE and one model
    GENUINELY beating the other in every single period (real deterministic dominance, not noise),
    mcs() must still correctly eliminate the dominated model -- the gate must not be so
    conservative it breaks legitimate certain-rejection detection."""
    T = 20
    L = np.column_stack([np.full(T, 1.0), np.full(T, 5.0)])  # model 0 strictly better, every period
    surv, p = MT.mcs(L, alpha=0.10, B=200, seed=0)
    assert surv == [0], f"expected model 0 alone to survive genuine dominance over {T} periods, got {surv}"


def test_scale_relative_floor_not_absolute():
    """The fix's core mathematical property: the degenerate-se floor is RELATIVE to the pair's own
    magnitude (per-pair, via max(|dbar[a]|, |dbar[b]|, |obs_diff|)), not a fixed absolute
    threshold -- so identical-shaped panels at very different loss LEVELS must behave the same
    way. Constructs two panels with the same relative structure (model 0 always exactly 1.5x model
    1's loss) at loss levels differing by 6 orders of magnitude, and confirms both eliminate the
    same way."""
    rng = np.random.default_rng(0)
    T = 20
    base = rng.normal(1.0, 0.05, T)
    for level in (1.0, 1e-6):
        L = np.column_stack([base * level, base * 1.5 * level])
        surv, p = MT.mcs(L, alpha=0.10, B=200, seed=0)
        assert surv == [0], f"level={level}: expected model 0 to survive, got {surv}"

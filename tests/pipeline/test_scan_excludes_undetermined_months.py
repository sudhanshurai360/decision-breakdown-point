"""scan()'s (state_eb_decision.py) E_all.dropna() really does exclude undetermined months, given
that eb_on() returns nullable "boolean" dtype -- run against the REAL 51-state panel's real
Oct-2025 gap.

The hazard: `E_all = pd.DataFrame({m: eb_on(...) for m in tied}).reindex(common).dropna()` is a
silent no-op if eb_on() returns plain `bool` dtype, which cannot hold NaN, so nothing is ever
excluded regardless of the gap. These tests lock in that the dropna() does real work -- the exact
downstream mechanism by which undetermined months are kept out of every scan()-derived count
(tot_eval, tot_flip, tot_ep, near/knife shares).

The exclusion does not move any cited eb.* canonical value, for a reason these tests check
directly: the gap lives in the SHARED `realized` series (not in any individual model's forecast),
so eb_on()'s undetermined-ness at a gap-adjacent month is IDENTICAL across every tied model for
that state-month. It can shrink the evaluated-months denominator, but it can never by itself
manufacture or hide a cross-model FLIP (disagreement), since disagreement requires the tied models
to differ, and they are all equally blind together at exactly the same months.
"""
import pandas as pd
import pytest

import state_eb_decision as SEB

# Gate ONLY on the data file's presence, then call load_state_ur() unguarded.
# The previous form wrapped that call in `try/except Exception`, so ANY failure inside
# load_state_ur() -- a genuine regression, not just a missing checkout -- silently turned this
# whole module into green skips. The skip must answer "is the panel on disk?" and nothing else;
# everything after that is allowed to fail loudly.
_HAS_STATE_UR = SEB.CACHE.exists()

pytestmark = pytest.mark.skipif(
    not _HAS_STATE_UR, reason=f"real state UR panel not available ({SEB.CACHE})")

_DF = SEB.load_state_ur() if _HAS_STATE_UR else None


def test_gap_adjacent_months_absent_from_a_real_states_e_all_index():
    """Directly reproduces scan()'s E_all construction for one real state and confirms the two
    real gap-adjacent months (Nov/Dec 2025) are NOT present in its index -- i.e. genuinely
    excluded from evaluation, not just internally NA-flagged and then miscounted anyway."""
    s = _DF.columns[0]
    u = _DF[s].dropna()
    nc = SEB.nowcast(u)
    common = u.index[u.index.year >= u.index.year.min() + SEB.WARMUP_YEARS]
    err = pd.DataFrame({m: (nc[m] - u).abs() for m in SEB.MODELS}).reindex(common).dropna()
    surv, _ = __import__("mcs_test").mcs(err.groupby(err.index.year).mean().to_numpy(float), alpha=0.10, seed=0)
    tied = [list(SEB.MODELS)[i] for i in surv]
    assert len(tied) >= 2, "fixture state must have a genuine tie to exercise E_all construction"
    E_all = pd.DataFrame({m: SEB.eb_on(nc[m].reindex(u.index), u, "published") for m in tied}).reindex(common).dropna()
    assert pd.Timestamp("2025-11-01") not in E_all.index
    assert pd.Timestamp("2025-12-01") not in E_all.index
    # a genuinely determined nearby month must still be present -- the fix must not over-exclude
    assert pd.Timestamp("2025-08-01") in E_all.index


def test_undeterminedness_is_shared_across_tied_models_so_it_cannot_move_a_flip_count():
    """The ANALYTICAL reason the exclusion cannot move any cited eb.* value, checked directly on the
    frame BEFORE dropna(): a flip requires the tied models to disagree, and the gap's
    undeterminedness is shared identically across all of them. So no row may be PARTIALLY
    undetermined -- every row is either fully determined or fully undetermined, and dropna()
    therefore removes whole rows from the denominator without ever touching a disagreement."""
    s = _DF.columns[0]
    u = _DF[s].dropna()
    nc = SEB.nowcast(u)
    common = u.index[u.index.year >= u.index.year.min() + SEB.WARMUP_YEARS]
    tied = list(SEB.MODELS)[:2]   # any 2+ models suffice to exercise the flip/disagreement logic
    E_raw = pd.DataFrame({m: SEB.eb_on(nc[m].reindex(u.index), u, "published") for m in tied}).reindex(common)
    na = E_raw.isna()

    # not vacuous: the real Oct-2025 gap must actually produce undetermined rows here
    assert int(na.all(axis=1).sum()) > 0, "fixture state has no undetermined months to exclude"
    partial = na.any(axis=1) & ~na.all(axis=1)
    assert not partial.any(), (
        f"{s}: {int(partial.sum())} month(s) undetermined for some tied models but not others -- "
        f"the shared-blindness argument for this exclusion no longer holds"
    )
    # and dropna() removes exactly those whole undetermined rows, nothing else
    assert len(E_raw.dropna()) == len(E_raw) - int(na.all(axis=1).sum())

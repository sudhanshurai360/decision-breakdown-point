"""code/instrument/flagship_fragility.py is touched by existing
tests (test_atomic_to_csv.py checks its CSV write path only; test_realised_t_binding_tiebreak.py /
test_election_and_flagship.py cross-check its `runner_up` against a DIFFERENT file's independent
binding-rival logic) -- but nothing anywhere hand-verifies the VALUE-computation logic that is
actually THIS file's own, not F.fragility()'s: the `_fragile_avg = bool(...)` recombination in
main() -- averaged winner_stability vs. F.FRAGILE_STABILITY_THRESH, OR'd with
reversal-and-small-k*/T, re-derived here because it needs the 20-seed-averaged stability, not the
seed-0 value F.fragility()'s own internal `fragile` field used -- and the pooled-MASE AR1 rank
(main()'s `pooled_rank = sb.mean(0).sort_values()`) that flagship.ar1_rank_long_window is emitted
from. (Both are quoted by expression rather than by line number, which drifts: as of this writing
they are at flagship_fragility.py:34-35 and :49, not the ~33-34/~48 an earlier revision recorded.)

Convention note: this codebase's existing style for logic embedded in a producer's main() (not
factored into its own function) is a disclosed "faithful copy" that runs the SAME real building
blocks (F.fragility(), NB.stable_stability() -- both called directly here, not reimplemented) with
the exact formula text copied from flagship_fragility.py's own source -- see
tests/pipeline/test_election_and_flagship.py's `_analyse_realised_t_local` for the established
precedent this file follows. n_boot is reduced from main()'s own 5000 to 1000 for test speed (same
justification test_election_and_flagship.py already uses); NB.stable_benchmark() (the SEPARATE,
much more expensive exchangeable-null computation main() also runs, for the UNRELATED
flagship.stability_null_* emits) is deliberately not exercised here, since `_fragile_avg` never
reads it.

A `test_main_source_still_matches_the_formula_this_file_hand_verifies` guard (last test below) pins
the exact formula/line TEXT in main()'s real source, so a change to the real file that this test
suite's faithful copies would otherwise silently miss is still caught.
"""
import inspect

import numpy as np
import pandas as pd
import pytest

import fragility as F
import flagship_fragility as FF
import null_budget as NB


# =================================================================================================
# Scenario A: a hand-designed 2-model, 6-year panel with a genuine REVERSAL and a small breakdown
# point (k*=1). modB wins 5 of 6 individual
# years (barely, by 0.01 each time) while ar1 wins the pooled (mean) comparison because it posts one
# enormous win in the last year -- the exact "one outlier year decides the pooled verdict" shape
# flagship.k_star/flagship.responsible_years exist to detect.
#   ar1:  [1.01, 1.01, 1.01, 1.01, 1.01, 0.10]   mean = 5.15/6 = 0.858333...
#   modB: [1.00, 1.00, 1.00, 1.00, 1.00, 5.00]   mean = 10.00/6 = 1.666667...
# Per-year winner (lower MASE): modB in years 1-5 (1.00 < 1.01), ar1 in year 6 (0.10 < 5.00) ->
# per-period plurality winner = modB (5 of 6). Pooled (mean) winner = ar1 (0.858 < 1.667).
# Removing year 6 alone: remaining-5 means are ar1=1.01, modB=1.00 -> modB now wins pooled ->
# k*=1, responsible=[2023] (year 6, the last label).
# =================================================================================================

YEARS_A = [2018, 2019, 2020, 2021, 2022, 2023]
AR1_A = np.array([1.01, 1.01, 1.01, 1.01, 1.01, 0.10])
MODB_A = np.array([1.00, 1.00, 1.00, 1.00, 1.00, 5.00])


def _fragility_result_a(n_boot=1000):
    L = {"ar1": AR1_A, "modB": MODB_A}
    w = np.ones(len(YEARS_A))
    return F.fragility(L, w, labels=YEARS_A, n_boot=n_boot, seed=0)


def test_pooled_and_per_period_winner_reversal_and_breakdown_point_hand_verified():
    """Direct call to the real F.fragility() (n_boot reduced from main()'s 5000 for test speed) on
    the hand-designed panel above -- every field checked here is hand-derivable from the raw MASE
    values, independent of calling fragility() a second time."""
    r = _fragility_result_a()
    assert r["pooled_winner"] == "ar1", (
        f"hand-computed pooled means: ar1={AR1_A.mean():.6f} < modB={MODB_A.mean():.6f}"
    )
    assert r["per_period_winner"] == "modB", "modB wins 5 of 6 individual years (1.00 < 1.01 each)"
    assert r["reversal"] is True, "pooled (ar1) and per-period (modB) winners differ"
    assert r["k_star"] == 1
    assert r["responsible"] == [2023], "deleting year 2023 alone flips the pooled winner to modB"
    assert r["k_over_T"] == pytest.approx(1 / 6)
    assert r["runner_up"] == "modB"
    assert r["n_binding_tied"] == 1, "only one possible rival exists with K=2 models"


def test_fragile_flag_faithful_copy_matches_flagship_fragility_source_formula_reversal_case():
    """Faithful copy of flagship_fragility.py::main()'s own `_fragile_avg = bool(...)`
    recombination: `(averaged winner_stability < FRAGILE_STABILITY_THRESH) OR (reversal AND k_over_T <
    FRAGILE_KSTAR_RATIO_THRESH)`. On Scenario A, the averaged stability alone is NOT low enough to
    trigger the first clause (checked explicitly below) -- fragile=True here is driven ENTIRELY by
    the second clause (reversal=True, k_over_T=1/6=0.167 < 0.25), so this genuinely exercises that
    OR branch, not just "some value happened to be True"."""
    r = _fragility_result_a()
    L = {"ar1": AR1_A, "modB": MODB_A}
    w = np.ones(len(YEARS_A))
    stab = NB.stable_stability(L, w, n_boot=1000)  # real function; n_boot reduced for test speed

    stability_clause = stab["value"] < F.FRAGILE_STABILITY_THRESH
    reversal_clause = r["reversal"] and r["k_over_T"] < F.FRAGILE_KSTAR_RATIO_THRESH
    fragile = bool(stability_clause or reversal_clause)

    assert stability_clause is False, (
        f"expected the averaged winner_stability ({stab['value']:.3f}) to be >= "
        f"{F.FRAGILE_STABILITY_THRESH} on this panel (ar1's pooled win is not fragile to period "
        f"resampling) -- if this now fires, the reversal_clause below is no longer the thing "
        f"actually driving `fragile`, and this test needs a new scenario"
    )
    assert reversal_clause is True
    assert fragile is True


def test_fragile_flag_faithful_copy_matches_flagship_fragility_source_formula_no_reversal_case():
    """A second, contrasting panel where ar1 wins EVERY year (no reversal -> the second OR clause
    is structurally False regardless of k*/T) by a healthy, noisy-but-consistent margin (not
    exactly constant, to avoid the degenerate case where every bootstrap resample gives the same
    pooled winner for trivial reasons) -- so stability should be at or near the ceiling, and `fragile`
    should be False via BOTH clauses, not just the one this test happens to check."""
    years = [2018, 2019, 2020, 2021, 2022, 2023]
    ar1 = np.array([1.0, 1.1, 0.9, 1.05, 0.95, 1.0])
    modb = np.array([2.0, 2.2, 1.8, 2.1, 1.9, 2.0])
    L = {"ar1": ar1, "modB": modb}
    w = np.ones(len(years))
    r = F.fragility(L, w, labels=years, n_boot=1000, seed=0)
    stab = NB.stable_stability(L, w, n_boot=1000)

    assert r["reversal"] is False, "ar1 wins every single year here (1.0<2.0, 1.1<2.2, ...)"
    stability_clause = stab["value"] < F.FRAGILE_STABILITY_THRESH
    reversal_clause = r["reversal"] and r["k_over_T"] < F.FRAGILE_KSTAR_RATIO_THRESH
    fragile = bool(stability_clause or reversal_clause)

    assert reversal_clause is False  # structurally, since r["reversal"] is False
    assert stability_clause is False, (
        f"expected a consistent, comfortable per-year margin to give near-rock-solid stability "
        f"(>= {F.FRAGILE_STABILITY_THRESH}); got {stab['value']:.3f}"
    )
    assert fragile is False


def test_ar1_pooled_rank_hand_verified():
    """Faithful copy of flagship_fragility.py::main()'s own `pooled_rank = sb.mean(0).sort_values()`
    line, which flagship.ar1_rank_long_window's `pooled_rank.index.get_loc("ar1") + 1` reads.
    Hand-verified: ar1's pooled mean (0.858333) is lower than modB's (1.666667), so ar1 must be
    rank 1 of 2."""
    sb = pd.DataFrame({"ar1": AR1_A, "modB": MODB_A}, index=YEARS_A)
    pooled_rank = sb.mean(0).sort_values()  # verbatim copy of main()'s own line
    assert list(pooled_rank.index) == ["ar1", "modB"]
    assert pooled_rank.index.get_loc("ar1") + 1 == 1
    assert pooled_rank.iloc[0] == pytest.approx(5.15 / 6)
    assert pooled_rank.iloc[1] == pytest.approx(10.00 / 6)


def test_main_source_still_matches_the_formula_this_file_hand_verifies():
    """Drift guard (same convention as test_density_audit_fetches_data.py's
    `test_main_calls_m_audit_fetch_not_a_hardcoded_path`): the faithful-copy tests above run their
    OWN copy of the `_fragile_avg` formula and the `pooled_rank` line, not flagship_fragility.py's
    real source -- so a change to the REAL formula (e.g. a flipped comparison operator, or a switch
    to the seed-0-only fragile computation) would not be caught by
    those tests alone. This one inspects main()'s actual source text for the exact substrings the
    faithful copies assume, so such a change fails HERE instead of silently diverging."""
    src = inspect.getsource(FF.main)
    assert '_stab["value"] < F.FRAGILE_STABILITY_THRESH' in src
    assert 'r["reversal"] and r["k_over_T"] < F.FRAGILE_KSTAR_RATIO_THRESH' in src
    assert "pooled_rank = sb.mean(0).sort_values()" in src

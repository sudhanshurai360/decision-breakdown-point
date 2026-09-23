"""dmcs.py's symmetric T-asymmetry robustness cut.

The concern it answers: `tied_stat` (the statistical-loss MCS) is computed on the FULL sample
(~36 years) while `tied_dec` (the decision-loss MCS) is computed on `near` (near-trigger months
only, median 9 near-trigger years, range 5-21) -- a real T-asymmetry between the two legs
`dmcs.tied_both` compares. A much-shorter-T MCS eliminates fewer candidates almost mechanically, so
`tied_both=44/44` in every retained state could be an artifact of that asymmetry rather than
genuine ambiguity.

The cut: `tied_stat_near = year_mcs(stat_loss, near)` applies the SAME near-trigger restriction to
the statistical loss, and `dmcs.tied_both_symmetric` compares `dMCS_decision` against
`MCS_stat_near` instead of the full-sample `MCS_stat`. On the real panel `MCS_stat` and
`MCS_stat_near` differ in 27 of the 44 states individually (so the T-restriction is not a no-op),
yet `tied_both_symmetric` is IDENTICAL to `tied_both` (44/44 either way) -- the T-asymmetry was not
the explanation.

Because a full run of dmcs.py takes real time (51-state nowcast + per-state MCS), this test does not
re-run the whole pipeline -- it checks structural invariants against the committed output, following
the same pattern as test_dmcs_near_trigger.py.
"""
import json
from pathlib import Path

import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DMCS_CSV = ROOT / "multiseries" / "results" / "dmcs.csv"
DMCS_CANON = ROOT / "multiseries" / "results" / "canonical" / "dmcs.json"

pytestmark = pytest.mark.skipif(not DMCS_CSV.exists(), reason="dmcs.csv not available")


def _dmcs_df():
    return pd.read_csv(DMCS_CSV)


def test_mcs_stat_near_column_present_and_genuinely_differs_from_full_sample():
    """The symmetric cut must be a real, distinct computation -- not a copy of MCS_stat under a new
    name. If every row were identical, the robustness check would be vacuous."""
    d = _dmcs_df()
    assert "MCS_stat_near" in d.columns
    n_differ = int((d.MCS_stat != d.MCS_stat_near).sum())
    assert n_differ > 0, (
        "MCS_stat_near is identical to MCS_stat in every state -- the near-trigger restriction on "
        "the statistical leg appears to be a no-op, which would make the symmetric robustness cut "
        "meaningless (on the committed panel they differ in 27 of the 44 states)"
    )


def test_tied_both_symmetric_recomputes_correctly_from_committed_csv():
    """Direct re-derivation from the committed dmcs.csv: tied_both_symmetric must equal the same
    formula as tied_both, substituting MCS_stat_near for MCS_stat -- not some other comparison."""
    d = _dmcs_df()
    tied_both = int(((d.dMCS_decision >= d.MCS_stat) & (d.dMCS_decision > 1)).sum())
    tied_both_sym = int(((d.dMCS_decision >= d.MCS_stat_near) & (d.dMCS_decision > 1)).sum())

    frag = json.loads(DMCS_CANON.read_text())
    assert frag["dmcs.tied_both"]["value"] == tied_both
    assert frag["dmcs.tied_both_symmetric"]["value"] == tied_both_sym


def test_separates_symmetric_column_matches_its_own_definition():
    d = _dmcs_df()
    expected = d.dMCS_decision < d.MCS_stat_near
    assert (d.separates_symmetric == expected).all()


def test_code_shape_both_legs_use_near_index_in_the_symmetric_cut():
    """Guard against the exact bug this fix closes: the symmetric statistical leg must be computed
    on `near`, not `common` (which would silently collapse back to the original, unmatched-T
    dmcs.mean_set_statistical and make the robustness cut a duplicate, not a real check)."""
    src = (ROOT / "code" / "pipeline" / "dmcs.py").read_text()
    assert "tied_stat_near = year_mcs(stat_loss, near)" in src


def test_mean_set_statistical_near_registered_in_canonical():
    frag = json.loads(DMCS_CANON.read_text())
    for key in ("dmcs.mean_set_statistical_near", "dmcs.separates_symmetric", "dmcs.tied_both_symmetric"):
        assert key in frag, f"{key} missing from dmcs.json -- the symmetric cut's emit() calls didn't run"


def test_wider_from_singleton_recomputes_correctly_from_committed_csv():
    """tied_both/wider_symmetric do not themselves check whether the statistical leg was ALREADY a
    singleton before the decision loss ran -- on the committed panel, 11 of the 26 states counted
    as "genuine ambiguity" (wider_symmetric) started from a clean, unambiguous MCS_stat_near==1.
    dmcs.wider_from_singleton discloses how many of the wider_symmetric states started from that
    clean baseline. Direct re-derivation from the committed
    dmcs.csv, not a copy of the emitted value -- if the underlying formula or data ever drifts, this
    fails at unit-test speed instead of only being caught by a full 15-minute pipeline rerun."""
    d = _dmcs_df()
    wider_sym = int((d.dMCS_decision > d.MCS_stat_near).sum())
    wider_from_singleton = int(((d.dMCS_decision > d.MCS_stat_near) & (d.MCS_stat_near == 1)).sum())

    frag = json.loads(DMCS_CANON.read_text())
    assert frag["dmcs.wider_symmetric"]["value"] == wider_sym
    assert frag["dmcs.wider_from_singleton"]["value"] == wider_from_singleton
    assert frag["dmcs.wider_from_singleton"]["of"] == wider_sym


def test_wider_from_singleton_state_list_matches_declared_states():
    """The emit's own `states=` extra field must list exactly the states the formula selects --
    catches a future refactor that changes the count but not the disclosed state list, or vice versa."""
    d = _dmcs_df()
    mask = (d.dMCS_decision > d.MCS_stat_near) & (d.MCS_stat_near == 1)
    expected_states = sorted(d[mask].state)

    frag = json.loads(DMCS_CANON.read_text())
    declared_states = frag["dmcs.wider_from_singleton"]["states"].split(";")
    assert declared_states == expected_states


def test_wider_from_singleton_is_not_vacuously_zero_or_equal_to_wider_symmetric():
    """Sanity bound: this disclosure only has real content if it's a genuine strict subset of
    wider_symmetric (some states have a real tie growing wider, not just clean singletons turning
    ambiguous) -- a value of 0 or equal to wider_symmetric would mean the two categories collapsed
    into one, undermining the whole point of the disclosure."""
    frag = json.loads(DMCS_CANON.read_text())
    n = frag["dmcs.wider_from_singleton"]["value"]
    of = frag["dmcs.wider_from_singleton"]["of"]
    assert 0 < n < of, (
        f"wider_from_singleton={n} of {of} -- expected a genuine strict subset (on the committed "
        f"panel, 11 of 26), not 0 (no clean-singleton states) or all of them (no "
        f"genuinely-tied-growing-wider states left)"
    )


# dmcs.wider_from_singleton_also_full_sample (S7 body's Wyoming-only sentence), covered the same
# way this file covers wider_from_singleton itself two tests above: an independent re-derivation
# from the committed dmcs.csv rather than a copy of the emitted value.
def test_wider_from_singleton_also_full_sample_recomputes_correctly_from_committed_csv():
    """dmcs.wider_from_singleton_also_full_sample: of the wider_from_singleton states (decision-loss
    |MCS| strictly wider than a CLEAN near-trigger-restricted statistical singleton, MCS_stat_near==1),
    how many are ALSO a singleton under the FULL-SAMPLE statistical comparison (MCS_stat==1) -- the
    S7 body sentence naming Wyoming as the only one of the 11. Re-derived directly from dmcs.csv's own
    columns, mirroring the exact mask dmcs.py's main() builds (wfs_mask), not a re-read of the
    already-emitted canonical value."""
    d = _dmcs_df()
    wfs_mask = (d.dMCS_decision > d.MCS_stat_near) & (d.MCS_stat_near == 1)
    wider_from_singleton = int(wfs_mask.sum())
    wfs_also_full_sample = int((d[wfs_mask].MCS_stat == 1).sum())

    frag = json.loads(DMCS_CANON.read_text())
    assert frag["dmcs.wider_from_singleton"]["value"] == wider_from_singleton
    assert frag["dmcs.wider_from_singleton_also_full_sample"]["value"] == wfs_also_full_sample
    assert frag["dmcs.wider_from_singleton_also_full_sample"]["of"] == wider_from_singleton


def test_wider_from_singleton_also_full_sample_state_list_matches_formula():
    """The emit's own `states=` field must list exactly the states the wfs_mask & (MCS_stat == 1)
    formula selects -- catches a future refactor that changes the count but not the disclosed state
    list, or vice versa. On the committed panel the set is exactly {WY} among the 11
    wider_from_singleton states. The decision/near legs are floored at the optional-TUR statute
    date while the statistical leg runs on the full record; flooring the statistical leg too would
    empty this set. No state name is pinned here (that would be a data literal, not a property),
    and an empty `states` string must parse as an empty list, not [""], so the formula check cannot
    false-fail on an empty set."""
    d = _dmcs_df()
    wfs_mask = (d.dMCS_decision > d.MCS_stat_near) & (d.MCS_stat_near == 1)
    expected_states = sorted(d[wfs_mask & (d.MCS_stat == 1)].state)

    frag = json.loads(DMCS_CANON.read_text())
    declared = frag["dmcs.wider_from_singleton_also_full_sample"]["states"]
    declared_states = [s for s in declared.split(";") if s]
    assert declared_states == expected_states
    assert len(declared_states) == frag["dmcs.wider_from_singleton_also_full_sample"]["value"]


def test_wider_from_singleton_also_full_sample_is_a_proper_subset():
    """Sanity bound mirroring test_wider_from_singleton_is_not_vacuously_zero_or_equal_to_
    wider_symmetric above: this disclosure only has content if it is strictly less than its own
    `of` (wider_from_singleton) -- otherwise every clean-near-trigger-singleton state would also be
    a clean full-sample singleton, and the distinction the S7 body sentence draws (the rest are
    two- or three-way ties on the full sample) would not exist.

    The structural bound `0 <= n < of` alone is too loose: it passes on the correct value (1) and
    equally on 0 with states="", which is what a mis-floored statistical leg produces. So the
    specific value is pinned as well. If a legitimate change moves it, this test's failure is the
    prompt to re-verify against dmcs.csv and update the pin, not to widen the range again."""
    frag = json.loads(DMCS_CANON.read_text())
    n = frag["dmcs.wider_from_singleton_also_full_sample"]["value"]
    of = frag["dmcs.wider_from_singleton_also_full_sample"]["of"]
    states = frag["dmcs.wider_from_singleton_also_full_sample"]["states"]
    assert (n, of, states) == (1, 11, "WY"), (
        f"wider_from_singleton_also_full_sample=({n}, of={of}, states={states!r}) -- expected "
        f"exactly (1, 11, 'WY'); if this legitimately changed, re-verify against dmcs.csv directly "
        f"(not against this test's own prior value) before updating the pin"
    )
    assert 0 <= n < of  # the structural bound this test also exists to guard, kept as a named check

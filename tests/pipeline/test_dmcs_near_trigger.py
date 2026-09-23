"""dmcs.py's decision loss is NOT the raw 0/1 misclassification rate over EVERY scored month,
and these tests pin the construction that replaces it. Since EB is a rare event
(~2-5% of state-months), an unconditional rate would be dominated by the trivial "everyone agrees
it's off" region and would read as near-zero and near-identical across models regardless of whether
they agree on anything that matters -- producing a headline that reads backwards: "genuine
ambiguity" in 51/51 states, when the real, verified-on-real-data mechanism is the opposite (models
mostly AGREE; the decision-tied set is a superset of the statistical-tied set in 51/51 states,
never smaller -- exactly what base-rate dominance predicts).

So the decision-loss MCS is restricted to near-trigger months: months where at least one of the
four candidate models' statute-rounded 3-month-average TUR sits within [6.0, 7.0], straddling the
6.5 trigger -- computed over all four models unconditionally, NOT restricted to the
statistically-tied subset, matching the phrasing in paper/draft_ssrn.md's Appendix 13.4. That is
feasible for 44 of 51 states (7 drop via year_mcs's existing len(M)<30 floor, no new threshold
introduced).

Because a full run of dmcs.py takes ~15 minutes (nowcast() over 51 states), this test does NOT
re-run the whole pipeline. It checks structural invariants against the committed output, plus a
fast, targeted re-derivation of the near-trigger construction for one real state (CA) -- enough to
catch a regression in the construction itself without paying the full pipeline's runtime.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DMCS_CSV = ROOT / "multiseries" / "results" / "dmcs.csv"
STATE_UR_PARQUET = ROOT / "multiseries" / "data" / "state_ur.parquet"

sys.path.insert(0, str(ROOT / "code" / "pipeline"))
sys.path.insert(0, str(ROOT / "code" / "instrument"))
sys.path.insert(0, str(ROOT / "code" / "gates"))

pytestmark = pytest.mark.skipif(not DMCS_CSV.exists(), reason="dmcs.csv not available")


def _dmcs_df():
    return pd.read_csv(DMCS_CSV)


def test_near_trigger_months_column_present_and_positive():
    d = _dmcs_df()
    assert "near_trigger_months" in d.columns
    assert (d.near_trigger_months > 0).all()


def test_near_trigger_floor_dropped_states_disclosed_not_silent():
    """dmcs.n_eligible (states with a long-enough panel) must exceed dmcs.n_states (states also
    surviving the near-trigger floor) -- and the gap must be visible via the canonical fragment,
    not silently absorbed."""
    import json
    frag = json.loads((ROOT / "multiseries" / "results" / "canonical" / "dmcs.json").read_text())
    n_eligible = frag["dmcs.n_eligible"]["value"]
    n_states = frag["dmcs.n_states"]["value"]
    assert n_eligible > n_states, (
        "expected the near-trigger floor to drop at least one state relative to the eligible count"
    )
    assert n_states == len(_dmcs_df())


def test_decision_loss_no_longer_computed_on_the_full_sample():
    """Direct code-shape guard: dmcs.py's main() must call year_mcs for the decision loss with a
    restricted `near` index, not the same `common` index used for the statistical loss -- the
    exact bug this fix closes (both calls previously shared `common`)."""
    src = (ROOT / "code" / "pipeline" / "dmcs.py").read_text()
    assert "year_mcs(dec_loss, near)" in src
    assert "year_mcs(dec_loss, common)" not in src


@pytest.mark.skipif(not STATE_UR_PARQUET.exists(), reason="state_ur.parquet not available")
def test_near_trigger_construction_matches_for_one_real_state():
    """Fast, targeted re-derivation (one state, no nowcast refit needed beyond what's already
    cached-equivalent) of the near-trigger month count, using the identical construction dmcs.py's
    main() uses, and compares against the committed dmcs.csv row for that state."""
    import state_eb_decision as SEB
    import decision_protocol as DP

    d = _dmcs_df()
    df = SEB.load_state_ur()
    state = d.iloc[0]["state"]  # whichever state sorts first in the committed CSV -- deterministic
    u = df[state].dropna()
    nc = SEB.nowcast(u)
    # Deliberately the imported constant, not a hardcoded `+ 15`: a literal here would reintroduce
    # the duplicated-literal pattern dmcs.py:41 avoids, and would leave this test silently checking
    # a stale boundary if WARMUP_YEARS ever changes.
    common = u.index[u.index.year >= u.index.year.min() + SEB.WARMUP_YEARS]
    # dmcs.py also floors the scored window at the optional-TUR statute date, as scan() does --
    # mirrored here (again via the imported constant, not a literal) so this re-derivation keeps
    # matching the committed CSV's construction.
    common = common[common >= SEB.EB_TUR_EFFECTIVE]
    A = pd.DataFrame({m: DP.statutory_round_tur(DP.three_month_average(nc[m].reindex(u.index), u))
                      for m in SEB.MODELS}).reindex(common)
    near = common[((A >= 6.0) & (A <= 7.0)).any(axis=1)]

    committed = int(d[d.state == state].near_trigger_months.iloc[0])
    assert len(near) == committed


def test_dmcs_n_eligible_check_matches_canonical():
    """The re-derived dmcs.n_eligible must agree with the committed canonical.json entry.

    This test used to open with `if "dmcs.n_eligible" not in doc: pytest.skip("not yet merged")`,
    phrased for a key that was still in flight. The key IS in canonical.json, so that branch was
    unreachable -- and had it ever become reachable it would have silently skipped exactly the case
    worth failing on: a shipped key disappearing. It is a hard assertion now."""
    import canonical as C
    import json
    doc = json.loads(C.OUT.read_text())
    assert "dmcs.n_eligible" in doc, (
        "dmcs.n_eligible is no longer present in canonical.json -- it is a shipped key, so its "
        "absence is a regression, not a reason to skip"
    )
    assert C.CHECKS["dmcs.n_eligible"]() == doc["dmcs.n_eligible"]["value"]

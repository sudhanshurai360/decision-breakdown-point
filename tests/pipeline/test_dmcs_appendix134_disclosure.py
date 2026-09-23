"""Regression tests for the Appendix 13.4 disclosure trio dmcs.py's main() emits:
`dmcs.dropped_near_trigger_states`,
`dmcs.lowest_mean_ur_states`, `dmcs.dropped_lowest_ur_overlap`.

The three keys register, as real citable canonical values, facts Appendix 13.4 states about the near-
trigger screen's excluded states: the "excluded 7" (states with a long enough UR panel to attempt the
comparison but dropped by year_mcs's len(M)<30 floor once restricted to near-trigger months: IA, ID,
IN, ND, NE, SD, UT), the 5 lowest-mean-UR states over the full 51-jurisdiction panel (ND, NE, NH, SD, VT),
and their 3-state overlap (ND, NE, SD) -- shown to make the point that low-UR states are LIKELY, but not
GUARANTEED, to lack near-trigger months (4 of the excluded 7, IA, ID, IN and UT, are not among the 5
lowest-UR states at all).

(dmcs.wider_from_singleton_also_full_sample is covered separately in
test_dmcs_symmetric_robustness_cut.py, alongside its own sibling wider_from_singleton.)

Unlike dmcs.csv-derived keys, this trio is computed from the FULL 51-jurisdiction state UR panel
(`df` in dmcs.py's main(), via `state_eb_decision.load_state_ur()`), not from dmcs.csv's 44 retained
rows -- dropped_near_trigger_states names states that never made it into dmcs.csv at all. Independent
re-derivation from the committed state_ur.parquet and dmcs.csv's own state column, following the same
pattern test_dmcs_near_trigger.py already established for reading the real UR panel without paying
for a full nowcast/MCS rerun.
"""
import json
from pathlib import Path

import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DMCS_CSV = ROOT / "multiseries" / "results" / "dmcs.csv"
DMCS_CANON = ROOT / "multiseries" / "results" / "canonical" / "dmcs.json"
STATE_UR_PARQUET = ROOT / "multiseries" / "data" / "state_ur.parquet"

pytestmark = pytest.mark.skipif(
    not (DMCS_CSV.exists() and DMCS_CANON.exists() and STATE_UR_PARQUET.exists()),
    reason="dmcs.csv, dmcs.json, or state_ur.parquet not available",
)


def _dmcs_states():
    return set(pd.read_csv(DMCS_CSV).state)


def _full_ur_panel():
    """The same full 51-jurisdiction panel dmcs.py's main() loads as `df`, via the real cached
    parquet -- not synthetic data, and not filtered to the >=300-month n_eligible floor (dmcs.py's
    own dropped_states/lowest_ur_states are both computed against the UNFILTERED df.columns/df.mean()
    the same way)."""
    return pd.read_parquet(STATE_UR_PARQUET)


def _canon():
    return json.loads(DMCS_CANON.read_text())


def test_dropped_near_trigger_states_recomputes_correctly_from_committed_data():
    """dmcs.dropped_near_trigger_states: set(df.columns) - set(dmcs.csv's state column) -- states
    with a long enough UR panel to attempt the comparison but dropped by year_mcs's len(M)<30 floor
    once restricted to near-trigger months (or, for states never reaching the earlier >=300-month
    n_eligible floor at all, dropped there instead -- both floors collapse into this one set-difference,
    exactly as dmcs.py's main() computes it)."""
    df = _full_ur_panel()
    retained = _dmcs_states()
    dropped = sorted(set(df.columns) - retained)

    frag = _canon()
    assert frag["dmcs.dropped_near_trigger_states"]["value"] == len(dropped)
    assert frag["dmcs.dropped_near_trigger_states"]["of"] == df.shape[1]
    declared = frag["dmcs.dropped_near_trigger_states"]["states"].split(";")
    assert declared == dropped


def test_lowest_mean_ur_states_recomputes_correctly_from_committed_data():
    """dmcs.lowest_mean_ur_states: the 5 states with the lowest mean UR over the FULL, unfiltered
    51-jurisdiction panel -- df.mean().sort_values().index[:5], exactly as dmcs.py's main() computes
    it (sorted alphabetically for the declared `states=` field, as every other dmcs.* states list is)."""
    df = _full_ur_panel()
    lowest = sorted(df.mean().sort_values().index[:5].tolist())

    frag = _canon()
    assert frag["dmcs.lowest_mean_ur_states"]["value"] == 5
    assert frag["dmcs.lowest_mean_ur_states"]["of"] == 5
    declared = frag["dmcs.lowest_mean_ur_states"]["states"].split(";")
    assert declared == lowest


def test_dropped_lowest_ur_overlap_recomputes_correctly_from_committed_data():
    """dmcs.dropped_lowest_ur_overlap: states in BOTH dropped_near_trigger_states and
    lowest_mean_ur_states -- the '3 of 5' overlap Appendix 13.4 reports, re-derived from the same two
    independently-recomputed sets the two tests above already check individually (not from the
    already-emitted canonical values), so a coincidental agreement between this test and dmcs.py's own
    bug could not paper over a real drift in either summand."""
    df = _full_ur_panel()
    retained = _dmcs_states()
    dropped = set(df.columns) - retained
    lowest = set(df.mean().sort_values().index[:5].tolist())
    overlap = sorted(dropped & lowest)

    frag = _canon()
    assert frag["dmcs.dropped_lowest_ur_overlap"]["value"] == len(overlap)
    assert frag["dmcs.dropped_lowest_ur_overlap"]["of"] == 5
    declared = frag["dmcs.dropped_lowest_ur_overlap"]["states"].split(";")
    assert declared == overlap


def test_dropped_lowest_ur_overlap_is_a_genuine_partial_overlap():
    """Sanity bound making Appendix 13.4's actual point testable: the excluded set must be RELATED
    to, but not IDENTICAL to, the lowest-UR-5 set -- 0 overlap would mean the near-trigger screen has
    nothing to do with a state's UR level, and 5 (full overlap) would mean the screen is nothing MORE
    than a low-UR filter. On the committed data the overlap is exactly 3 of 5 (ND, NE, SD), with 4
    of the 7 excluded states (IA, ID, IN, UT) absent from the lowest-UR-5 set entirely."""
    frag = _canon()
    overlap = frag["dmcs.dropped_lowest_ur_overlap"]["value"]
    dropped_states = set(frag["dmcs.dropped_near_trigger_states"]["states"].split(";"))
    lowest_states = set(frag["dmcs.lowest_mean_ur_states"]["states"].split(";"))
    assert 0 < overlap < 5, (
        f"dropped_lowest_ur_overlap={overlap} -- expected a genuine partial overlap (on the "
        f"committed data, 3 of 5), not 0 (screen unrelated to UR level) or 5 (screen is just a "
        f"low-UR filter)"
    )
    assert dropped_states - lowest_states, (
        "expected at least one excluded state NOT among the 5 lowest-UR states (on the committed "
        "data, IA, ID, IN and UT) -- Appendix 13.4's point is that the screen is related to, not "
        "identical to, UR level"
    )

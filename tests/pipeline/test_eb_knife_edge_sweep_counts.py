"""Regression test for a gate blind spot: `canonical.py`'s `CHECKS` registry only re-derives a
canonical key's single scalar `value` field, never its companion fields -- there is no independent
re-derivation registered anywhere for companion fields like n/of/band. That gap can let a real
numeric error ship. A knife-edge float-precision change moves
`eb.knife_edge_sweep_genuine_states`'s `counts` companion array (the per-threshold sweep,
[24, 5, 2, 2, 2] vs [7, 4, 2, 2, 2]) while leaving the single `value` field (the 0.3pp entry, 2)
untouched -- so `canonical.py --check` reports PASS even if `paper/draft_ssrn.md`'s prose still
quotes the stale "4 states at 0.2pp, 7 at 0.1pp" figures. Adding a
general companion-field-checking mechanism to canonical.py's CHECKS architecture is a bigger,
riskier change than this specific gap warrants; this standalone test closes the gap narrowly instead
-- independently re-deriving both `eb.genuine_divergent_states` and the full
`eb.knife_edge_sweep_genuine_states.counts` array from the raw per-flip-month spreads, via a
DIFFERENT code path than `state_eb_decision.py::main()`'s own aggregation.
"""
import json
from pathlib import Path

import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
RES = ROOT / "multiseries" / "results"
CANON = ROOT / "paper" / "canonical.json"
SWEEP_CSV = RES / "state_eb_knife_edge_sweep.csv"
DECISION_CSV = RES / "state_eb_decision.csv"

pytestmark = pytest.mark.skipif(
    not (CANON.exists() and SWEEP_CSV.exists() and DECISION_CSV.exists()),
    reason="canonical.json / state_eb results not available",
)


def _canon():
    return json.loads(CANON.read_text())


def test_genuine_divergent_states_matches_a_fresh_csv_rederivation():
    """Independent re-derivation of eb.genuine_divergent_states straight from state_eb_decision.csv
    (flip_episodes>0 and flip_months>knife_edge), not via state_eb_decision.py's own r[...] slice."""
    canon = _canon()
    r = pd.read_csv(DECISION_CSV)
    genuine = r[(r["flip_episodes"] > 0) & (r["flip_months"] > r["knife_edge"])]
    assert canon["eb.genuine_divergent_states"]["value"] == len(genuine)
    assert sorted(canon["eb.genuine_divergent_states"]["states"]) == sorted(genuine["state"].tolist())


def test_knife_edge_sweep_counts_array_matches_a_fresh_csv_rederivation():
    """Closes the gap the single-value check leaves open: independently re-derive the FULL per-threshold sweep
    array (not just the single 0.3pp value canonical.py's own CHECKS registry re-derives) from
    state_eb_knife_edge_sweep.csv, and confirm it matches paper/canonical.json's stored `counts`."""
    canon = _canon()
    sweep = pd.read_csv(SWEEP_CSV).sort_values("threshold_pp")
    entry = canon["eb.knife_edge_sweep_genuine_states"]
    assert list(entry["thresholds_pp"]) == pytest.approx(sweep["threshold_pp"].tolist())
    assert list(entry["counts"]) == sweep["genuine_states"].tolist(), (
        f"canonical.json's eb.knife_edge_sweep_genuine_states.counts "
        f"({entry['counts']}) does not match a fresh re-derivation from "
        f"{SWEEP_CSV.name} ({sweep['genuine_states'].tolist()}) -- this is exactly the staleness "
        f"class that let a real prose mismatch almost ship, since canonical.py --check never "
        f"verifies this companion array"
    )
    # The 0.3pp entry must equal eb.genuine_divergent_states by construction (documented in the
    # emit()'s own spec= string) -- verify that constraint independently too.
    idx03 = sweep.index[sweep["threshold_pp"] == 0.3][0]
    assert sweep.loc[idx03, "genuine_states"] == canon["eb.genuine_divergent_states"]["value"]


def test_draft_prose_quotes_the_same_sweep_counts_as_canonical_json():
    """Guards a specific staleness failure mode: the sweep values embedded in
    paper/draft_ssrn.md's S7 prose (the "N states at 0.2pp, M at 0.1pp" sentence) must match
    canonical.json's counts, not an earlier run's numbers."""
    canon = _canon()
    counts = canon["eb.knife_edge_sweep_genuine_states"]["counts"]
    thresholds = canon["eb.knife_edge_sweep_genuine_states"]["thresholds_pp"]
    c02 = counts[thresholds.index(0.2)]
    c01 = counts[thresholds.index(0.1)]
    draft_name = "draft_ssrn.md"
    draft = (ROOT / "paper" / draft_name).read_text()
    assert f"**{c02}** states at 0.2pp" in draft, (
        f"{draft_name} does not quote the current 0.2pp sweep count ({c02}) in the expected "
        f"phrasing -- either the prose is stale, or its wording changed and this test's "
        f"expected phrase needs updating to match"
    )
    assert f"**{c01}**" in draft.split("states at 0.2pp", 1)[1][:200], (
        f"{draft_name} does not quote the current 0.1pp sweep count ({c01}) near the 0.2pp "
        f"figure -- either the prose is stale, or its wording changed"
    )

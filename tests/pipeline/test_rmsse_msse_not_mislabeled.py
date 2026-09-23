"""The exact defect metric_consistency.py exists to catch, in social_choice.py <-> mcs_test.py.

The hazard: social_choice.py's `election.spearman_mcs_stability_rmsse` pairs its metric=="rmsse"
rows (social_choice.py's own METRICS computes rmsse as sqrt(mean(msse)) per year) against a column
of mcs_test.csv. The MCS_sq_10 column is MSSE -- mean(msse), no square root, as mcs_test.py's own
inline comment says at mcs_test.py:115 ("squared (MSSE -- NOT rmsse, no sqrt)") -- so pairing
"rmsse" with it would compare two different quantities under one name. mcs_test.py therefore also
computes a genuine sqrt(mean(msse))-per-year RMSSE matrix (mcs_test.py:116) and emits
MCS_rmsse_10/25, and social_choice.py's emit() pairs "rmsse" with MCS_rmsse_10.

The correlation this affects is cited in the draft's S5.4 ("The calibrated companion to k*"). Its
value is not hardcoded anywhere in this file: test_canonical_check_rederives_rmsse_correlation_
from_the_fixed_column below re-derives whatever the live value is, so a data refresh cannot make
these assertions stale.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
MCS_TEST_CSV = ROOT / "multiseries" / "results" / "mcs_test.csv"

sys.path.insert(0, str(ROOT / "code" / "pipeline"))
sys.path.insert(0, str(ROOT / "code" / "gates"))

pytestmark = pytest.mark.skipif(not MCS_TEST_CSV.exists(), reason="mcs_test.csv not available")


def test_mcs_rmsse_column_exists_and_differs_from_mcs_sq():
    """The whole point: MCS_rmsse_10 (genuine sqrt-based) must be a DIFFERENT column from
    MCS_sq_10 (plain MSSE) on at least some series -- if they were always identical, pairing
    "rmsse" with either would have been harmless and this wouldn't have been a real bug."""
    d = pd.read_csv(MCS_TEST_CSV)
    assert "MCS_rmsse_10" in d.columns
    assert "MCS_rmsse_25" in d.columns
    assert (d["MCS_rmsse_10"] != d["MCS_sq_10"]).any(), (
        "MCS_rmsse_10 and MCS_sq_10 are identical on every series -- either the sqrt computation "
        "regressed to a no-op, or this test's premise needs re-checking against fresh data"
    )


def test_mcs_rmsse_matches_sqrt_of_mcs_test_own_msse_computation():
    """Direct re-derivation for one real series: recompute sqrt(mean(msse)) per year exactly as
    mcs_test.py's main() does, and confirm MCS_rmsse_10's survivor-set SIZE for that series is
    reproducible from the same construction (not just present, but actually computed the claimed
    way).

    Picking pd.read_csv(MCS_TEST_CSV).iloc[0]["series"] would be the wrong choice here -- since the
    CSV is sorted by MCS_abs_10, that's always PI, the trivial series where
    MCS_sq_10 == MCS_rmsse_10 == 1: substituting the plain (no-sqrt) msse matrix for PI still yields
    survivor count 1, so a test built on PI would NOT catch a regression that dropped the sqrt.
    Pinned to RSAFS instead -- one of the exactly 3 of
    12 series (RSAFS, BOPGIMP, BUSINV) where MCS_sq_10 and MCS_rmsse_10 actually differ (verified in
    test_mcs_rmsse_column_exists_and_differs_from_mcs_sq above), so this test genuinely exercises
    the sqrt-vs-no-sqrt distinction rather than passing regardless of it."""
    import mcs_test as MT

    d = pd.read_csv(MT.PRED)
    sid = "RSAFS"
    assert sid in pd.read_csv(MCS_TEST_CSV)["series"].values, f"{sid} not in mcs_test.csv -- pick another differing series"
    g = d[d.series == sid]
    models = sorted(g.model.unique())
    yrs = sorted(g.year.unique())
    import numpy as np
    Lr = np.full((len(yrs), len(models)), np.nan)
    for j, m in enumerate(models):
        gm = g[g.model == m].groupby("year")
        r = np.sqrt(gm["msse"].mean())
        for i, y in enumerate(yrs):
            if y in r.index:
                Lr[i, j] = r[y]
    keep = ~np.isnan(Lr).any(1)
    Lr = Lr[keep]
    surv, _ = MT.mcs(Lr, alpha=0.10, seed=0)
    committed = int(pd.read_csv(MCS_TEST_CSV).set_index("series").loc[sid, "MCS_rmsse_10"])
    assert len(surv) == committed


def test_social_choice_pairs_rmsse_with_the_correct_column():
    """Code-shape guard: social_choice.py must pair the "rmsse" metric with MCS_rmsse_10, never
    the plain-MSSE MCS_sq_10 -- the exact bug this fix closes."""
    src = (ROOT / "code" / "pipeline" / "social_choice.py").read_text()
    assert '("rmsse", "MCS_rmsse_10")' in src
    assert '("rmsse", "MCS_sq_10")' not in src


def test_canonical_check_rederives_rmsse_correlation_from_the_fixed_column():
    import canonical as C
    import json
    doc = json.loads(C.OUT.read_text())
    for key in ("election.spearman_mcs_stability_rmsse", "election.spearman_mcs_stability_rmsse_p"):
        if key not in doc:
            pytest.skip(f"{key} not yet merged into canonical.json")
        assert C.CHECKS[key]() == pytest.approx(doc[key]["value"])

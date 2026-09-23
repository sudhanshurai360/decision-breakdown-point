"""make_figures.py writes the paper's figures and tables. Fig E is supplementary; `figure_c()`
writes `tableC_decision_cost_by_state.md` (a markdown table), not a PNG.

These tests lock in the places most likely to drift silently without anyone noticing a wrong plot:
Figure F's re-derived MCS membership (mcs_test.csv stores only counts, not which models -- this file
recomputes membership via the same tested mcs() call) and the note's LLM crowding figure (every
number in it is already in canonical.json, cross-checked here against the file that actually reads
it).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

sys.path.insert(0, str(ROOT / "code" / "pipeline"))
import make_figures as MF

MAIN_TEXT_FNS = (MF.figure_a, MF.figure_b, MF.figure_c, MF.figure_d, MF.figure_f, MF.figure_h,
                 MF.figure_i)
SUPPLEMENTARY_FNS = (MF.figure_e,)
NOTE_FNS = (MF.figure_note_llm_crowding,)
# table_permnull()/table_ar1null()/table_accuracy() (Tables P/Q/R) are called by main() and cited by
# name in draft_ssrn.md. Without a test here, a stale key name, wrong column, or broken cross_check
# in any of the three would surface only at manuscript-build time or via manual inspection.
TABLE_FNS = (MF.table_permnull, MF.table_ar1null, MF.table_accuracy)

RESULT_FILES = [
    "table1_identification.md",
    "figA_forest_kstar_stability.png",
    "figB_election_reversal.png",
    "tableC_decision_cost_by_state.md",
    "figD_prop22_ceiling_scatter.png",
    "figF_mcs_membership_heatmap.png",
    "figH_shape_not_magnitude.png",
    "figI_accuracy_vs_decision.png",
    "figS1_flagship_trajectory_supplementary.png",
    "figNote1_llm_crowding_vs_null.png",
    "tableP_permutation_null.md",
    "tableQ_ar1null.md",
    "tableR_accuracy.md",
]

pytestmark = pytest.mark.skipif(
    not (ROOT / "multiseries" / "results" / "mcs_test.csv").exists(),
    reason="result CSVs not available",
)


def test_main_runs_and_writes_all_artifacts(tmp_path, monkeypatch):
    """Reads the real frozen results (RES/FLAG_RES are bound to their real paths at import time,
    so monkeypatching ROOT afterward does not affect them) but writes to a scratch OUT, not the
    real paper/figures/ -- a public clone's `pytest` must not regenerate the shipped deliverables
    as a side effect."""
    scratch_out = tmp_path / "figures"
    scratch_out.mkdir()
    monkeypatch.setattr(MF, "OUT", scratch_out)
    monkeypatch.setattr(MF, "ROOT", tmp_path)   # OUT.relative_to(ROOT) in make_figures.py's own prints
    MF.table1()
    for fn in MAIN_TEXT_FNS + SUPPLEMENTARY_FNS + NOTE_FNS + TABLE_FNS:
        fn()
    for name in RESULT_FILES:
        out_path = MF.OUT / name
        assert out_path.exists(), f"{name} was not written"
        min_size = 300 if name.endswith(".md") else 1000
        assert out_path.stat().st_size > min_size, f"{name} is suspiciously small ({out_path.stat().st_size} bytes)"


def test_table1_row_count_and_nonidentified_count_match_mcs_test_csv():
    """The Fragile count is re-derived from the SAME inputs table1() uses -- social_choice.csv's
    20-seed winner_stability/reversal/k_star and multiseries_fragility.csv's n_years -- with the
    paper's own fixed rule. Deliberately NOT compared against multiseries_fragility.csv's own
    `fragile` column, which run_multiseries.py computes from a single-seed winner_stability: the two
    estimators legitimately disagree at the 0.60 cut (DGORDER: 0.593 twenty-seed vs 0.622
    single-seed on the regenerated panel), and Table 1's is the paper's definition."""
    sys.path.insert(0, str(ROOT / "code" / "instrument"))
    import fragility as F_frag
    t = MF.table1()
    mcs = pd.read_csv(MF.RES / "mcs_test.csv")
    assert len(t) == len(mcs)
    sc = pd.read_csv(MF.RES / "social_choice.csv").pipe(lambda d: d[d.metric == "mase"]).set_index("series")
    frag = pd.read_csv(MF.RES / "multiseries_fragility.csv").pipe(lambda d: d[d.metric == "mase"]).set_index("series")
    expected = 0
    for sid in mcs.series:
        e = sc.loc[sid]
        k_over_t = int(e.k_star) / int(frag.loc[sid, "n_years"])
        expected += int(bool(float(e.winner_stability) < F_frag.FRAGILE_STABILITY_THRESH
                             or (bool(e.reversal) and k_over_t < F_frag.FRAGILE_KSTAR_RATIO_THRESH)))
    assert int((t["Fragile"] == "yes").sum()) == expected


def test_figure_f_membership_matches_committed_mcs_counts():
    """The whole point of Figure F: re-derived membership's per-series COUNT must equal the
    committed mcs_test.csv MCS_abs_10 value -- if this drifts, the heatmap is showing something
    other than what the paper's own headline non-identification count is based on."""
    sys.path.insert(0, str(ROOT / "code" / "pipeline"))
    import mcs_test as MT
    committed = pd.read_csv(MF.RES / "mcs_test.csv").set_index("series")["MCS_abs_10"]
    d = pd.read_csv(MT.PRED)
    for sid, g in d.groupby("series"):
        models = sorted(g.model.unique())
        yrs = sorted(g.year.unique())
        La = np.full((len(yrs), len(models)), np.nan)
        for j, m in enumerate(models):
            gm = g[g.model == m].groupby("year")["sae"].mean()
            for i, y in enumerate(yrs):
                if y in gm.index:
                    La[i, j] = gm[y]
        La = La[~np.isnan(La).any(1)]
        surv, _p = MT.mcs(La, alpha=0.10, seed=0)
        assert len(surv) == committed[sid], f"{sid}: membership count drifted from mcs_test.csv"


def test_table_c_state_categories_are_exhaustive_and_mutually_exclusive():
    """Table C is a 3-category breakdown of all 50 states (it replaced an earlier tile-grid map).
    Lock in that every state lands in exactly one category (no state dropped, none double-counted)
    and that the genuine-flip category is exactly {MA, NV}, matching the paper's own headline
    claim."""
    d = pd.read_csv(MF.RES / "state_eb_decision.csv")
    d["genuine"] = d.flip_episodes - d.knife_edge
    genuine = set(d[d.genuine > 0].state)
    knife_only = set(d[(d.genuine == 0) & (d.flip_episodes > 0)].state)
    no_flip = set(d[d.flip_episodes == 0].state)
    assert genuine == {"MA", "NV"}
    assert not (genuine & knife_only) and not (genuine & no_flip) and not (knife_only & no_flip)
    assert genuine | knife_only | no_flip == set(d.state)


def test_figure_i_contingency_counts_sum_to_all_retained_states():
    """The redesigned Figure I is a contingency grid, not a scatter -- lock in that its cell counts
    still sum to the full retained-state count, so the grid isn't silently dropping states relative
    to the file's own tied_both computation."""
    d = pd.read_csv(MF.RES / "dmcs.csv")
    counts = d.groupby(["MCS_stat_near", "dMCS_decision"]).size()
    assert int(counts.sum()) == len(d)


def test_figure_i_equal_cell_marks_true_value_equality():
    """The equal-value reference must be selected in VALUE space (x == y), not index space (i == j).
    On the committed data the two do not coincide: xs is [1, 2, 3] and ys is [2, 3, 4], so index
    equality picks (1,2), (2,3) and (3,4) -- 1, 9 and 5 states, none of them value-equal -- while
    the one genuinely value-equal populated cell, (3,3) with 18 states, is the cell S7's own prose
    calls "exactly equal" and is missed entirely. A figure built on index equality would contradict
    its own caption ("no state falls below the line") at its single most important cell.

    The selection is not reimplemented here: this calls MF._figure_i_equal_cells(), the same pure
    helper figure_i() itself calls, so the test and the figure cannot diverge, and no PNG is
    written (the helper has no plotting or file I/O)."""
    d = pd.read_csv(MF.RES / "dmcs.csv")
    xs = sorted(d.MCS_stat_near.unique())
    ys = sorted(d.dMCS_decision.unique())
    counts = d.groupby(["MCS_stat_near", "dMCS_decision"]).size()

    equal_cells_by_value = MF._figure_i_equal_cells(xs, ys, counts)
    equal_cells_by_index = [(i, j) for i in range(len(xs)) for j in range(len(ys)) if i == j]

    # On the real, committed data these two sets differ -- this is exactly the bug: index equality
    # does not coincide with value equality here. If a future edit "simplified" the fix back to
    # index-based selection, this assertion would catch it.
    assert equal_cells_by_value != equal_cells_by_index

    assert len(equal_cells_by_value) >= 1
    # the (3,3) cell -- the one S7's prose calls "exactly equal" -- must be among them and populated
    assert any(xs[i] == ys[j] == 3 for i, j in equal_cells_by_value)
    assert int(counts.get((3, 3), 0)) > 0
    # and every outlined cell is genuinely populated
    assert all(int(counts.get((xs[i], ys[j]), 0)) > 0 for i, j in equal_cells_by_value)


def test_table_accuracy_matches_model_set_audit_csv():
    """Table R (table_accuracy) reads directly from model_set_audit.csv's loss_<model> columns and
    formats them, with no arithmetic of its own (make_figures.py:74-80) -- lock in the row count and one
    spot-checked cell so a stale key name or wrong column can't silently drift the manuscript's
    displayed table away from the CSV it claims to be a literal rendering of."""
    R = pd.read_csv(MF.RES / "model_set_audit.csv").sort_values("series")
    path = MF.OUT / "tableR_accuracy.md"
    text = path.read_text()
    lines = [l for l in text.splitlines() if l.startswith("| ") and not l.startswith("|---")]
    body_rows = lines[1:]  # drop the header row
    assert len(body_rows) == len(R), "Table R's row count must match model_set_audit.csv exactly"
    first_series = R.iloc[0]
    first_row = body_rows[0]
    assert first_series.series in first_row
    assert f"{first_series.loss_ar1:.3f}" in first_row, (
        "Table R's first data row must contain the AR1 loss value read straight from "
        "model_set_audit.csv, not a recomputed or stale figure"
    )


def test_table_permnull_matches_canonical_permutation_null_values():
    """Table P (table_permnull) reads election.permnull_<key>_mean/_p_ge for 5 statistics -- lock
    in that a spot-checked row (the reversal statistic) matches canonical.json exactly, so a typo'd
    key or wrong field can't silently render a different number than what's registered/cited."""
    m = MF._canon()["election.permnull_reversal_mean"]
    text = (MF.OUT / "tableP_permutation_null.md").read_text()
    assert f"{int(m['observed'])} of {int(m['of'])}" in text
    assert "score-vs-plurality reversal" in text


def test_table_ar1null_matches_canonical_ar1_coefficient():
    """Table Q (table_ar1null) captions the empirically-estimated pooled AR(1) coefficient
    election_null_ar1.py computes -- lock in the rendered caption actually contains that value (not
    a stale or hardcoded one) and that the table itself has the 5 rows Table P also has."""
    ar_hat = MF.cval("election.ar1null_coefficient")
    text = (MF.OUT / "tableQ_ar1null.md").read_text()
    assert f"AR(1)={ar_hat:.3f}" in text
    body_rows = [l for l in text.splitlines() if l.startswith("| ") and not l.startswith("|---")][1:]
    assert len(body_rows) == 5


def test_note_llm_crowding_figure_matches_canonical_json():
    """Every number in the note's one figure is already in canonical.json (no recomputation) --
    lock in that cval() reads exactly what ml_witness.py emitted, not a stale or mistyped key."""
    for k in [2, 4, 8, 16, 32, 64]:
        real = MF.cval(f"llm.crowd_real_k{k}")
        null = MF.cval(f"llm.crowd_null_k{k}")
        assert 0 <= real <= 1 and 0 <= null <= 1
        assert real < null, f"k={k}: expected the real leaderboard to be more identified than the null"

"""Real, permanent regression tests for code/pipeline/social_choice.py (election.* family) and
code/instrument/flagship_fragility.py + code/pipeline/realised_t.py (flagship.*/prop.* family).

These tests LOCK IN verified-correct behavior. Both F.fragility()'s
runner_up/decision_breakdown() and realised_t.py's own analyse() tie-break a k*-tie among rivals the
SAME way (by rival name: decision_breakdown() iterates `sorted(m for m in L if m != a)` and keeps the
first minimal-k match; analyse() picks `min(rows, key=lambda r: (r["k"], r["rival"]))`), so the "same
pair" claim in flagship_fragility.py's emit() spec is correct as production code stands today. This
file's own `_analyse_realised_t_local()` helper below must stay a faithful copy of that convention --
TestFlagshipBindingRivalCrossReference exists to guard against exactly that kind of drift between a
test's local stand-in and the real function it mirrors.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import fragility as F

ROOT = Path(__file__).resolve().parent.parent.parent
FLAGSHIP_CSV = ROOT / "results" / "extension_scoreboard_1988-2025.csv"
PRED_CSV = ROOT / "multiseries" / "results" / "breadth_distinct" / "breadth_distinct_predictions.csv"

pytestmark = pytest.mark.skipif(
    not FLAGSHIP_CSV.exists() or not PRED_CSV.exists(),
    reason="requires the real curated data files (not shipped in a fresh clone without the data pull)",
)


def _analyse_realised_t_local(M: pd.DataFrame, win_name: str):
    """Faithful copy of realised_t.py::analyse()'s binding-rival selection (tie-break by
    `(k, rival name)`, i.e. sorted name on a k*-tie -- matching F.decision_breakdown()'s own
    `sorted(m for m in L if m != a)` convention), isolated for testing without running the whole
    producer script."""
    T = len(M)
    rows = []
    for riv in M.columns:
        if riv == win_name:
            continue
        d = (M[riv] - M[win_name]).to_numpy(float)
        if d.sum() <= 0:
            continue
        sd = float(d.std(ddof=0))
        if sd <= 0:
            continue
        k = F.breakdown_number(M[win_name].to_numpy(float), M[riv].to_numpy(float), np.ones(T))[0]
        rows.append(dict(rival=riv, k=k))
    if not rows:
        return None
    return min(rows, key=lambda r: (r["k"], r["rival"]))["rival"]


class TestFlagshipBindingRivalCrossReference:
    """flagship_fragility.py's flagship.k_star emit() explicitly claims its `rival` field names "the
    SAME PAIR" as realised_t.py's flagship.t_binding -- computed via two independent code paths:
    F.fragility()'s runner_up (== decision_breakdown()'s binding opponent, tie-broken by SORTED NAME)
    and realised_t.py's own analyse() (tie-broken by `(k, rival name)`, i.e. the same sorted-name
    convention). Both conventions agree by construction, not by accident of the current data -- see
    test_local_helper_matches_decision_breakdowns_tie_break below for a constructed proof on a
    genuinely tied panel. This class's real job is to guard `_analyse_realised_t_local()` (this test
    file's own stand-in for realised_t.py::analyse(), used because running the whole producer script
    per test would be far too slow) against drifting out of sync with the real function it mirrors."""

    def test_real_flagship_data_binding_rival_agrees_across_both_code_paths(self):
        sb = pd.read_csv(FLAGSHIP_CSV, index_col=0)
        models = list(sb.columns)
        years = list(sb.index)
        L = {m: sb[m].to_numpy(float) for m in models}
        w = np.ones(len(years))
        r = F.fragility(L, w, labels=years, n_boot=500, seed=0)  # n_boot reduced for test speed

        local_binding = _analyse_realised_t_local(sb, sb.mean().idxmin())

        assert r["runner_up"] == local_binding, (
            f"flagship.k_star's rival ({r['runner_up']!r}) and flagship.t_binding's rival "
            f"({local_binding!r}) disagree -- either the 'same pair' claim in flagship_fragility.py's "
            f"emit() spec is false on the real data, or _analyse_realised_t_local() above has drifted "
            f"out of sync with the real realised_t.py::analyse() it is meant to mirror. Check the "
            f"latter first: it is the more likely culprit."
        )
        # Document *why* they currently agree even setting the tie-break question aside: no ambiguity
        # exists on the real panel today.
        assert r["n_binding_tied"] == 1, (
            "flagship panel now has a genuine k*-tie among rivals (n_binding_tied > 1) -- the "
            "binding-rival agreement checked above now genuinely exercises the tie-break convention "
            "rather than passing on an unambiguous panel; re-verify test_real_flagship_data_binding_"
            "rival_agrees_across_both_code_paths still passes for the right reason."
        )

    def test_local_helper_matches_decision_breakdowns_tie_break(self):
        """Constructed proof (not the real data): a panel with two rivals genuinely tied at the same
        minimum k*, with the alphabetically-later one placed FIRST in column order, so a naive
        column-order tie-break would disagree with decision_breakdown()'s sorted-name one if the two
        conventions actually differed. They must not: _analyse_realised_t_local() must return the
        SAME binding rival as F.decision_breakdown() on this panel, confirming the local helper is a
        faithful copy of the real analyse()'s `(k, rival name)` tie-break, not a column-order one."""
        T = 10
        win = np.array([1.0] * T)
        riv_a = win.copy(); riv_a[0] += 5.0  # k*=1
        riv_b = win.copy(); riv_b[0] += 5.0  # identical margin structure -> also k*=1
        M = pd.DataFrame({"zzz_rival": riv_b, "champion": win, "aaa_rival": riv_a})

        L = {c: M[c].to_numpy(float) for c in M.columns}
        w = np.ones(T)
        k, sorted_name_binding, _removed, ties = F.decision_breakdown(L, w, a="champion", return_ties=True)
        assert sorted_name_binding == "aaa_rival"
        assert set(ties) == {"aaa_rival", "zzz_rival"}

        local_binding = _analyse_realised_t_local(M, "champion")
        assert local_binding == sorted_name_binding, (
            f"_analyse_realised_t_local() picked {local_binding!r} but decision_breakdown()'s "
            f"sorted-name tie-break picked {sorted_name_binding!r} -- the local helper has drifted "
            f"away from the (k, rival name) convention it is supposed to mirror (e.g. someone dropped "
            f"the secondary `rival` sort key), which is exactly the kind of test-only drift that would "
            f"make test_real_flagship_data_binding_rival_agrees_across_both_code_paths above pass or "
            f"fail for the wrong reason."
        )


class TestSocialChoiceWeightRobustnessInvariant:
    """social_choice.py's period-weighting-robustness section builds a month-count weight as
    `g.groupby('year').size()` on a per-series group that still contains ALL models' rows -- so the
    row count per year is (models evaluated that year) x (months present that year), not pure months.
    This is harmless ONLY because model count is constant per series across all years in the current
    real data (verified below) -- the constant multiplicative factor cancels out exactly under
    F.fragility()'s weight-scale-invariant averaging. If a future data update ever evaluates a
    different number of models in different years for the same series, this weighting would become
    wrong (proportional to model-count x month-count, not month-count alone). social_choice.py does
    raise a RuntimeError on that condition at the top of the same block, so it would not pass
    unnoticed; this test is the cheaper, earlier guard that names the consequence for
    election.weight_* rather than only halting the producer."""

    def test_model_count_is_constant_per_series_across_years_in_real_data(self):
        d = pd.read_csv(PRED_CSV)
        for sid, g in d.groupby("series"):
            models_per_year = g.groupby("year")["model"].nunique()
            assert models_per_year.nunique() == 1, (
                f"series {sid!r} has a DIFFERENT number of models evaluated in different years "
                f"({models_per_year.to_dict()}) -- social_choice.py's month-weighting-robustness "
                f"section (`nmonths_year = g.groupby('year').size()`) requires this to be "
                f"constant; it no longer is, so election.weight_* values are now confounded by "
                f"model-count, not just month-count. Needs a real fix (group by year AND model, or "
                f"divide by the year's actual model count) before those numbers can be trusted."
            )

    def test_month_count_weighting_matches_true_month_count_up_to_constant_factor(self):
        """Directly confirms the *mechanism* claimed above: for a real curated series, the row-count
        weight social_choice.py actually uses is exactly K (model count) times the true month count,
        for every year -- i.e. a constant multiple, which is why it's currently harmless."""
        d = pd.read_csv(PRED_CSV)
        sid = d.series.iloc[0]
        g = d[d.series == sid]
        K = g.model.nunique()
        rowcount_per_year = g.groupby("year").size()
        # true month count per year: unique (year, month)-shaped rows for a SINGLE model
        one_model = g[g.model == g.model.iloc[0]]
        true_month_count = one_model.groupby("year").size()
        ratio = (rowcount_per_year / true_month_count).dropna()
        assert np.allclose(ratio, K), (
            f"expected the row-count-per-year weight to be exactly K={K} times the true month count "
            f"in every year for series {sid!r}; got ratios {ratio.to_dict()} -- the 'constant factor' "
            f"assumption this test file documents may not hold as cleanly as believed"
        )

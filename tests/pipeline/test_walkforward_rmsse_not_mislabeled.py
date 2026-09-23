"""walkforward.py's `agg_metric(detail, "rmsse")` must apply the square root to the per-year
mean(msse). Returning the plain per-year mean(msse) -- no square root anywhere -- and feeding it
into fragility()'s pooled_winner/k_star/winner_stability machinery via score_multiseries.py and
run_multiseries.py would mislabel MSSE as "rmsse" throughout. The argument "sqrt is rank-preserving
so the pooled MSSE-winner == RMSSE-winner" is true only for a SINGLE aggregation stage;
agg_metric's actual pipeline aggregates in two stages (within-year mean, then a separate weighted
mean across years in fragility()), so mean(sqrt(x)) != sqrt(mean(x)) applies -- the same principle
as mcs_test.py/social_choice.py's RMSSE column. Empirically: isolating only the aggregator on
identical input data, pooled_winner disagreed on 7 of 13 series and k* on 5 of 13 between the
plain-mean convention and social_choice.py's genuine sqrt(mean(msse))-per-year one (on the
13-series panel of the time).

Correspondingly, score_multiseries.py must NOT apply its own `pooled ** 0.5` after agg_metric: L2
is already on the RMSSE scale, so a second sqrt would double-square-root it. THAT half is covered
behaviourally, not here: see
tests/pipeline/test_score_multiseries_core.py::test_rescore_series_rmsse_sqrt_applied_per_year_not_to_pooled_mean,
which hand-derives winner_pooled_loss == 2.0 and so fails on any second sqrt (2.0 -> 1.414) no
matter how it is spelled. This file used to carry a `assert "pooled ** 0.5" not in src` guard
alongside it; that guard forbade one SPELLING only -- appending a genuine
`pooled = float(np.sqrt(pooled))` to score_multiseries.py left it green -- so it was removed rather
than left in place looking like coverage it never provided.

SCOPE NOTE. This
pipeline (walkforward.py / score_multiseries.py / run_multiseries.py) still has no emit() calls and is
not in reproduce.py's PIPELINE, and that much was right. But `run_multiseries.py --combine`'s
multiseries_predictions.csv is BYTE-IDENTICAL (verified by direct SHA-256 comparison) to
breadth_distinct_predictions.csv under a different name/path -- and THAT file IS read directly,
repeatedly, by code/gates/canonical.py (feeding cited canonical values including data.curated_models,
data.curated_n_models, data.curated_series, data.curated_year_max) and by
code/gates/metric_consistency.py. So this fix (and any other change to this pipeline) DOES have a live
downstream consequence on the published manuscript, just reached one hop removed rather than via
emit()/reproduce.py directly.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "code" / "pipeline"))

import walkforward as WF  # noqa: E402


def _synthetic_detail():
    return pd.DataFrame({
        "series": ["A"] * 6,
        "model": ["m1"] * 3 + ["m2"] * 3,
        "year": [2020, 2020, 2021] * 2,
        "sae": [1, 2, 3, 4, 5, 6],
        "msse": [1, 4, 9, 16, 25, 36],
    })


def test_rmsse_loss_is_sqrt_of_per_year_mean_msse():
    agg = WF.agg_metric(_synthetic_detail(), "rmsse")
    # hand-computed: m1/2020 msse=[1,4] -> mean=2.5 -> sqrt=1.5811388...
    got = agg[(agg.model == "m1") & (agg.year == 2020)]["loss"].iloc[0]
    assert got == pytest.approx(np.sqrt(2.5))
    # m1/2021 msse=[9] -> mean=9 -> sqrt=3.0
    got2 = agg[(agg.model == "m1") & (agg.year == 2021)]["loss"].iloc[0]
    assert got2 == pytest.approx(3.0)


def test_mase_loss_unaffected_by_the_rmsse_fix():
    agg = WF.agg_metric(_synthetic_detail(), "mase")
    got = agg[(agg.model == "m1") & (agg.year == 2020)]["loss"].iloc[0]
    assert got == pytest.approx(1.5)   # mean(sae)=[1,2] -> 1.5, no sqrt for mase


def test_rmsse_loss_is_not_the_plain_unrooted_mean():
    """Directly guards against the exact regression: sqrt must actually be applied, not just
    present in a comment. If agg_metric silently stopped applying sqrt, this fails."""
    agg = WF.agg_metric(_synthetic_detail(), "rmsse")
    got = agg[(agg.model == "m2") & (agg.year == 2020)]["loss"].iloc[0]
    plain_mean = (16 + 25) / 2
    assert got == pytest.approx(np.sqrt(plain_mean))
    assert got != pytest.approx(plain_mean)  # would fail if sqrt regressed to a no-op


# NOTE: the downstream "score_multiseries.py must not double-square-root" guard that used to live
# here was a string check (`assert "pooled ** 0.5" not in src`) and is deliberately gone -- see this
# module's docstring. The behaviour is covered in
# tests/pipeline/test_score_multiseries_core.py::test_rescore_series_rmsse_sqrt_applied_per_year_not_to_pooled_mean.

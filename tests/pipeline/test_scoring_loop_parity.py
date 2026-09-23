"""run_multiseries.py::run_series() and score_multiseries.py::rescore_series() each hand-copy the
identical inner scoring loop (complete-years filter -> WF.agg_metric()/WF.to_fragility_inputs() ->
F.fragility(..., shock_periods=SHOCK_YEARS, n_boot=N_BOOT, seed=0) -> a result-row dict sharing 13
fields verbatim). This IS a value-computation path that
feeds real results CSVs, so the chosen response was to add a regression test locking the two
functions' agreement on the shared fields BEFORE attempting any merge. This file is that test. A
full production-code merge was deliberately NOT attempted: a locking regression test, not a forced
merge, is this codebase's standard response when correctness stakes are non-trivial and the current
duplication has already been manually re-verified accurate multiple times with no drift found.

Strategy: feed both functions byte-identical underlying (y_true, y_pred, scale) data via
monkeypatching their respective upstream data-fetch calls (WF.run() for run_series(),
WF.load_vintages()/WF.realized()/WF.year_scales() for rescore_series()), so both reach
WF.agg_metric() with an equivalent DataFrame -- then assert the two functions' output rows agree
exactly on all 13 documented shared fields, and that each function's own extra fields are exactly
the ones the registry documents (year_min/year_max/concentration for run_series();
scale/winner_pooled_loss for rescore_series()) -- not a numeric re-test of F.fragility() itself
(already covered elsewhere), but a structural + numeric parity check of the two call sites.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "code" / "pipeline"))
sys.path.insert(0, str(ROOT / "code" / "instrument"))

import numpy as np
import pandas as pd
import pytest

import run_multiseries as RM
import score_multiseries as SM

SHARED_FIELDS = [
    "series", "metric", "n_years", "pooled_winner", "per_period_winner", "runner_up",
    "k_star", "k_over_T", "reversal", "responsible", "shock_coincidence",
    "winner_stability", "fragile",
]
RUN_SERIES_EXTRA = {"year_min", "year_max", "concentration"}
RESCORE_SERIES_EXTRA = {"scale", "winner_pooled_loss"}


def _synthetic_detail():
    """A detail-shaped DataFrame matching WF.run()'s own documented output schema: [series, model,
    target, year, y_true, y_pred, scale_abs, scale_sq, sae, msse]. 3 models with clearly separated
    biases (so pooled_winner/k_star are non-degenerate), 5 non-shock years, scale held at 1.0 so
    sae/msse reduce to plain absolute/squared error -- easy to hand-verify and, more importantly,
    trivially reproducible by rescore_series()'s own from-scratch sae/msse computation once fed the
    same scale."""
    years = [2015, 2016, 2017, 2018, 2019]
    bias = {"m0": 1.0, "m1": 3.0, "m2": 8.0}
    rng = np.random.default_rng(42)
    rows = []
    for yr in years:
        y_true = 100.0
        for mdl, b in bias.items():
            y_pred = y_true + b + rng.normal(0, 0.3)
            rows.append(dict(series="S1", model=mdl, target=pd.Timestamp(f"{yr}-06-01"), year=yr,
                              y_true=y_true, y_pred=y_pred, scale_abs=1.0, scale_sq=1.0))
    detail = pd.DataFrame(rows)
    detail["sae"] = (detail.y_true - detail.y_pred).abs() / detail.scale_abs
    detail["msse"] = (detail.y_true - detail.y_pred) ** 2 / detail.scale_sq
    return detail


def _run_both(monkeypatch, detail):
    # N_BOOT is reduced for speed only, and identically on both sides; seed=0 is fixed in both call
    # sites, so the parity assertions below are unaffected by the reduction.
    monkeypatch.setattr(RM, "N_BOOT", 50)
    monkeypatch.setattr(SM, "N_BOOT", 50)
    monkeypatch.setattr(RM.WF, "run", lambda sid, models, eval_start=RM.EVAL_START,
                         eval_end=RM.EVAL_END, m=RM.M: detail.copy())

    scale_by_year = detail.drop_duplicates("year").set_index("year")["scale_abs"].to_dict()

    def fake_year_scales(truth, years, m=12, squared=False, trail=None):
        return dict(scale_by_year)

    monkeypatch.setattr(SM.WF, "load_vintages", lambda sid: None)
    monkeypatch.setattr(SM.WF, "realized", lambda v, sid="", guard_before=None: None)   # signature mirrors WF.realized, including guard_before
    monkeypatch.setattr(SM.WF, "year_scales", fake_year_scales)

    run_rows, run_detail = RM.run_series("S1")
    pred = detail[["series", "model", "target", "year", "y_true", "y_pred"]].copy()
    rescore_rows = SM.rescore_series("S1", pred)
    return run_rows, rescore_rows


def test_shared_fields_match_exactly_on_identical_underlying_data(monkeypatch):
    detail = _synthetic_detail()
    run_rows, rescore_rows = _run_both(monkeypatch, detail)

    run_by_metric = {r["metric"]: r for r in run_rows}
    rescore_by_metric = {r["metric"]: r for r in rescore_rows}
    assert set(run_by_metric) == {"mase", "rmsse"}
    assert set(rescore_by_metric) == {"mase", "rmsse"}

    for metric in ("mase", "rmsse"):
        a, b = run_by_metric[metric], rescore_by_metric[metric]
        for field in SHARED_FIELDS:
            assert field in a, f"run_series() missing documented shared field {field!r}"
            assert field in b, f"rescore_series() missing documented shared field {field!r}"
            av, bv = a[field], b[field]
            if isinstance(av, float):
                assert av == pytest.approx(bv), f"{metric}.{field}: run={av!r} rescore={bv!r}"
            else:
                assert av == bv, f"{metric}.{field}: run={av!r} rescore={bv!r}"


def test_extra_fields_match_the_registrys_documented_split(monkeypatch):
    """Locks the field-count claim this file's own SHARED_FIELDS/RUN_SERIES_EXTRA/RESCORE_SERIES_EXTRA
    constants make (16 total fields in run_series(), 15 in rescore_series(), 13 shared) -- if a
    future edit silently adds/drops a field on either side, this fails instead of that claim
    quietly going stale."""
    detail = _synthetic_detail()
    run_rows, rescore_rows = _run_both(monkeypatch, detail)

    run_keys = set(run_rows[0].keys())
    rescore_keys = set(rescore_rows[0].keys())
    assert run_keys - set(SHARED_FIELDS) == RUN_SERIES_EXTRA
    assert rescore_keys - set(SHARED_FIELDS) == RESCORE_SERIES_EXTRA
    assert len(run_keys) == 16
    assert len(rescore_keys) == 15
    assert len(run_keys & rescore_keys) == 13


def test_fragility_call_shape_matches(monkeypatch):
    """Spies on F.fragility() itself and asserts both call sites pass the same shock_periods/n_boot/
    seed kwargs, and per-metric-identical L2/w2/labels -- the actual "drift" risk this test file
    exists to catch (a future edit changing one call site's kwarg shape without the other)."""
    detail = _synthetic_detail()
    calls = []
    real_fragility = RM.F.fragility

    def spy(L2, w2, **kwargs):
        calls.append((dict(L2), np.asarray(w2).copy(), kwargs))
        return real_fragility(L2, w2, **kwargs)

    monkeypatch.setattr(RM, "F", type("F", (), {"fragility": staticmethod(spy),
                                                  "breakdown_number": RM.F.breakdown_number})())
    monkeypatch.setattr(SM, "F", type("F", (), {"fragility": staticmethod(spy),
                                                  "breakdown_number": SM.F.breakdown_number})())
    _run_both(monkeypatch, detail)

    assert len(calls) == 4   # 2 metrics x 2 functions
    kwarg_shapes = [{k: v for k, v in c[2].items() if k != "labels"} for c in calls]
    for shape in kwarg_shapes[1:]:
        assert shape == kwarg_shapes[0], "shock_periods/n_boot/seed kwargs differ across call sites"

"""best_arm.py's main() builds its per-(model,year) loss matrix L, and needs a guard against a
missing model-year cell. Unguarded, a missing cell makes the inner `.mean()` average an empty slice
-> NaN, which studentized_gap() then silently classifies as t=+inf ("perfect separation") rather
than "undefined" -- NaN fails both of that
function's floor comparisons and falls through to the final `return np.inf`, feeding
H = sum(1/t^2) as if the gap were infinitely wide (needs ZERO more samples), backwards for a
genuine data gap. Fixed by dropping any year with an incomplete model set before computing
studentized_gap/H, the same drop-incomplete-years pattern kstar_persistence.py and
combination_benchmark.py already use.

Verified genuinely latent against the real curated panel (all 12 series have complete 6-model x
all-years coverage today) -- this test constructs a synthetic panel with a deliberately missing
cell, monkeypatching best_arm.py's I/O paths and emit()/flush() so it touches no real data files.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "code" / "pipeline"))
sys.path.insert(0, str(ROOT / "code" / "instrument"))

import best_arm as BA  # noqa: E402


def _write_predictions(path, rows):
    pd.DataFrame(rows, columns=["series", "model", "year", "sae"]).to_csv(path, index=False)


def test_missing_model_year_cell_drops_that_year_instead_of_propagating_inf(monkeypatch, tmp_path, capsys):
    models = ["m0", "m1", "m2"]
    years = [2020, 2021, 2022]
    rows = []
    # "complete" series: full 3 models x 3 years, m2 clearly worst so it's never the pooled best.
    rng = np.random.default_rng(0)
    for y in years:
        for i, m in enumerate(models):
            rows.append({"series": "complete", "model": m, "year": y,
                         "sae": 1.0 + 0.3 * i + 0.01 * rng.normal()})
    # "gappy" series: identical construction, EXCEPT model m2's 2021 row is missing entirely.
    for y in years:
        for i, m in enumerate(models):
            if m == "m2" and y == 2021:
                continue  # the missing (model, year) cell
            rows.append({"series": "gappy", "model": m, "year": y,
                         "sae": 1.0 + 0.3 * i + 0.01 * rng.normal()})

    pred_path = tmp_path / "predictions.csv"
    _write_predictions(pred_path, rows)
    mcs_path = tmp_path / "mcs_test.csv"
    pd.DataFrame({"series": ["complete", "gappy"], "MCS_abs_10": [1, 2]}).to_csv(mcs_path, index=False)

    monkeypatch.setattr(BA, "PRED", pred_path)
    monkeypatch.setattr(BA, "MCSF", mcs_path)
    monkeypatch.setattr(BA, "MULTI", tmp_path)
    (tmp_path / "results").mkdir()
    monkeypatch.setattr(BA, "emit", lambda *a, **k: None)
    monkeypatch.setattr(BA, "flush", lambda: None)

    BA.main()

    captured = capsys.readouterr()
    assert "gappy" in captured.out and "dropping 1 of 3 years" in captured.out, (
        "expected an explicit [warn] print naming the series and the dropped-year count; got:\n"
        + captured.out
    )

    out = pd.read_csv(tmp_path / "results" / "best_arm.csv").set_index("series")
    assert out.loc["gappy", "T"] == 2, (
        "the incomplete 2021 year must be dropped from T, not silently kept with a NaN/inf-tainted "
        f"gap -- got T={out.loc['gappy', 'T']}"
    )
    assert out.loc["complete", "T"] == 3, "the complete series must be completely unaffected"
    assert np.isfinite(out.loc["gappy", "gap2_t"]) and np.isfinite(out.loc["gappy", "H"]), (
        "gap2_t/H must be finite, ordinary values computed on the 2 complete years -- not +inf from "
        "the fixed bug, and not NaN from a botched fix"
    )


def test_series_with_zero_complete_years_fails_loudly(monkeypatch, tmp_path):
    """The edge of the missing-cell guard above: dropping EVERY year (a missing model-year cell in
    each of the series' years) leaves T=0. Unguarded, `L.mean(0)` on an empty array reproduces the
    exact backwards result the guard targets (gap2_t=inf, H=0.0), then crashes confusingly
    downstream (`round()` on NaN, far from the real cause). It must instead raise a clear
    RuntimeError naming the series, at the point where zero complete years is discovered."""
    # `models` is derived per-series from whichever models appear AT ALL for that series
    # (`sorted(g.model.unique())`) -- a model that never appears isn't "missing", it's simply not
    # in that series' model list. To force T=0, both m0 and m1 must each appear for SOME year (so
    # both land in `models`), but never together in the SAME year -- alternating which one is
    # present/absent guarantees every single year has an incomplete model set.
    rows = [
        {"series": "all_gappy", "model": "m0", "year": 2020, "sae": 1.0},   # m1 missing in 2020
        {"series": "all_gappy", "model": "m1", "year": 2021, "sae": 1.2},   # m0 missing in 2021
    ]

    pred_path = tmp_path / "predictions.csv"
    _write_predictions(pred_path, rows)
    mcs_path = tmp_path / "mcs_test.csv"
    pd.DataFrame({"series": ["all_gappy"], "MCS_abs_10": [1]}).to_csv(mcs_path, index=False)

    monkeypatch.setattr(BA, "PRED", pred_path)
    monkeypatch.setattr(BA, "MCSF", mcs_path)
    monkeypatch.setattr(BA, "MULTI", tmp_path)
    (tmp_path / "results").mkdir()
    monkeypatch.setattr(BA, "emit", lambda *a, **k: None)
    monkeypatch.setattr(BA, "flush", lambda: None)

    with pytest.raises(RuntimeError, match="all_gappy.*every year"):
        BA.main()


def test_no_missing_cells_in_the_real_curated_panel_today():
    """Confirms the guard is genuinely latent against the real curated panel, mirroring
    test_best_arm_degenerate_tie.py's own analogous check for the se==0 floor."""
    pred = ROOT / "multiseries" / "results" / "breadth_distinct" / "breadth_distinct_predictions.csv"
    if not pred.exists():
        pytest.skip("frozen loss matrix not available")
    d = pd.read_csv(pred)
    gaps = 0
    for sid, g in d.groupby("series"):
        models = sorted(g.model.unique())
        years = sorted(g.year.unique())
        L = np.array([[g[(g.model == m) & (g.year == y)]["sae"].mean() for m in models] for y in years])
        gaps += int(np.isnan(L).any(1).sum())
    assert gaps == 0, (
        f"{gaps} year(s) across the real curated panel now have an incomplete model set -- the "
        f"missing-cell guard is no longer purely latent, verify best_arm.csv/canonical values "
        f"before trusting them unchanged"
    )

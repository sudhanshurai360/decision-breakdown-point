"""social_choice.py's PERIOD-WEIGHTING ROBUSTNESS block (main(), the `nmonths_year` construction).

`nmonths_year = g.groupby("year").size()` is NOT a pure per-year month count: `g` spans every
model, so `.size()` counts (distinct months x distinct models) per year. This formula is left
UNCHANGED here -- rewriting it without new evidence that it is actually wrong would itself be a
risk, not a fix. It is HARMLESS as currently used: fragility()'s weighted
means are invariant to any positive, per-cell-constant weight rescaling, and the model count is
constant across years within a series in the real data (checked directly below and via a permanent
guard in social_choice.py itself, which raises RuntimeError if it is ever violated) -- so the extra
factor cancels and `nmonths_year`'s relative year-to-year weighting is identical to pure
month-count weighting despite not literally being one.

These tests lock in: (1) the real data currently satisfies the constant-model-count invariant the
harmlessness argument depends on, and (2) social_choice.py's guard would correctly fail loudly on a
synthetic panel that violates it, rather than silently reweighting years by a no-longer-constant
factor.
"""
import re
import textwrap
from pathlib import Path

import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SC_SRC = ROOT / "code" / "pipeline" / "social_choice.py"
PRED = ROOT / "multiseries" / "results" / "breadth_distinct" / "breadth_distinct_predictions.csv"


@pytest.fixture(scope="module")
def real_predictions():
    try:
        return pd.read_csv(PRED)
    except FileNotFoundError:
        pytest.skip("real breadth_distinct_predictions.csv not available")


def test_real_data_has_constant_model_count_per_series_year(real_predictions):
    """The invariant social_choice.py's guard enforces: every series in the real data has
    the SAME number of models represented in every year of that series' panel."""
    d = real_predictions
    for sid, g in d.groupby("series"):
        n_models_per_year = g.groupby("year")["model"].nunique()
        assert n_models_per_year.nunique() <= 1, (
            f"{sid}: model count varies by year -- the harmlessness argument for nmonths_year "
            f"no longer holds for this series"
        )


def test_size_equals_month_count_times_model_count_on_real_data(real_predictions):
    """Directly verifies the mechanism: g.groupby('year').size() == (distinct months) x
    (distinct models), confirming `nmonths_year` is what the comment says it is, not merely
    assumed to be."""
    d = real_predictions
    for sid, g in d.groupby("series"):
        size_based = g.groupby("year").size()
        month_based = g.groupby("year")["target"].nunique()
        n_models = g.groupby("year")["model"].nunique()
        assert (size_based == month_based * n_models).all(), f"{sid}: size() != months x models"


def _production_guard_source() -> str:
    """Lifts the guard's ACTUAL source text out of social_choice.py's main(). The guard is inline in
    main() and cannot be imported on its own, so the test below executes this extracted text rather
    than a hand-copied paraphrase of it: delete the guard, or downgrade it to a warning, and the
    extraction or the execution below fails."""
    src = SC_SRC.read_text(encoding="utf-8")
    m = re.search(
        r"^([ \t]*n_models_per_year = g\.groupby.*?raise RuntimeError\(.*?\)\n)",
        src,
        re.DOTALL | re.MULTILINE,
    )
    assert m is not None, (
        "social_choice.py no longer contains the `n_models_per_year` / `raise RuntimeError` guard "
        "that the nmonths_year harmlessness argument depends on"
    )
    return textwrap.dedent(m.group(1))


def test_production_guard_raises_on_synthetic_varying_model_count():
    """A panel that genuinely violates the invariant (a model added mid-panel) must trip
    social_choice.py's own guard -- and must trip it as a RuntimeError, i.e. a hard failure that
    survives `python -O`, not a bare assert."""
    d = pd.DataFrame({
        "series": ["X"] * 5,
        "model": ["a", "b", "a", "b", "c"],   # year 2 has 3 models, year 1 has 2
        "year": [1, 1, 2, 2, 2],
        "target": ["2020-01-01", "2020-02-01", "2021-01-01", "2021-02-01", "2021-03-01"],
    })
    g = d[d.series == "X"]
    assert g.groupby("year")["model"].nunique().nunique() > 1   # the fixture really does violate it

    guard = _production_guard_source()
    with pytest.raises(RuntimeError, match="model count varies by year"):
        exec(compile(guard, str(SC_SRC), "exec"), {"sid": "X", "g": g})


def test_production_guard_stays_silent_on_a_conforming_panel():
    """Companion: the same extracted guard must NOT fire on a panel with a constant model count --
    otherwise it would be a guard that rejects the real data too."""
    d = pd.DataFrame({
        "series": ["X"] * 4,
        "model": ["a", "b", "a", "b"],
        "year": [1, 1, 2, 2],
        "target": ["2020-01-01", "2020-02-01", "2021-01-01", "2021-02-01"],
    })
    g = d[d.series == "X"]
    exec(compile(_production_guard_source(), str(SC_SRC), "exec"), {"sid": "X", "g": g})

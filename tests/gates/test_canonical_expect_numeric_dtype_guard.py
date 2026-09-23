"""Regression test: canonical.py's `idxmax()`/`idxmin()`/`.rank()`/`sort_values()` call sites
(_sc_argmax, m4sub.distinct_winners/pareto_dominant/rank_corr_smape_rmsse, density.flip_rate_*)
had no dtype guard -- these pandas operations do NOT raise on an object/string-dtype column, they
silently operate LEXICOGRAPHICALLY instead of numerically (e.g. sort_values() on a string column
puts "10" before "2"), which would produce a wrong-but-plausible number rather than a crash if a
column ever drifted to string dtype (a future pandas version bump, or a hand-edited/stale CSV).
Currently inert -- every real column checked is genuinely float64 on disk -- but this is the one
gate file whose stated role is protecting a published number, so worth closing even though not
currently live.

Fixed with a shared `_expect_numeric()` helper, called immediately before each of these operations.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "code" / "gates"))

import canonical as C  # noqa: E402


def test_expect_numeric_passes_through_a_real_numeric_series():
    s = pd.Series([1.0, 2.0, 3.0], name="x")
    assert C._expect_numeric(s, "test") is s


def test_expect_numeric_raises_on_string_dtype():
    s = pd.Series(["1", "10", "2"], name="x")
    with pytest.raises(TypeError, match="expected column"):
        C._expect_numeric(s, "test-source")


def test_expect_numeric_error_names_the_source_and_column():
    s = pd.Series(["a", "b"], name="my_col")
    with pytest.raises(TypeError, match="my_col") as exc_info:
        C._expect_numeric(s, "my_source.csv")
    assert "my_source.csv" in str(exc_info.value)


def test_sc_argmax_raises_if_concentration_column_is_string_dtype(monkeypatch, tmp_path):
    """End-to-end: _sc_argmax() must raise a TypeError, not silently return a lexicographically
    wrong row, if social_choice.csv's concentration column were ever string-dtype. Forces object
    dtype directly (pd.read_csv's own type inference would otherwise auto-convert a numeric-
    looking string column like "10.5"/"2.3" back to float, defeating a CSV-round-trip approach)."""
    fake = pd.DataFrame({
        "series": ["A", "B"], "metric": ["mase", "mase"],
        "concentration": pd.array(["10.5", "2.3"], dtype=object),
    })
    d = tmp_path / "results"
    d.mkdir()
    monkeypatch.setattr(C, "RES", d)
    monkeypatch.setattr(C.pd, "read_csv", lambda path, **kw: fake)  # _sc_argmax reads exactly one CSV

    with pytest.raises(TypeError, match="concentration"):
        C._sc_argmax("cell")


def test_density_sorted_by_n_models_raises_if_n_models_is_string_dtype(monkeypatch, tmp_path):
    fake = pd.DataFrame({"n_models": pd.array(["2", "10", "4"], dtype=object),
                          "flip_rate": [0.1, 0.2, 0.3]})
    d = tmp_path / "results"
    d.mkdir()
    monkeypatch.setattr(C, "RES", d)
    monkeypatch.setattr(C.pd, "read_csv", lambda path, **kw: fake)

    with pytest.raises(TypeError, match="n_models"):
        C._density_sorted_by_n_models()


def test_real_gate_checks_still_pass_on_the_real_committed_csvs():
    """Sanity check: the fix must not change any real, currently-passing CHECKS entry -- confirms
    every real on-disk column these sites touch is genuinely numeric today."""
    if not (C.RES / "social_choice.csv").exists():
        pytest.skip("multiseries/results/social_choice.csv not available")
    # Must not raise.
    C._sc_argmax("cell")
    if (C.RES / "m4_density_curve.csv").exists():
        C._density_sorted_by_n_models()


# --- the same guard applied across the remaining unguarded sites -------------
# A blind re-derivation check noted ~15 more unguarded .max()/.min() sites sharing the exact
# same risk (silent lexicographic-not-numeric comparison on a string-dtype column). Closed all of
# them with the same _expect_numeric() helper. A representative sample below -- covering
# a helper-function site, an inline CHECKS-dict lambda, and a builtin max()/min()-on-an-Index site,
# the 3 distinct shapes the 15 fixes took -- confirms each actually raises on bad dtype, not just
# that the code compiles.

def test_shape_ratio_raises_if_sae_column_is_string_dtype(monkeypatch, tmp_path):
    fake = pd.DataFrame({
        "series": ["A", "A"], "target": ["2020-01-01", "2020-02-01"],
        "sae": pd.array(["10.5", "2.3"], dtype=object),
    })
    d = tmp_path / "results" / "breadth_distinct"
    d.mkdir(parents=True)
    monkeypatch.setattr(C, "RES", tmp_path / "results")
    monkeypatch.setattr(C.pd, "read_csv", lambda path, **kw: fake)
    with pytest.raises(TypeError, match="sae"):
        C._shape_ratio("monthly_max")


def test_mcssize_sweep_raises_if_alpha_column_is_string_dtype(monkeypatch, tmp_path):
    fake = pd.DataFrame({"alpha": pd.array(["0.10", "0.05"], dtype=object), "nonidentified": [1, 2]})
    d = tmp_path / "results"
    d.mkdir()
    monkeypatch.setattr(C, "RES", d)
    monkeypatch.setattr(C.pd, "read_csv", lambda path, **kw: fake)
    with pytest.raises(TypeError, match="alpha"):
        C._mcssize_sweep().sort_values("alpha")


def test_election_concentration_max_check_raises_if_concentration_is_string_dtype(monkeypatch, tmp_path):
    """An inline CHECKS-dict lambda (not a helper function) -- confirms the guard was actually
    wired into the lambda itself, not just proven for the shared helpers above."""
    fake = pd.DataFrame({"concentration": pd.array(["10.5", "2.3"], dtype=object)})
    d = tmp_path / "results"
    d.mkdir()
    monkeypatch.setattr(C, "RES", d)
    monkeypatch.setattr(C.pd, "read_csv", lambda path, **kw: fake)
    with pytest.raises(TypeError, match="concentration"):
        C.CHECKS["election.concentration_max"]()


def test_flagship_year_max_check_raises_if_index_is_string_dtype(monkeypatch, tmp_path):
    """The builtin max()/min()-on-an-Index shape (_flag_score().index), distinct from the pandas
    .max()/.min() shape every other site above uses."""
    fake = pd.DataFrame({"ar1": [1.0, 2.0]}, index=pd.Index(["1999", "2020"], name="year"))
    d = tmp_path / "results"
    d.mkdir()
    monkeypatch.setattr(C, "ROOT_RES", d)
    monkeypatch.setattr(C.pd, "read_csv", lambda path, **kw: fake)
    with pytest.raises(TypeError, match="year"):
        C.CHECKS["data.flagship_year_max"]()

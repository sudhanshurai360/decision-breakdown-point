"""Mutation-verified tests for code/instrument/flagship_fragility.py's main(). Two of its emits are
otherwise unprotected, against these two mutants (named F1/F2 below and referred to by those names
in this file only):

  F2  `mcs25 = [models[i] for i in surv10]` (copy-paste of the alpha=0.10 set) -> flagship.mcs_25
      would read "nbeats|nhits|tft" instead of "nbeats|tft"; the gate cannot catch it because it
      reads the CSV the same main() writes.
  F1  `get_loc("ar1")` without the `+ 1`                        -> flagship.ar1_rank_long_window
      "5 of 7" would become 4 of 7 (caught by the gate only, not by any test).

main() is run end-to-end with only its I/O boundaries replaced: RES points at a tmp dir holding
the scoreboard/contrast CSVs, emit()/flush() are captured, and the two expensive multi-seed null
computations in null_budget (stable_stability: 20 seeds x 5000 boots; stable_benchmark: the
exchangeable null) are stubbed because neither feeds the fields under test. F.fragility() and
MCS.mcs() are the REAL estimators.

Test 1 is a synthetic 12-year, 4-model scoreboard whose k*/responsible/concentration/AR1-rank are
hand-derived (docstring) and whose alpha=0.10 and alpha=0.25 confidence sets differ (the
elimination step's p-value, 0.231, lies between the two levels). Test 2 is a GOLDEN PIN on the
frozen results/extension_scoreboard_1988-2025.csv, recomputing through the real main() the
values S3.6 prints (canonical keys flagship.pooled_winner, .k_star, .responsible_years,
.concentration, .mcs_10, .mcs_25, .ar1_rank_long_window).
"""
import pandas as pd
import pytest

import flagship_fragility as FF
import mcs_test as MCS


def _run_main(monkeypatch, tmp_path, scoreboard: pd.DataFrame, contrast: pd.DataFrame):
    """Run FF.main() against `scoreboard` (years x models) with I/O boundaries stubbed; return
    ({key: emit-record}, flagship_fragility.csv row)."""
    res = tmp_path / "results"
    res.mkdir()
    scoreboard.to_csv(res / "extension_scoreboard_1988-2025.csv")
    contrast.to_csv(res / "contrast_exhibit.csv")
    monkeypatch.setattr(FF, "RES", res)

    emitted = {}

    def fake_emit(key, value, *, spec, cites=None, cross_check=None, **extra):
        assert key not in emitted, f"duplicate emit {key}"
        emitted[key] = {"value": value, **extra}

    monkeypatch.setattr(FF, "emit", fake_emit)
    monkeypatch.setattr(FF, "flush", lambda: None)
    monkeypatch.setattr(FF.NB, "stable_stability",
                        lambda L, w, n_boot=5000: {"value": 0.7, "lo": 0.69, "hi": 0.71, "sd": 0.01,
                                                   "n_seeds": 20, "n_boot": n_boot})
    monkeypatch.setattr(FF.NB, "stable_benchmark",
                        lambda K, T: {"median": 0.5, "p05": 0.3, "p95": 0.9, "p05_lo": 0.29,
                                      "p05_hi": 0.31, "p05_sd": 0.01, "p95_lo": 0.89, "p95_hi": 0.91,
                                      "p95_sd": 0.01, "n_eff": 100, "n_seeds": 20})
    FF.main()
    row = pd.read_csv(res / "flagship_fragility.csv").iloc[0]
    return emitted, row


# ---------------------------------------------------------------------------------------------
# Synthetic scoreboard: 12 years (2011-2022), models good / near / ar1 / bad
# ---------------------------------------------------------------------------------------------
GOOD = [0.64, 0.81, 0.65, 0.63, 1.04, 0.75, 0.66, 1.33, 0.72, 0.68, 1.38, 1.13]
NEAR = [0.92, 0.96, 0.41, 1.00, 1.07, 0.76, 0.47, 1.12, 0.84, 0.83, 1.49, 1.32]
AR1 = [1.41, 1.16, 0.87, 0.71, 1.18, 1.17, 1.33, 1.62, 0.56, 0.69, 1.28, 1.30]
BAD = [4.32, 4.40, 4.31, 4.08, 4.15, 4.05, 4.17, 4.83, 3.96, 4.51, 4.81, 4.98]
YEARS = list(range(2011, 2023))


def _synthetic_scoreboard():
    sb = pd.DataFrame({"good": GOOD, "near": NEAR, "ar1": AR1, "bad": BAD}, index=YEARS)
    ce = pd.DataFrame({"pooled_RT_2011-25": [1.0, 1.1, 1.2, 4.0], "pooled_1989-25": [0.9, 1.0, 1.1, 4.0]},
                      index=["good", "near", "ar1", "bad"])
    return sb, ce


def test_main_synthetic_mcs25_from_alpha_25_and_ar1_rank_is_one_based(monkeypatch, tmp_path):
    """kills: F2 (mcs25 built from surv10) and F1 (AR1 rank without +1).
    Column sums: good 10.42, near 11.19, ar1 13.28, bad 52.57 -> pooled means 0.868 < 0.933 <
    1.107 < 4.381, so the pooled ranking is good, near, ar1, bad and AR1 is rank 3 of 4 (1-based;
    get_loc alone would say 2).
    k* against near: c = near - good = [.28,.15,-.24,.37,.03,.01,-.19,-.21,.12,.15,.11,.19],
    M = 0.77; greedy removes .37 (2014) -> .40, .28 (2011) -> .12, .19 (2022) -> -.07 <= 0:
    k* = 3, responsible 2014;2011;2022, concentration = .37/.77 = 0.48052. near is the unique
    binding rival (ar1 needs 7 deletions, bad 12), so n_binding_tied = 1 and main() does not
    raise. good also wins the most years outright (7 of 12, against near 3 and ar1 2), so it is the
    per-period winner too -> no reversal.
    MCS (seed 0, B=2000, block 3): the first elimination (bad) is certain; the second step's
    p-value is 0.231, so at alpha=0.10 {good, near} survive but at alpha=0.25 only {good} does.
    The alpha=0.25 emit must therefore DIFFER from the alpha=0.10 one and equal MCS.mcs at 0.25."""
    sb, ce = _synthetic_scoreboard()
    e, row = _run_main(monkeypatch, tmp_path, sb, ce)

    assert e["flagship.pooled_winner"]["value"] == "good"
    assert e["flagship.k_star"]["value"] == 3 and e["flagship.k_star"]["of"] == 12
    assert e["flagship.k_star"]["rival"] == "near" and e["flagship.k_star"]["n_binding_tied"] == 1
    assert e["flagship.responsible_years"]["value"] == "2014;2011;2022"
    assert e["flagship.concentration"]["value"] == pytest.approx(round(0.37 / 0.77, 3))
    assert e["flagship.reversal"]["value"] is False

    Lmat = sb.to_numpy(float)
    models = list(sb.columns)
    exp10 = "|".join(models[i] for i in MCS.mcs(Lmat, alpha=0.10, seed=0)[0])
    exp25 = "|".join(models[i] for i in MCS.mcs(Lmat, alpha=0.25, seed=0)[0])
    assert exp10 == "good|near" and exp25 == "good", "fixture must separate the two alphas"
    assert e["flagship.mcs_10"]["value"] == exp10
    assert e["flagship.mcs_25"]["value"] == exp25
    assert e["flagship.mcs_25"]["of"] == 4
    assert row["MCS_25"] == "good" and int(row["MCS_25_size"]) == 1
    assert row["MCS_10"] == "good|near" and int(row["MCS_10_size"]) == 2

    assert e["flagship.ar1_rank_long_window"]["value"] == 3
    assert e["flagship.ar1_rank_long_window"]["of"] == 4


def test_main_refuses_to_emit_a_point_concentration_when_binding_rivals_tie(monkeypatch, tmp_path):
    """A mutant that emits concentration_hi in place of concentration is number-neutral only
    because n_binding_tied == 1 forces lo == hi == point, and main()'s RuntimeError guard is what
    keeps it that way. Pin the guard: duplicate `near` as `near2` so two rivals tie for binding
    (same k* = 3), and main() must raise before writing/emitting anything."""
    sb, ce = _synthetic_scoreboard()
    sb = sb.assign(near2=sb["near"].to_numpy())
    ce = pd.concat([ce, pd.DataFrame(ce.loc[["near"]].to_numpy(), index=["near2"], columns=ce.columns)])
    with pytest.raises(RuntimeError, match="n_binding_tied=2"):
        _run_main(monkeypatch, tmp_path, sb, ce)
    assert not (tmp_path / "results" / "flagship_fragility.csv").exists()


# ---------------------------------------------------------------------------------------------
# GOLDEN PIN on the frozen flagship scoreboard (S3.6's flagship exhibit)
# ---------------------------------------------------------------------------------------------
_SB = FF.RES / "extension_scoreboard_1988-2025.csv"
_CE = FF.RES / "contrast_exhibit.csv"


@pytest.mark.skipif(not (_SB.exists() and _CE.exists()),
                    reason="frozen flagship scoreboard / contrast exhibit not present in this checkout")
def test_main_real_scoreboard_pin_s36_flagship_exhibit(monkeypatch, tmp_path):
    """GOLDEN PIN (kills F1, F2 and any regression in the k*/MCS path on the real input).
    Recomputes, through the real main(), the S3.6 flagship numbers from the frozen
    results/extension_scoreboard_1988-2025.csv: nbeats wins pooled; k* = 3 of 38 against tft
    (years 1988, 2009, 2010); concentration 0.388 (the canonical flagship.concentration value;
    S3.6's prose prints it rounded, as 0.39); MCS alpha=0.10 = {nbeats, nhits, tft}, alpha=0.25
    = {nbeats, tft}; AR1 ranks 5 of 7 by pooled MASE. Protects canonical keys flagship.pooled_winner,
    flagship.k_star, flagship.responsible_years, flagship.concentration, flagship.mcs_10,
    flagship.mcs_25, flagship.ar1_rank_long_window (the 20-seed stability / exchangeable-null keys
    are NOT pinned here: their producers are stubbed for speed)."""
    sb = pd.read_csv(_SB, index_col=0)
    ce = pd.read_csv(_CE, index_col=0)
    e, row = _run_main(monkeypatch, tmp_path, sb, ce)

    assert e["flagship.pooled_winner"]["value"] == "nbeats"
    assert e["flagship.per_period_winner"]["value"] == "nbeats"
    assert e["flagship.k_star"]["value"] == 3 and e["flagship.k_star"]["of"] == 38
    assert e["flagship.k_star"]["rival"] == "tft" and e["flagship.k_star"]["n_binding_tied"] == 1
    assert e["flagship.responsible_years"]["value"] == "1988;2009;2010"
    assert e["flagship.concentration"]["value"] == pytest.approx(0.388, abs=5e-4)
    assert e["flagship.reversal"]["value"] is False
    assert e["flagship.condorcet_winner"]["value"] == "nbeats"
    assert e["flagship.mcs_10"]["value"] == "nbeats|nhits|tft"
    assert e["flagship.mcs_25"]["value"] == "nbeats|tft"
    assert e["flagship.ar1_rank_long_window"]["value"] == 5
    assert e["flagship.ar1_rank_long_window"]["of"] == 7
    assert row["MCS_25"] == "nbeats|tft" and int(row["MCS_25_size"]) == 2

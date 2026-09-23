"""Mutation-verified tests for combination_benchmark.py's `_pooled_best_rank()`.
`per_series_panels()`'s other tests use TWO-model panels, for which mean == median, so neither of
the two most plausible one-token slips in the S4.4 scoring step is detectable there:
  M1  `piv.mean(axis=1)`   -> `piv.median(axis=1)`   (median-of-forecasts combination;
                                                       paper would read 1/12 -> 10/12, rank 6 -> 4)
  M2  `np.median(singles_pooled)` -> `np.mean(...)`   (compare to the MEAN single;
                                                       comb.beats_median_single 1 -> 12)
  M3  rank `<` -> `<=`                                (no ties today; latent)
All three passed the 6 existing tests. This file uses THREE-model panels with asymmetric errors so
mean and median provably differ, tests `_pooled_best_rank()` directly on hand-built loss dicts, and
adds one fast golden pin of the S4.4 counts through the real per_series_panels()+_pooled_best_rank()
path on the frozen predictions file (no MCS / null_budget, so it runs in seconds).

Every test here was verified to FAIL on the named mutant and PASS on the frozen module. Relies on
tests/conftest.py for sys.path (does NOT insert code/ paths itself).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import combination_benchmark as CB  # noqa: E402  (path via tests/conftest.py)

COLS = ["series", "model", "year", "target", "y_pred", "y_true", "scale_abs", "sae"]


def _rows(series, model, y_pred, y_true=100.0, scale_abs=1.0, years=(2015, 2016, 2017, 2018, 2019)):
    return [dict(series=series, model=model, year=yr, target=f"{yr}-06-15", y_pred=y_pred,
                 y_true=y_true, scale_abs=scale_abs, sae=abs(y_true - y_pred) / scale_abs)
            for yr in years]


def test_combination_is_the_mean_of_point_forecasts_not_the_median(monkeypatch, tmp_path):
    """kills: M1 `piv.mean(axis=1)` -> `piv.median(axis=1)`.
    Three models forecast 90 / 100 / 200 against truth 100 (scale 1) every year:
      mean-of-forecasts  = 130 -> sae_comb = 30
      median-of-forecasts = 100 -> sae_comb = 0
    The module's spec text ('built from the point forecasts (mean of y_pred)') fixes the mean."""
    rows = _rows("S1", "lo", 90.0) + _rows("S1", "mid", 100.0) + _rows("S1", "hi", 200.0)
    pred = tmp_path / "pred.csv"
    pd.DataFrame(rows, columns=COLS).to_csv(pred, index=False)
    monkeypatch.setattr(CB, "PRED", pred)
    L, T = CB.per_series_panels()["S1"]
    assert T == 5
    np.testing.assert_allclose(L[CB.COMB], np.full(5, 30.0))
    assert not np.allclose(L[CB.COMB], 0.0), "median-of-forecasts would give 0 -- that is M1"
    np.testing.assert_allclose(L["lo"], 10.0)
    np.testing.assert_allclose(L["mid"], 0.0)
    np.testing.assert_allclose(L["hi"], 100.0)


def _L(singles, comb, T=5):
    L = {m: np.full(T, float(v)) for m, v in singles.items()}
    L[CB.COMB] = np.full(T, float(comb))
    return L


def test_pooled_best_rank_compares_to_the_median_single_not_the_mean():
    """kills: M2 `np.median(singles_pooled)` -> `np.mean(...)`.
    Singles pooled = {1, 2, 10}, combination = 3:  median single = 2 (comb does NOT beat it),
    mean single = 13/3 = 4.33 (comb WOULD beat it). Rank = 1 + #{singles < 3} = 1 + 2 = 3."""
    L = _L({"a": 1, "b": 2, "c": 10}, comb=3)
    pooled, best_single, comb_wins, med_single, rank = CB._pooled_best_rank(L, np.ones(5))
    assert pooled == {"a": 1.0, "b": 2.0, "c": 10.0, CB.COMB: 3.0}
    assert best_single == "a"
    assert comb_wins == False  # noqa: E712  -- `== False` covers np.bool_ as well as bool; `is False` would not
    assert med_single == 2.0, f"median of {{1,2,10}} is 2, got {med_single} (mean would be 4.33)"
    assert rank == 3
    assert not (pooled[CB.COMB] < med_single), "the combination must NOT beat the median single here"
    assert pooled[CB.COMB] < np.mean([1, 2, 10]), "sanity: it WOULD beat the mean -- so M2 is distinguishable"


def test_pooled_best_rank_uses_equal_year_weights_and_the_comb_wins_strictly():
    """Weighted pooling and the comb_wins direction: singles {5, 6, 7} vs combination 4 -> wins,
    rank 1, median single 6. With weights [1,1,1,1,0] and a combination whose 5th year is a huge
    outlier, that year must not count (np.average with weights)."""
    L = _L({"a": 5, "b": 6, "c": 7}, comb=4)
    L[CB.COMB][4] = 1e6
    w = np.array([1.0, 1.0, 1.0, 1.0, 0.0])
    pooled, best_single, comb_wins, med_single, rank = CB._pooled_best_rank(L, w)
    assert pooled[CB.COMB] == 4.0
    assert best_single == "a" and bool(comb_wins) is True
    assert med_single == 6.0 and rank == 1


def test_pooled_best_rank_tie_with_a_single_breaks_toward_the_combination():
    """kills: M3 rank `<` -> `<=` (and pins comb_wins strict `<` on an exact tie).
    Singles {3, 5, 10}, combination = 3: exactly tied with the best single. rank counts singles
    STRICTLY below the combination -> 1 + 0 = 1 (favours the combination); `<=` would give 2.
    comb_wins on an exact tie is False (strict)."""
    L = _L({"a": 3, "b": 5, "c": 10}, comb=3)
    pooled, best_single, comb_wins, med_single, rank = CB._pooled_best_rank(L, np.ones(5))
    assert rank == 1
    assert bool(comb_wins) is False
    assert med_single == 5.0


# --- GOLDEN PIN on the frozen curated panel ------------------------------------------------------

_ROOT = Path(__file__).resolve().parents[2]
_REAL_PRED = _ROOT / "multiseries" / "results" / "breadth_distinct" / "breadth_distinct_predictions.csv"


@pytest.mark.skipif(not _REAL_PRED.exists(), reason="frozen breadth_distinct_predictions.csv not available")
def test_golden_pin_s44_counts_from_frozen_predictions(monkeypatch):
    """GOLDEN PIN: the three S4.4 headline counts and their
    five-model counterparts, recomputed through the REAL per_series_panels() + _pooled_best_rank()
    on the frozen predictions file (no MCS/null_budget, so this is seconds not hours). Protects
    comb.beats_best_single (1 of 12), comb.beats_median_single (1 of 12), comb.median_rank_of_7
    (6.0), comb.rank6_series (11), comb.sn_worst_series (11), comb.winning_series (RSAFS),
    comb5.beats_best_single (1), comb5.beats_median_single (11), comb5.median_rank_of_7 (3.0).
    Every one of these cells was re-derived independently.
    kills: M1 (beats_median 1 -> 10, median rank 6 -> 4), M2 (beats_median 1 -> 12) on real data."""
    monkeypatch.setattr(CB, "PRED", _REAL_PRED)

    def _summ(panels):
        wins, beats_med, ranks, sn_worst, winners = 0, 0, [], 0, []
        for sid, (L, T) in panels.items():
            pooled, best, cw, med, rank = CB._pooled_best_rank(L, np.ones(T))
            wins += int(cw); beats_med += int(pooled[CB.COMB] < med); ranks.append(rank)
            worst = max((m for m in pooled if m != CB.COMB), key=lambda m: pooled[m])
            sn_worst += int(worst == CB.BENCHMARK_SN)
            if cw:
                winners.append(sid)
        return len(panels), wins, beats_med, float(np.median(ranks)), sum(r == 6 for r in ranks), sn_worst, winners

    n, wins, beats_med, med_rank, n_rank6, sn_worst, winners = _summ(CB.per_series_panels())
    assert n == 12
    assert wins == 1 and winners == ["RSAFS"]
    assert beats_med == 1
    assert med_rank == 6.0
    assert n_rank6 == 11
    assert sn_worst == 11

    n5, wins5, beats_med5, med_rank5, _, _, _ = _summ(CB.per_series_panels(exclude=(CB.BENCHMARK_SN,)))
    assert n5 == 12
    assert wins5 == 1
    assert beats_med5 == 11
    assert med_rank5 == 3.0

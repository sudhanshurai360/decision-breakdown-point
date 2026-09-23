"""Mutation-verified tests for kstar_persistence.py's main() aggregation and its
curated loader: three number-changing mutations survive
tests/pipeline/test_kstar_persistence.py (which covers split_half/analyze_series only):

  M2  grouping `k1 <= 1` / `k1 > 1` -> `k1 < 1` / `k1 >= 1`   (S13.5's "10 of 37" / "33 of 49")
  M4  curated_series_losses() reads ["msse"] instead of ["sae"]  (curated arm on the wrong loss)
  M5  Spearman on raw `k1` instead of `k1_over_T1`               (S13.5's "+0.37 (p<0.001)")

main() is exercised here end-to-end with its I/O boundaries swapped for hand-built fixtures:
the per-series rows come from a stubbed analyze_series (that function has its own hand-verified
tests already), emit()/flush() are captured, OUT goes to tmp_path. Every expected value is
derived by hand in the docstrings.
"""
import numpy as np
import pandas as pd
import pytest

import kstar_persistence as KP


# ---------------------------------------------------------------------------------------------
# curated_series_losses(): per-year MEAN of `sae` (absolute, MASE-style), NaN-years dropped
# ---------------------------------------------------------------------------------------------
def test_curated_series_losses_is_per_year_mean_sae_not_msse(monkeypatch, tmp_path):
    """kills: M4 (["sae"] -> ["msse"]).
    One series, models A/B, years 2000-2001 with two targets each (plus a 2002 row for A only):
      A 2000: sae [0.2, 0.4] -> 0.3 ; msse [1.0, 1.0] -> 1.0
      A 2001: sae [0.6, 0.8] -> 0.7 ; msse [2.0, 2.0] -> 2.0
      B 2000: sae [0.5, 0.5] -> 0.5 ; msse [3.0, 3.0] -> 3.0
      B 2001: sae [0.9, 0.1] -> 0.5 ; msse [4.0, 4.0] -> 4.0
      A 2002: sae [9.9]              (B has no 2002 row -> that year is NaN for B -> dropped)
    Expected La (years x sorted models) = [[0.3, 0.5], [0.7, 0.5]], yrs_kept = [2000, 2001].
    Every msse value differs from its sae mean, so a loader on the wrong column cannot match."""
    rows = []
    def add(model, year, targets, sae, msse):
        for t, s, q in zip(targets, sae, msse):
            rows.append({"series": "S1", "model": model, "target": f"{year}-{t:02d}-01", "year": year,
                         "y_true": 1.0, "y_pred": 1.0, "scale_abs": 1.0, "scale_sq": 1.0,
                         "sae": s, "msse": q})
    add("A", 2000, [1, 2], [0.2, 0.4], [1.0, 1.0])
    add("A", 2001, [1, 2], [0.6, 0.8], [2.0, 2.0])
    add("B", 2000, [1, 2], [0.5, 0.5], [3.0, 3.0])
    add("B", 2001, [1, 2], [0.9, 0.1], [4.0, 4.0])
    add("A", 2002, [1], [9.9], [5.0])
    pred = tmp_path / "pred.csv"
    pd.DataFrame(rows).to_csv(pred, index=False)
    monkeypatch.setattr(KP, "PRED", pred)

    out = KP.curated_series_losses()
    assert list(out) == ["S1"]
    La, models, yrs = out["S1"]
    assert models == ["A", "B"]
    assert yrs == [2000, 2001], "the year with a NaN for one model must be dropped"
    np.testing.assert_allclose(La, np.array([[0.3, 0.5], [0.7, 0.5]]))


# ---------------------------------------------------------------------------------------------
# main(): the k*<=1 / k*>1 split, the Spearman inputs, and the persistence counts
# ---------------------------------------------------------------------------------------------
def _row(sid, source, k1, T1, persists, rank_percentile, loss_gap, excluded=False):
    T2 = T1
    if excluded:
        return {"series": sid, "source": source, "T": T1 + T2, "T1": T1, "T2": T2, "excluded": True,
                "winner1": None, "k1": np.nan, "k1_over_T1": np.nan, "winner_stability1": np.nan,
                "winner2": None, "persists": None, "rank1_in_2": np.nan, "rank_percentile": np.nan,
                "loss_gap": np.nan}
    return {"series": sid, "source": source, "T": T1 + T2, "T1": T1, "T2": T2, "excluded": False,
            "winner1": "A", "k1": int(k1), "k1_over_T1": float(k1 / T1), "winner_stability1": 0.5,
            "winner2": "A" if persists else "B", "persists": bool(persists),
            "rank1_in_2": 0 if persists else 1, "rank_percentile": float(rank_percentile),
            "loss_gap": float(loss_gap)}


# Six included rows (4 curated, 2 fredmd) + one excluded curated row. Chosen so that
#   (i)  k1 == 1 rows exist (the <= vs < boundary), with k1 = 0 also present;
#   (ii) the rank order of k1/T1 differs from the rank order of raw k1 (r4: 2/8 = 0.25 > r5: 3/16
#        = 0.1875, whereas raw k1 orders them the other way), so the two Spearman inputs give
#        different rho.
ROWS = {
    "r0": _row("r0", "curated", 0, 3, None, np.nan, np.nan, excluded=True),
    "r1": _row("r1", "curated", 0, 8, False, 0.00, 0.2),    # k/T 0.000
    "r2": _row("r2", "curated", 1, 8, True, 1.00, 0.0),     # k/T 0.125
    "r3": _row("r3", "curated", 1, 10, False, 0.50, 0.4),   # k/T 0.100
    "r4": _row("r4", "curated", 2, 8, True, 1.00, 0.0),     # k/T 0.250
    "r5": _row("r5", "fredmd", 3, 16, True, 1.00, 0.0),     # k/T 0.1875
    "r6": _row("r6", "fredmd", 5, 10, False, 0.25, 0.9),    # k/T 0.500
}


@pytest.fixture
def captured_main(monkeypatch, tmp_path):
    """Run KP.main() with its I/O boundaries stubbed; return {key: emit-record}."""
    emitted = {}

    def fake_emit(key, value, *, spec, cites=None, cross_check=None, **extra):
        assert key not in emitted, f"duplicate emit {key}"
        emitted[key] = {"value": value, **extra}

    curated = {s: (None, None, None) for s, r in ROWS.items() if r["source"] == "curated"}
    fredmd = {s: (None, None, None) for s, r in ROWS.items() if r["source"] == "fredmd"}
    monkeypatch.setattr(KP, "curated_series_losses", lambda: curated)
    monkeypatch.setattr(KP, "fredmd_series_losses", lambda: fredmd)

    def fake_analyze(sid, La, models, yrs, source):
        assert ROWS[sid]["source"] == source, (sid, source)
        return dict(ROWS[sid])

    monkeypatch.setattr(KP, "analyze_series", fake_analyze)
    monkeypatch.setattr(KP, "emit", fake_emit)
    monkeypatch.setattr(KP, "flush", lambda: None)
    monkeypatch.setattr(KP, "OUT", tmp_path / "kstar_persistence.csv")
    KP.main()
    return emitted


def test_main_groups_on_first_half_kstar_at_most_one(captured_main):
    """kills: M2 (k1 <= 1 -> k1 < 1, k1 > 1 -> k1 >= 1).
    Included rows: k1 = [0, 1, 1, 2, 3, 5]; persists = [F, T, F, T, T, F].
      k*<=1 group = {r1, r2, r3}: n = 3, persisting = 1 (r2)      -> rate 1/3
      k*>1  group = {r4, r5, r6}: n = 3, persisting = 2 (r4, r5)  -> rate 2/3
    A `k1 < 1` split would put only r1 in the small group (n=1, rate 0)."""
    e = captured_main
    assert e["kstar_persistence.n_kstar1"]["value"] == 3
    assert e["kstar_persistence.n_kstar_gt1"]["value"] == 3
    assert e["kstar_persistence.n_persist_kstar1"]["value"] == 1
    assert e["kstar_persistence.n_persist_kstar_gt1"]["value"] == 2
    assert e["kstar_persistence.persist_rate_kstar1"]["value"] == pytest.approx(1 / 3)
    assert e["kstar_persistence.persist_rate_kstar_gt1"]["value"] == pytest.approx(2 / 3)
    assert e["kstar_persistence.persist_rate_kstar1"]["n"] == 3
    assert e["kstar_persistence.persist_rate_kstar_gt1"]["n"] == 3


def test_main_spearman_uses_k1_over_T1_not_raw_k1(captured_main):
    """kills: M5 (Spearman on raw k1).
    k1/T1 ranks (ascending): r1=1, r3=2, r2=3, r5=4, r4=5, r6=6 (all distinct).
    (a) vs persists y = [0, 0, 1, 1, 1, 0] in that rank order. Spearman = Pearson(rank, y):
        mean rank 3.5, mean y 0.5; sum (r-3.5)(y-0.5) = 1.25+0.75-0.25+0.25+0.75-1.25 = 1.5;
        sum (r-3.5)^2 = 17.5; sum (y-0.5)^2 = 1.5 -> rho = 1.5/sqrt(26.25) = 0.29277.
        With raw k1 = [0,1,1,2,3,5] (ranks 1, 2.5, 2.5, 4, 5, 6 for r1,r2,r3,r4,r5,r6):
        numerator 1.25-0.5+0.5+0.25+0.75-1.25 = 1.0; sum (r-3.5)^2 = 17 -> 0.19803. Different.
    (b) vs rank_percentile: y = [0, 0.5, 1, 1, 1, 0.25] in k/T rank order -> y-ranks
        [1, 3, 5, 5, 5, 2]; sum (r-3.5)(ry-3.5) = 6.25+0.75-0.75+0.75+2.25-3.75 = 5.5;
        sum (ry-3.5)^2 = 15.5 -> rho = 5.5/sqrt(17.5*15.5) = 0.33397."""
    e = captured_main
    assert e["kstar_persistence.spearman_kstar_over_T_vs_persist"]["value"] == pytest.approx(
        1.5 / np.sqrt(17.5 * 1.5), abs=1e-9)
    assert e["kstar_persistence.spearman_kstar_over_T_vs_rank_percentile"]["value"] == pytest.approx(
        5.5 / np.sqrt(17.5 * 15.5), abs=1e-9)
    assert e["kstar_persistence.spearman_kstar_over_T_vs_persist"]["n"] == 6


def test_main_counts_and_mean_gap_hand_computed(captured_main, tmp_path):
    """Overall bookkeeping on the same fixture: 7 attempted, 1 excluded, 6 included (4 curated,
    2 fredmd), 3 persisting -> 0.5; non-persisting rows r1, r3, r6 have loss_gap 0.2, 0.4, 0.9
    -> mean 0.5 over n=3. Also: the CSV main() writes has one row per attempted series."""
    e = captured_main
    assert e["kstar_persistence.n_series_attempted"]["value"] == 7
    assert e["kstar_persistence.n_excluded_min_T"]["value"] == 1
    assert e["kstar_persistence.n_included"]["value"] == 6
    assert e["kstar_persistence.n_curated_included"]["value"] == 4
    assert e["kstar_persistence.n_fredmd_included"]["value"] == 2
    assert e["kstar_persistence.n_persist"]["value"] == 3
    assert e["kstar_persistence.persist_rate_overall"]["value"] == pytest.approx(0.5)
    assert e["kstar_persistence.mean_loss_gap_when_not_persisting"]["value"] == pytest.approx(0.5)
    assert e["kstar_persistence.mean_loss_gap_when_not_persisting"]["n"] == 3
    written = pd.read_csv(tmp_path / "kstar_persistence.csv")
    assert len(written) == 7 and set(written.series) == set(ROWS)

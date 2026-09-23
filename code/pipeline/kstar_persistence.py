"""
kstar_persistence.py — does k* predict anything a practitioner cares about?
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
# --- paths, following the same convention every other code/pipeline/ stage uses ----
ROOT = HERE.parent.parent                  # <repo root>/code/<subdir>/ -> <repo root>
MULTI = ROOT / "multiseries"                # data + results live HERE, not under code/
INSTR = ROOT / "code" / "instrument"        # shared estimator

sys.path.insert(0, str(HERE)); sys.path.insert(0, str(INSTR))
import fragility as F
import fred_md_sweep as FS                  # load_fredmd() + series_losses() — reused, not reimplemented
from canonical_emit import emit, flush

PRED = MULTI / "results" / "breadth_distinct" / "breadth_distinct_predictions.csv"
OUT = MULTI / "results" / "kstar_persistence.csv"
MIN_T_PER_HALF = 8                          # same floor used elsewhere (mcs_test.MIN_T_FOR_ZERO_SE)
                                             # for a stable per-half mean
CITE = ["S13.5: k* predictive-persistence check"]


def curated_series_losses():
    """{series: (La, models, years)} for the 12 curated series, La = years x models absolute
    (sae, MASE-style) per-year mean loss matrix."""
    d = pd.read_csv(PRED)
    out = {}
    for sid, g in d.groupby("series"):
        models = sorted(g.model.unique())
        yrs = sorted(g.year.unique())
        La = np.full((len(yrs), len(models)), np.nan)
        for j, m in enumerate(models):
            gm = g[g.model == m].groupby("year")["sae"].mean()
            for i, y in enumerate(yrs):
                if y in gm.index:
                    La[i, j] = gm[y]
        keep = ~np.isnan(La).any(1)
        if (~keep).sum():
            print(f"  [warn] {sid}: dropping {(~keep).sum()} year(s) with a NaN before the persistence test")
        La = La[keep]
        yrs_kept = [y for y, k in zip(yrs, keep) if k]
        out[sid] = (La, models, yrs_kept)
    return out


def fredmd_series_losses():
    """{series: (La, models, years)} for the FRED-MD population sweep, via fred_md_sweep.py's own
    load_fredmd()/series_losses() — identical filters (len(s)>=180, series_losses' internal eval-year
    gate, all-finite La) to fred_md_sweep.main()'s own usable-series loop, so the population here is
    the same 74-series population that stage reports."""
    df = FS.load_fredmd()
    out = {}
    for col in df.columns:
        s = FS.load_series_gap_safe(df, col)
        if s is None:
            continue
        res = FS.series_losses(s.to_numpy(float), s.index)
        if res is None:
            print(f"  [skip] {col}: series_losses() returned no result", flush=True)
            continue
        La, _, models, yrs = res
        if not np.all(np.isfinite(La)):
            print(f"  [skip] {col}: non-finite loss in the levels arm (La)", flush=True)
            continue
        out[col] = (La, models, yrs)
    return out


def split_half(La, yrs):
    """Chronological midpoint split: first ceil(T/2) years vs the rest."""
    T = len(yrs)
    n1 = int(np.ceil(T / 2))
    return La[:n1], La[n1:], yrs[:n1], yrs[n1:]


def analyze_series(sid, La, models, yrs, source):
    """One row of the honest test: first-half k* vs second-half persistence, for one series."""
    T = len(yrs)
    La1, La2, yrs1, yrs2 = split_half(La, yrs)
    T1, T2 = len(yrs1), len(yrs2)
    if T1 < MIN_T_PER_HALF or T2 < MIN_T_PER_HALF:
        return {"series": sid, "source": source, "T": T, "T1": T1, "T2": T2, "excluded": True,
                "winner1": None, "k1": np.nan, "k1_over_T1": np.nan, "winner_stability1": np.nan,
                "winner2": None, "persists": None, "rank1_in_2": np.nan, "rank_percentile": np.nan,
                "loss_gap": np.nan}
    L1 = {m: La1[:, j] for j, m in enumerate(models)}
    L2 = {m: La2[:, j] for j, m in enumerate(models)}
    w1, w2 = np.ones(T1), np.ones(T2)
    winner1 = F.pooled_winner(L1, w1)
    k1, _opp1, _removed1 = F.decision_breakdown(L1, w1, a=winner1)
    stab1, _freq1 = F.winner_stability(L1, w1, seed=0)
    winner2 = F.pooled_winner(L2, w2)
    persists = bool(winner2 == winner1)
    means2 = {m: float(np.average(L2[m], weights=w2)) for m in L2}
    order2 = sorted(means2, key=lambda m: means2[m])           # best (lowest loss) first, on SECOND half
    Kmods = len(order2)
    rank1_in_2 = order2.index(winner1)                          # 0 = still the best on the second half
    rank_percentile = (1.0 - rank1_in_2 / (Kmods - 1)) if Kmods > 1 else np.nan
    loss_gap = max(0.0, means2[winner1] - means2[winner2])       # 0 exactly when persists is True
    return {
        "series": sid, "source": source, "T": T, "T1": T1, "T2": T2, "excluded": False,
        "winner1": winner1, "k1": int(k1), "k1_over_T1": float(k1 / T1),
        "winner_stability1": float(stab1), "winner2": winner2, "persists": persists,
        "rank1_in_2": int(rank1_in_2), "rank_percentile": float(rank_percentile),
        "loss_gap": float(loss_gap),
    }


def main():
    rows = []
    for sid, (La, models, yrs) in curated_series_losses().items():
        rows.append(analyze_series(sid, La, models, yrs, "curated"))
    for sid, (La, models, yrs) in fredmd_series_losses().items():
        rows.append(analyze_series(sid, La, models, yrs, "fredmd"))
    R = pd.DataFrame(rows)
    R.to_csv(OUT, index=False)

    n_attempted = len(R)
    inc = R[~R.excluded].copy()
    inc["persists"] = inc["persists"].astype(bool)
    excl = R[R.excluded]
    n_excluded = int(len(excl))
    n_included = int(len(inc))
    n_curated_included = int((inc.source == "curated").sum())
    n_fredmd_included = int((inc.source == "fredmd").sum())

    n_persist = int(inc.persists.sum())
    persist_rate = n_persist / n_included

    g_small = inc[inc.k1 <= 1]
    g_large = inc[inc.k1 > 1]
    n_small, n_large = int(len(g_small)), int(len(g_large))
    rate_small = float(g_small.persists.mean()) if n_small else float("nan")
    rate_large = float(g_large.persists.mean()) if n_large else float("nan")

    rho_rank, p_rank = stats.spearmanr(inc.k1_over_T1, inc.rank_percentile)
    rho_pers, p_pers = stats.spearmanr(inc.k1_over_T1, inc.persists.astype(int))
    t1_lo, t1_hi = int(inc.T1.min()), int(inc.T1.max())

    nonpersist = inc[~inc.persists]
    n_nonpersist = int(len(nonpersist))
    mean_gap = float(nonpersist.loss_gap.mean()) if n_nonpersist else float("nan")

    emit("kstar_persistence.n_series_attempted", n_attempted,
         spec="series considered before the >=8-years-per-half minimum-T gate: the 12 curated series "
              "(breadth_distinct_predictions.csv) plus the 74 series surviving fred_md_sweep.py's own "
              "length/finiteness filters",
         cites=CITE,
         cross_check="kstar_persistence.csv: len(R)")
    emit("kstar_persistence.n_excluded_min_T", n_excluded, of=n_attempted,
         spec=f"series excluded because the first or second half of the chronological split has fewer "
              f"than MIN_T_PER_HALF={MIN_T_PER_HALF} years",
         cites=CITE,
         cross_check="kstar_persistence.csv: excluded.sum()")
    emit("kstar_persistence.n_included", n_included, of=n_attempted,
         spec="series analyzed after the minimum-T-per-half gate; n_series_attempted - n_excluded_min_T",
         cites=CITE,
         cross_check="kstar_persistence.csv: (~excluded).sum()")
    emit("kstar_persistence.n_curated_included", n_curated_included, of=n_included,
         spec="of n_included, how many are curated (vs FRED-MD population) series",
         cites=CITE,
         cross_check="kstar_persistence.csv: ((~excluded) & (source=='curated')).sum()")
    emit("kstar_persistence.n_fredmd_included", n_fredmd_included, of=n_included,
         spec="of n_included, how many are FRED-MD population-sweep series",
         cites=CITE,
         cross_check="kstar_persistence.csv: ((~excluded) & (source=='fredmd')).sum()")
    emit("kstar_persistence.n_persist", n_persist, of=n_included,
         spec="included series where the first-half pooled winner is STILL the pooled winner on the "
              "second half (exact model-name match)",
         cites=CITE,
         cross_check="kstar_persistence.csv: ((~excluded) & persists).sum()")
    emit("kstar_persistence.persist_rate_overall", persist_rate,
         spec="n_persist / n_included",
         cites=CITE,
         cross_check="kstar_persistence.csv: kstar_persistence.n_persist / kstar_persistence.n_included")
    emit("kstar_persistence.n_kstar1", n_small,
         spec="included series with first-half k*<=1 (k*=1, the task's 'most fragile' classification, "
              "plus the rarer degenerate k*=0 exact-tie case, at least as fragile as k*=1)",
         cites=CITE,
         cross_check="kstar_persistence.csv: ((~excluded) & (k1 <= 1)).sum()")
    emit("kstar_persistence.n_kstar_gt1", n_large,
         spec="included series with first-half k*>1 ('larger k*', the comparison group)",
         cites=CITE,
         cross_check="kstar_persistence.csv: ((~excluded) & (k1 > 1)).sum()")
    emit("kstar_persistence.persist_rate_kstar1", rate_small, n=n_small,
         spec="second-half persistence rate among the k*<=1 ('most fragile') group",
         cites=CITE,
         cross_check="kstar_persistence.csv: persists.mean() restricted to (~excluded) & (k1 <= 1)")
    emit("kstar_persistence.persist_rate_kstar_gt1", rate_large, n=n_large,
         spec="second-half persistence rate among the k*>1 ('larger k*') group",
         cites=CITE,
         cross_check="kstar_persistence.csv: persists.mean() restricted to (~excluded) & (k1 > 1)")
    emit("kstar_persistence.n_persist_kstar1", int(g_small.persists.sum()), of=n_small,
         spec="included series with first-half k*<=1 whose first-half winner persists into the "
              "second half -- the raw numerator of persist_rate_kstar1",
         cites=CITE,
         cross_check="kstar_persistence.csv: ((~excluded) & (k1 <= 1) & persists).sum()")
    emit("kstar_persistence.n_persist_kstar_gt1", int(g_large.persists.sum()), of=n_large,
         spec="included series with first-half k*>1 whose first-half winner persists into the "
              "second half -- the raw numerator of persist_rate_kstar_gt1",
         cites=CITE,
         cross_check="kstar_persistence.csv: ((~excluded) & (k1 > 1) & persists).sum()")
    emit("kstar_persistence.spearman_kstar_over_T_vs_rank_percentile", float(rho_rank), n=n_included,
         p_value=float(p_rank),
         spec=f"Spearman rank correlation, across included series, between first-half k*/T1 (normalized "
              f"breakdown point, since T1 varies {t1_lo}-{t1_hi} across this population and a raw k* "
              f"count is not comparable across series with different-length first halves) and the "
              f"second-half rank-percentile of the first-half winner among all candidate models (1.0 = "
              f"still the best model on the second half; lower = it fell further down the second-half "
              f"ranking). This is the headline continuous predictive-validity check for k*",
         cites=CITE,
         cross_check="kstar_persistence.csv: spearman(k1_over_T1, rank_percentile) over ~excluded rows")
    emit("kstar_persistence.spearman_kstar_over_T_vs_persist", float(rho_pers), n=n_included,
         p_value=float(p_pers),
         spec="Spearman rank correlation, across included series, between first-half k*/T1 and the "
              "binary second-half persistence indicator (1 = same model still wins, 0 = it does not)",
         cites=CITE,
         cross_check="kstar_persistence.csv: spearman(k1_over_T1, persists.astype(int)) over ~excluded rows")
    emit("kstar_persistence.mean_loss_gap_when_not_persisting", mean_gap, n=n_nonpersist,
         spec="mean second-half loss gap (second-half winner's mean loss subtracted from the first-half "
              "winner's mean second-half loss) among the n_included - n_persist series where the "
              "first-half winner does NOT persist -- how badly, on average, the stale winner loses when "
              "it loses",
         cites=CITE,
         cross_check="kstar_persistence.csv: loss_gap.mean() restricted to (~excluded) & (~persists)")
    flush()

    pd.set_option("display.width", 200)
    print("=== k* PREDICTIVE-PERSISTENCE CHECK: does first-half k* predict second-half winner persistence? ===")
    print(f"  series attempted: {n_attempted}  (12 curated + {n_attempted - 12} FRED-MD population)")
    print(f"  excluded (< {MIN_T_PER_HALF} years in a half): {n_excluded}")
    print(f"  included: {n_included}  ({n_curated_included} curated, {n_fredmd_included} FRED-MD)")
    print(R.to_string(index=False))
    print(f"\n  overall second-half persistence rate: {n_persist}/{n_included} = {persist_rate:.3f}")
    print("\n  --- grouped comparison: k*<=1 ('most fragile') vs k*>1 ('larger k*') ---")
    print(f"  k*<=1: {n_small} series, persistence rate {rate_small:.3f}")
    print(f"  k*>1 : {n_large} series, persistence rate {rate_large:.3f}")
    print(f"  difference (k*>1 minus k*<=1): {rate_large - rate_small:+.3f}" if n_small and n_large else
          "  (one group empty -- no difference computable)")
    print("\n  --- Spearman correlations (single chronological-midpoint split, not p-hacked over split points) ---")
    print(f"  Spearman(k1/T1, second-half rank-percentile of first-half winner) = {rho_rank:+.3f}  (p={p_rank:.3f}, n={n_included})")
    print(f"  Spearman(k1/T1, persists 0/1)                                     = {rho_pers:+.3f}  (p={p_pers:.3f}, n={n_included})")
    print(f"\n  mean loss gap when the first-half winner does NOT persist: {mean_gap:.4f}  (n={n_nonpersist})")
    tag = "PREDICTIVE" if (abs(rho_rank) >= 0.3 and p_rank < 0.05) else "NOT DEMONSTRATED PREDICTIVE at this n"
    print(f"\n  READING: k* is {tag} of second-half winner persistence in this single honest split.")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()

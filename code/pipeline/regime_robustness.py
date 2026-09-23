"""regime_robustness.py — IS THE NON-IDENTIFICATION JUST COVID?
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "instrument"))
import mcs_test as MT
import run_multiseries as RM   # for its shared _atomic_to_csv() helper
from canonical_emit import emit, flush
RES = HERE.parent.parent / "multiseries" / "results"
COVID = (2020, 2021)
N_PLACEBO = 200   # random placebo draws per series per deletion size
ALPHA = 0.10


def per_series_panels():
    """{series: (years, {model: per-year mean absolute scaled loss})} for the curated 12."""
    d = pd.read_csv(MT.PRED)
    out = {}
    for sid, g in d.groupby("series"):
        per = g.groupby(["model", "year"])["sae"].mean().reset_index()
        piv = per.pivot(index="year", columns="model", values="sae").dropna()
        out[sid] = (list(piv.index), {m: piv[m].to_numpy(float) for m in piv.columns})
    return out


def mcs_size(L: dict, mask: np.ndarray) -> int:
    """|MCS| on the sub-panel selected by a boolean mask over periods. L: {model: array(T)}."""
    models = sorted(L)
    M = np.column_stack([L[m][mask] for m in models])
    return len(MT.mcs(M, alpha=ALPHA, seed=0)[0])


def main():
    panels = per_series_panels()
    rows = []
    for sid, (years, L) in panels.items():
        years = np.asarray(years)
        T = len(years)
        full_mask = np.ones(T, bool)
        covid_mask = np.isin(years, COVID)
        n_drop = int(covid_mask.sum())
        MCS_full = mcs_size(L, full_mask)
        MCS_ex_covid = mcs_size(L, ~covid_mask) if n_drop < T else MCS_full
        placebo = []
        if n_drop > 0:
            eligible = np.flatnonzero(~covid_mask)
            if len(eligible) < n_drop:
                print(f"  [warn] {sid}: only {len(eligible)} non-COVID year(s) available, fewer "
                      f"than the {n_drop} COVID year(s) being simulated away -- placebo draws will "
                      f"be undersized and not truly size-matched to the real deletion", flush=True)
            rng = np.random.default_rng(0)
            for _ in range(N_PLACEBO):
                drop = rng.choice(eligible, min(n_drop, len(eligible)), replace=False)
                m = np.ones(T, bool)
                m[drop] = False
                placebo.append(mcs_size(L, m))
        placebo = np.asarray(placebo, float)
        rows.append({
            "series": sid, "T": T, "n_covid_years": n_drop,
            "MCS_full": MCS_full, "MCS_ex_covid": MCS_ex_covid,
            "MCS_placebo_mean": float(placebo.mean()) if len(placebo) else np.nan,
            "placebo_pctile_of_excovid": float(((placebo < MCS_ex_covid).mean()
                                                 + 0.5 * (placebo == MCS_ex_covid).mean()) * 100)
            if len(placebo) else np.nan,
        })
    R = pd.DataFrame(rows).sort_values("series")
    _headline = pd.read_csv(MT.OUT).set_index("series")["MCS_abs_10"]
    _cmp = R.set_index("series")["MCS_full"].reindex(_headline.index)
    if not (_cmp == _headline).all():
        raise RuntimeError(
            f"regime_robustness.py's MCS_full disagrees with mcs_test.csv's MCS_abs_10 for "
            f"{(_cmp != _headline).sum()} series -- same full curated panel, same estimator, should be "
            f"identical; a divergence here means the new mcs_size()/per_series_panels() plumbing is "
            f"computing something different from the paper's headline estimator")
    _csv_out = RES / "regime_robustness.csv"
    RM._atomic_to_csv(R, _csv_out)
    n = len(R)
    n_full = int((R.MCS_full > 1).sum())
    n_excovid = int((R.MCS_ex_covid > 1).sum())
    placebo_pct = float((R.MCS_placebo_mean > 1).mean() * 100)
    print(f"=== IS THE NON-IDENTIFICATION JUST COVID? ({n} curated series) ===")
    print(R.to_string(index=False))
    print(f"\n  non-identified (|MCS|>1)  FULL sample: {n_full}/{n}")
    print(f"  non-identified (|MCS|>1)  EX-COVID   : {n_excovid}/{n}")
    print(f"  series whose MEAN placebo |MCS| exceeds 1: {placebo_pct:.1f}% ({N_PLACEBO} draws/series, "
          f"same NUMBER of years dropped as COVID, drawn from non-COVID years)")
    print(f"  median across series of the (tie-corrected) percentile of the ex-COVID |MCS| within its "
          f"own placebo distribution: {R.placebo_pctile_of_excovid.median():.1f}%")
    print("  READ: if ex-COVID and FULL agree, and the placebo distribution makes ex-COVID look "
          "TYPICAL of dropping any equal-sized set of non-COVID years (not an outlier), the "
          "non-identification finding does not depend on the pandemic being in the sample.")
    emit("regime.nonidentified_full", n_full, of=n,
         spec=f"series with |MCS|>1 on the full sample; absolute loss, alpha={ALPHA}",
         cites=["S4.1 body: COVID-regime robustness check"],
         cross_check="regime_robustness.csv: (MCS_full > 1).sum()")
    emit("regime.nonidentified_excovid", n_excovid, of=n,
         spec=f"series with |MCS|>1 after deleting each series' own COVID years ({COVID[0]}-{COVID[1]}); "
              f"alpha={ALPHA}",
         cites=["S4.1 body: COVID-regime robustness check"],
         cross_check="regime_robustness.csv: (MCS_ex_covid > 1).sum()")
    emit("regime.placebo_nonidentified_pct", placebo_pct, n_series=n,
         spec=f"share of series whose MEAN placebo |MCS| exceeds 1; placebo drops the same NUMBER of "
              f"non-COVID years as each series' own COVID years, {N_PLACEBO} draws per series, "
              f"isolating the sample-size effect from the COVID effect",
         cites=["S13.3: COVID-regime robustness check"],  # the full placebo derivation is in
         # Appendix B (S13.3); S4.1 keeps only the headline movement and the 3-series exception.
         cross_check="regime_robustness.csv: (MCS_placebo_mean > 1).mean() * 100")
    _moved = R[R.MCS_ex_covid < R.MCS_full]
    _extreme_set = set(R[R.placebo_pctile_of_excovid <= 2].series)
    _moved_set = set(_moved.series)
    _rel = ("are exactly the same series" if _moved_set == _extreme_set else
            f"overlap on {len(_moved_set & _extreme_set)} of {len(_moved_set)} (not the full set -- "
            f"{sorted(_moved_set - _extreme_set)} fall without sitting at an extreme placebo percentile)")
    emit("regime.series_mcs_fell_excovid", int(len(_moved)), of=n,
         states=";".join(sorted(_moved.series)),
         spec=f"curated series whose |MCS| strictly DECREASED when their own COVID years were "
              f"removed. Read beside regime.placebo_extreme_series: the two sets {_rel}.",
         cites=["S4.1 body: COVID-regime robustness check"],
         cross_check="regime_robustness.csv: (MCS_ex_covid < MCS_full).sum()")
    emit("regime.placebo_extreme_series", int((R.placebo_pctile_of_excovid <= 2).sum()), of=n,
         states=";".join(sorted(R[R.placebo_pctile_of_excovid <= 2].series)),
         spec="curated series whose ex-COVID |MCS| sits at or below the 2nd percentile of their own "
              "200-draw placebo distribution -- for these the COVID removal is more extreme than "
              "essentially any equal-sized random year-drop, the pattern the aggregate median hides",
         cites=["S4.1 body: COVID-regime robustness check"],
         cross_check="regime_robustness.csv: (placebo_pctile_of_excovid <= 2).sum()")
    emit("regime.placebo_below_excovid_median", float(R.placebo_pctile_of_excovid.median()), n_series=n,
         spec=f"median across series of the tie-corrected (mid-rank) percentile of the ex-COVID |MCS| "
              f"within its own {N_PLACEBO}-draw placebo distribution; 50 means the ex-COVID result is "
              f"typical of dropping an equal-sized set of non-COVID years -- drawn at random, NOT a "
              f"contiguous block. |MCS| is a small integer and placebo distributions are often heavily "
              f"tied, so a strict less-than count would misread a degenerate, perfectly-typical case "
              f"(100% of draws equal to the ex-COVID value) as maximally extreme rather than as 50",
         cites=["S13.3: COVID-regime robustness check"],
         cross_check="regime_robustness.csv: placebo_pctile_of_excovid.median()")
    print(f"\n  wrote {RES/'regime_robustness.csv'}")
    flush()


if __name__ == "__main__":
    main()

"""flagship_null_ar1.py — applies election_null_ar1.py's AR(1) recalibration to the flagship
exhibit's OWN calibrated null, not just the curated 24-cell panel.
"""
import sys, warnings, json
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "code" / "instrument"))
from election_null_ar1 import stable_benchmark_ar1
from canonical_emit import emit, flush

RES = ROOT / "results"
        # NOT multiseries/results -- the flagship tier's own files live here, matching
        # flagship_fragility.py's own RES constant.
MRES = ROOT / "multiseries" / "results"
SB_CSV = RES / "extension_scoreboard_1988-2025.csv"


def _lag1(x):
    x = x - x.mean()
    if len(x) < 5 or x.std() == 0:
        return None
    den = float(np.sum(x[:-1] ** 2))
    if den <= 0:
        return None
    return float(np.sum(x[1:] * x[:-1]) / den)


def _finite_columns(sb):
    cols = []
    for m in sb.columns:
        x = sb[m].to_numpy(float)
        if not np.all(np.isfinite(x)):
            print(f"  [warn] {m}: NaN/inf in the flagship scoreboard -- skipped from the AR(1) estimate")
            continue
        cols.append(m)
    return cols


def estimate_flagship_ar1_levels(sb):
    """Lag-1 OLS autocorrelation of each model's demeaned annual-loss LEVEL series in the
    flagship scoreboard -- a superseded calibration basis, retained for disclosure only."""
    rows = []
    for m in _finite_columns(sb):
        phi = _lag1(sb[m].to_numpy(float))
        if phi is not None:
            rows.append({"model": m, "ar1_estimate": phi})
    return pd.DataFrame(rows)


def estimate_flagship_ar1(sb):
    """Lag-1 OLS autocorrelation of every pairwise annual-loss DIFFERENTIAL in the flagship
    scoreboard (all K(K-1)/2 pairs), demeaned."""
    rows = []
    cols = _finite_columns(sb)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            phi = _lag1(sb[a].to_numpy(float) - sb[b].to_numpy(float))
            if phi is not None:
                rows.append({"pair": f"{a}-{b}", "ar1_estimate": phi})
    return pd.DataFrame(rows)


def main():
    sb = pd.read_csv(SB_CSV, index_col=0)
    _K, _T = sb.shape[1], sb.shape[0]
    coeffs_df = estimate_flagship_ar1(sb)              # pairwise DIFFERENTIALS: the calibration basis
    levels_df = estimate_flagship_ar1_levels(sb)       # LEVELS: a superseded basis, disclosure only
    ar_hat = float(coeffs_df["ar1_estimate"].median())
    ar_levels = float(levels_df["ar1_estimate"].median())
    print(f"=== FLAGSHIP EXHIBIT NULL UNDER EMPIRICAL AR(1) (K={_K}, T={_T}) ===")
    print(f"  AR(1) estimates on pairwise loss differentials: n={len(coeffs_df)}  median={ar_hat:.4f}  "
          f"pairs={dict(zip(coeffs_df.pair, coeffs_df.ar1_estimate.round(4)))}")
    print(f"  (loss LEVELS, the superseded basis, for disclosure: n={len(levels_df)}  median={ar_levels:.4f}  "
          f"models={dict(zip(levels_df.model, levels_df.ar1_estimate.round(4)))})")

    nbd_ar1 = stable_benchmark_ar1(_K, _T, ar_hat)
    print(f"  AR(1)-adjusted null: median={nbd_ar1['median']:.4f}  p05={nbd_ar1['p05']:.4f}  "
          f"(band [{nbd_ar1['p05_lo']:.4f}, {nbd_ar1['p05_hi']:.4f}], sd={nbd_ar1['p05_sd']:.4f})")

    _ff_frag = json.loads((MRES / "canonical" / "flagship_fragility.json").read_text())
    p05_iid = float(_ff_frag["flagship.stability_null_p05"]["value"])
    obs = float(_ff_frag["flagship.winner_stability"]["value"])
    delta_p05 = nbd_ar1["p05"] - p05_iid
    print(f"  rise in the null's fifth percentile from i.i.d. ({p05_iid:.3f}) to AR(1)-adjusted: {delta_p05:+.4f}")
    print(f"  observed winner_stability={obs:.3f}  vs i.i.d. p05={p05_iid:.3f}  vs AR(1)-adjusted p05={nbd_ar1['p05']:.4f}")
    above_ar1 = obs >= nbd_ar1["p05"]
    print(f"  still comfortably above the AR(1)-adjusted noise floor: {above_ar1}")

    pd.concat([coeffs_df.assign(basis="differential"), levels_df.assign(basis="level")],
              ignore_index=True).to_csv(RES / "flagship_null_ar1_coefficients.csv", index=False)
    pd.DataFrame([{
        "K": _K, "T": _T, "ar_hat": ar_hat,
        "null_median": nbd_ar1["median"], "null_p05": nbd_ar1["p05"],
        "null_p05_lo": nbd_ar1["p05_lo"], "null_p05_hi": nbd_ar1["p05_hi"], "null_p05_sd": nbd_ar1["p05_sd"],
        "p05_iid": p05_iid, "delta_p05": delta_p05, "observed": obs, "above_ar1_p05": above_ar1,
    }]).to_csv(RES / "flagship_null_ar1.csv", index=False)

    emit("flagship.ar1_coefficient_empirical", round(ar_hat, 4), n=int(len(coeffs_df)),
         levels_median=round(ar_levels, 4), n_levels=int(len(levels_df)),
         spec="pooled median lag-1 OLS autocorrelation, demeaned, across every pairwise annual-loss "
              "DIFFERENTIAL of the flagship exhibit's own 7 models (21 pairs) -- the empirical AR(1) "
              "coefficient used to recalibrate the flagship's own exchangeable null, mirroring "
              "election.ar1_coefficient_empirical (same differential basis, same reasoning) but "
              "estimated from the flagship's own loss data. levels_median is the same estimator on "
              "the 7 loss LEVEL series, a superseded basis, disclosed but not used",
         cites=["S3.6 body: flagship exhibit AR(1)-adjusted null"],
         cross_check="flagship_null_ar1_coefficients.csv: median(ar1_estimate) where basis=='differential'")
    emit("flagship.stability_null_median_ar1", round(nbd_ar1["median"], 3), n_models=_K, n_periods=_T,
         n_eff=nbd_ar1["n_eff"],
         spec=f"median winner_stability under the exchangeable null at the flagship's own K={_K}, "
              f"T={_T}, with the empirically-estimated AR(1) dependence above applied to every "
              f"model's series, instead of i.i.d. draws -- the flagship-exhibit analogue of "
              f"election.stability_null_median_ar1",
         cites=["S3.6 body: flagship exhibit AR(1)-adjusted null"],
         cross_check="flagship_null_ar1.csv: null_median")
    emit("flagship.stability_null_p05_ar1", round(nbd_ar1["p05"], 3), n_models=_K, n_periods=_T,
         band=[round(nbd_ar1["p05_lo"], 3), round(nbd_ar1["p05_hi"], 3)], sd=round(nbd_ar1["p05_sd"], 4),
         n_eff=nbd_ar1["n_eff"],
         spec=f"fifth percentile of winner_stability under the exchangeable null at the flagship's "
              f"own K={_K}, T={_T}, with the empirically-estimated AR(1) dependence above applied "
              f"to every model's series",
         cites=["S3.6 body: flagship exhibit AR(1)-adjusted null"],
         cross_check="flagship_null_ar1.csv: null_p05")
    emit("flagship.ar1_delta_p05", round(delta_p05, 4), p05_iid=round(p05_iid, 3),
         spec="rise in the flagship's own null fifth percentile from the i.i.d. exchangeable null "
              "(flagship.stability_null_p05) to this stage's AR(1)-adjusted null",
         cites=["S3.6 body: flagship exhibit AR(1)-adjusted null"],
         cross_check="flagship_null_ar1.csv: delta_p05")
    emit("flagship.above_ar1_adjusted_p05", bool(above_ar1), observed=round(obs, 3),
         threshold=round(nbd_ar1["p05"], 3),
         spec=f"whether the flagship's observed winner_stability ({obs:.3f}) still clears the "
              f"AR(1)-adjusted noise floor, not just the i.i.d. one it was originally read against",
         cites=["S3.6 body: flagship exhibit AR(1)-adjusted null"],
         cross_check="flagship_null_ar1.csv: above_ar1_p05")
    flush()


if __name__ == "__main__":
    main()

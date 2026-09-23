"""m4_perseries_kstar.py — per-series k* and winner_stability on the REAL M4 competition submissions
(not a proxy), completing the instrument run Section 9 explicitly disclosed as not yet done.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "instrument"))
from m4_submissions_audit import DATA, load_data, parse_submission, score
from fragility import decision_breakdown, pooled_winner, winner_stability
from canonical_emit import emit, flush
RES = HERE.parent.parent / "multiseries" / "results"
N_BOOT = 2000
       # matches winner_stability's own function default, used explicitly (not left implicit) so a
       # reader of this file sees the same number the emitted spec string states


def local_submissions():
    """Already-extracted submission folders on disk -- no network, no re-download."""
    return sorted(p.name for p in DATA.glob("submission-*") if p.is_dir())


def build_matrix():
    truth = load_data()
    subs = local_submissions()
    print(f"{len(subs)} locally-extracted submissions")
    per_sub_mase = {}
    for name in subs:
        csvs = list((DATA / name).rglob("*.csv"))
        fc = parse_submission(csvs)
        row = {}
        for sid, yp in fc.items():
            if sid in truth:
                yt, m, sa, ssq = truth[sid]
                sc = score(yp, yt, sa, ssq)
                if sc:
                    row[sid] = sc["MASE"]
        if row:
            per_sub_mase[name] = pd.Series(row)
            print(f"  {name}: scored {len(row)} series", flush=True)
    if len(per_sub_mase) != len(subs):
        missing = sorted(set(subs) - set(per_sub_mase))
        raise RuntimeError(
            f"{len(missing)} of {len(subs)} locally-extracted submissions produced no scored rows and "
            f"were silently dropped from the panel: {missing} -- a real gap here would shrink K below "
            f"what local_submissions() found on disk with no error, and either way the panel below must "
            f"not silently score a different submission set without saying so.")
    return pd.DataFrame(per_sub_mase).T   # submissions x series, NaN where a submission missed a series


def main():
    M = build_matrix()
    n_missing = int(M.isna().sum().sum())
    if n_missing:
        raise RuntimeError(
            f"{n_missing} missing (submission, series) MASE values -- build_matrix() previously "
            f"documented every submission as covering all scored series; a real gap here would mean "
            f"either a scoring failure or a genuinely partial submission, and either way the panel "
            f"below must not silently drop to a different set of periods per model without saying so.")
    K, T = M.shape
    L = {name: M.loc[name].to_numpy(float) for name in M.index}
    w = np.ones(T)
    champion = pooled_winner(L, w)
    k_star, binding_opp, _ = decision_breakdown(L, w, a=champion)
    stability, _ = winner_stability(L, w, n_boot=N_BOOT, block=1, seed=0)
    print(f"=== M4 REAL PER-SERIES k*/winner_stability ({K} submissions x {T} series) ===")
    print(f"  pooled champion (lowest mean MASE): {champion}")
    print(f"  k* = {k_star} of {T} series (fraction {k_star / T:.6f}); binding opponent: {binding_opp}")
    print(f"  winner_stability (n_boot={N_BOOT}, block=1 i.i.d. over series): {stability:.4f}")
    pd.DataFrame([{
        "K": K, "T": T, "champion": champion, "k_star": k_star, "k_star_frac": k_star / T,
        "binding_opponent": binding_opp, "winner_stability": stability, "n_boot": N_BOOT,
    }]).to_csv(RES / "m4_perseries_kstar.csv", index=False)

    emit("m4sub.k_star", int(k_star), of=int(T), frac=round(k_star / T, 6),
         binding_opponent=binding_opp,
         spec=f"decision_breakdown()'s k* for the pooled champion ({champion}) against all "
              f"{K - 1} rival submissions, on the real per-series MASE matrix -- the fewest series "
              f"whose removal unseats the champion. `frac` is k*/T with T = series here, not years "
              f"as on the time-series panels S2 reports k* on elsewhere",
         cites=["S9 body: M4 instrument run"],
         cross_check="m4_perseries_kstar.csv: k_star")
    emit("m4sub.winner_stability", round(float(stability), 4), n_boot=N_BOOT, block=1,
         spec=f"winner_stability on the real M4 panel ({K} submissions x {T} series), block=1 "
              f"(i.i.d. resampling over series: the M4 series are a cross-section with no "
              f"temporal adjacency, so a block bootstrap has nothing to preserve), "
              f"n_boot={N_BOOT}. Descriptive only: no calibrated null is "
              f"computed at this (K, T)",
         cites=["S9 body: M4 instrument run"],
         cross_check="m4_perseries_kstar.csv: winner_stability")
    flush()


if __name__ == "__main__":
    main()

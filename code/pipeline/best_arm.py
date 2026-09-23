"""
best_arm.py — verify the best-arm-identification reframe of k* on existing panels (re-score only).
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
ROOT  = HERE.parent.parent                 # <repo root>/code/<subdir>/ -> <repo root>
MULTI = ROOT / "multiseries"               # data + results live HERE, not under code/
INSTR = ROOT / "code" / "instrument"       # shared estimator

PRED = MULTI / "results" / "breadth_distinct" / "breadth_distinct_predictions.csv"
MCSF = MULTI / "results" / "mcs_test.csv"

sys.path.insert(0, str(INSTR))
from canonical_emit import emit, flush
import run_multiseries as RM   # for its shared _atomic_to_csv() helper

_SE_REL_FLOOR = 1e-12


def studentized_gap(li, lbest):
    """Studentized suboptimality gap (paired t-statistic) of arm `li` vs the best arm `lbest`,
    both length-T arrays of per-year losses."""
    dd = np.asarray(li, float) - np.asarray(lbest, float)
    T = len(dd)
    se = dd.std(ddof=1) / np.sqrt(T) if T > 1 else 0.0
    pair_scale = float(np.mean(np.abs(li)) + np.mean(np.abs(lbest))) / 2.0
    floor = _SE_REL_FLOOR * pair_scale
    if se > floor:
        return float(dd.mean() / se)
    if abs(dd.mean()) <= floor:
        return 0.0
    return np.inf


def main():
    d = pd.read_csv(PRED); mcs = pd.read_csv(MCSF).set_index("series")
    rows = []
    for sid, g in d.groupby("series"):
        models = sorted(g.model.unique()); years = sorted(g.year.unique())
        L = np.array([[g[(g.model == m) & (g.year == y)]["sae"].mean() for m in models] for y in years])  # T x K
        keep = ~np.isnan(L).any(1)
        if not keep.all():
            print(f"  [warn] {sid}: dropping {int((~keep).sum())} of {len(years)} years with a "
                  f"missing model-year cell", flush=True)
        if not keep.any():
            raise RuntimeError(
                f"{sid}: every year has a missing model-year cell -- no complete year survives to "
                f"compute a loss matrix from")
        years, L = [y for y, k in zip(years, keep) if k], L[keep]
        T = len(years)
        mean = L.mean(0); order = np.argsort(mean); best = order[0]
        # studentized suboptimality gap of each arm vs the best (paired, per-year)
        tstat = {}
        for i in range(len(models)):
            if i == best: continue
            tstat[i] = studentized_gap(L[:, i], L[:, best])
        gap2 = min(tstat.values())                         # studentized best-vs-2nd gap (small => near-tie)
        H = float(np.sum([1.0 / max(t, 1e-6) ** 2 for t in tstat.values()]))   # fixed-confidence complexity proxy
        rows.append({"series": sid, "gap2_t": gap2, "H": H, "T": T,
                     "MCS": int(mcs.loc[sid, "MCS_abs_10"]) if sid in mcs.index else np.nan,
                     "sci": mcs.loc[sid, "sci"] if "sci" in mcs.columns and sid in mcs.index else np.nan})
    r = pd.DataFrame(rows).sort_values("gap2_t")
    print("=== BEST-ARM VERIFICATION (does gap-structure, not shock-proneness, govern identifiability?) ===")
    print(r.assign(**{"gap2_t": r.gap2_t.round(2), "H": r.H.round(1)}).to_string(index=False))
    n = len(r)
    RM._atomic_to_csv(r, MULTI / "results" / "best_arm.csv")
    def sp(a, b):
        rho, p = stats.spearmanr(r[a], r[b], nan_policy="omit"); return rho, p
    print("\n  === HONEST reporting: a CONCEPTUAL reframe, NOT a demonstrated law ===")
    gap_rho, gap_p = sp("gap2_t", "MCS")
    H_rho, H_p = sp("H", "MCS")
    for (rho, p), lab in [((gap_rho, gap_p), "gap (best-vs-2nd)"), ((H_rho, H_p), "complexity H")]:
        print(f"  Spearman({lab:18s}, |MCS|) = {rho:+.2f}  (p={p:.2f}, n={n})")
    gapH_rho, _ = sp("gap2_t", "H")
    print(f"  NOTE: gap and H are the SAME quantity (Spearman = {gapH_rho:+.2f}) -> report ONE relationship, not two.")
    if "PI" in set(r.series):
        _ident = sorted(r.loc[r.MCS == 1, "series"])
        assert _ident == ["PI"], (
            f"identified-series set changed: {_ident} -- the excl-PI leverage check below is no "
            f"longer 'drop the identified series'; update the emit specs and this exclusion together"
        )
    excl = r[r.series != "PI"]
    ex_gap_rho, ex_gap_p = stats.spearmanr(excl["gap2_t"], excl["MCS"], nan_policy="omit")
    ex_H_rho, ex_H_p = stats.spearmanr(excl["H"], excl["MCS"], nan_policy="omit")
    gap_drop_pct = round(100 * (abs(gap_rho) - abs(ex_gap_rho)) / abs(gap_rho))
    H_drop_pct = round(100 * (abs(H_rho) - abs(ex_H_rho)) / abs(H_rho))
    print(f"  excl-PI: Spearman(gap, |MCS|) = {ex_gap_rho:+.4f} ({gap_drop_pct}% relative drop); "
          f"Spearman(H, |MCS|) = {ex_H_rho:+.4f} ({H_drop_pct}% relative drop)")
    sci_rho = sci_p = np.nan
    if r["sci"].notna().any():
        sci_rho, sci_p = sp("sci", "MCS")
        print(f"  Spearman(shock-proneness, |MCS|) = {sci_rho:+.2f} (p={sci_p:.2f}) -- the well-powered NULL, for contrast.")
    print("\n  CAVEATS TO DISCLOSE: (1) |MCS| and gap/H are both functions of the SAME per-year loss panel -> the")
    print(f"  correlation is largely DEFINITIONAL; (2) at n={n} none reaches p<0.05; (3) {gap_drop_pct}-{H_drop_pct}% rides on the")
    print("  one identified series PI. This is NOT a demonstrated law. The genuine non-circular content is the")
    print("  H-vs-EXOGENOUS-T framing (identifiability = gap-structure x sample size), which EXPLAINS the well-powered")
    print("  shock-proneness null as a best-arm FEATURE. Cite the established fixed-confidence bounds")
    print("  (Kaufmann-Cappe-Garivier; Jamieson-Nowak) as the PORT; do NOT claim a new theorem.")

    emit("bestarm.spearman_gap_mcs", float(gap_rho), n=n,
         spec="Spearman rho between the studentized best-vs-2nd gap (gap2_t) and |MCS|, n=12 curated series",
         cites=["S4.3: the gap and |MCS| correlate at Spearman"],
         cross_check="best_arm.csv: spearmanr(gap2_t, MCS)")
    emit("bestarm.spearman_gap_mcs_p", float(gap_p), n=n,
         spec="p-value for bestarm.spearman_gap_mcs",
         cites=["S4.3: the gap and |MCS| correlate at Spearman"],
         cross_check="best_arm.csv: spearmanr(gap2_t, MCS) p-value")
    emit("bestarm.spearman_H_mcs", float(H_rho), n=n,
         spec="Spearman rho between the fixed-confidence complexity proxy H and |MCS|, n=12 curated series",
         cites=["S4.3: the complexity proxy H and |MCS| correlate at"],
         cross_check="best_arm.csv: spearmanr(H, MCS)")
    emit("bestarm.spearman_H_mcs_p", float(H_p), n=n,
         spec="p-value for bestarm.spearman_H_mcs",
         cites=["S4.3: the complexity proxy H and |MCS| correlate at"],
         cross_check="best_arm.csv: spearmanr(H, MCS) p-value")
    emit("bestarm.spearman_gap_H", float(gapH_rho), n=n,
         spec="Spearman rho between gap2_t and H -- near-identical quantities by construction, reported to "
              "show the two correlations above are one relationship, not two independent confirmations",
         cites=["S4.3: gap and H are near-identical quantities"],
         cross_check="best_arm.csv: spearmanr(gap2_t, H)")
    if not np.isnan(sci_rho):
        emit("bestarm.spearman_sci_mcs", float(sci_rho), n=n,
             spec="Spearman rho between shock-proneness (sci) and |MCS| -- the well-powered null from S4.2, "
                  "shown for contrast against the gap/H correlations",
             cites=["S4.3: compared to ... for shock-proneness, the well-powered null of Section 4.2"],
             cross_check="best_arm.csv: spearmanr(sci, MCS)")
        emit("bestarm.spearman_sci_mcs_p", float(sci_p), n=n,
             spec="p-value for bestarm.spearman_sci_mcs",
             cites=["S4.3: compared to ... for shock-proneness, the well-powered null of Section 4.2"],
             cross_check="best_arm.csv: spearmanr(sci, MCS) p-value")
    emit("bestarm.spearman_gap_mcs_excl_pi", float(ex_gap_rho), n=n - 1,
         spec="bestarm.spearman_gap_mcs recomputed excluding Personal Income (PI), the single identified "
              "series that carries most of the correlation's leverage",
         cites=["S4.3: excluding it, the gap-|MCS| correlation collapses"],
         cross_check="best_arm.csv: spearmanr(gap2_t, MCS) on series != 'PI'")
    emit("bestarm.spearman_gap_mcs_excl_pi_p", float(ex_gap_p), n=n - 1,
         spec="p-value for bestarm.spearman_gap_mcs_excl_pi",
         cites=["S4.3: excluding it, the gap-|MCS| correlation collapses"],
         cross_check="best_arm.csv: spearmanr(gap2_t, MCS) p-value on series != 'PI'")
    emit("bestarm.spearman_gap_mcs_excl_pi_drop_pct", gap_drop_pct,
         spec="relative drop in |Spearman rho| from bestarm.spearman_gap_mcs to its excl-PI counterpart",
         cites=["S4.3: the excl-PI relative drop"],
         cross_check="round(100 * (abs(gap_rho) - abs(ex_gap_rho)) / abs(gap_rho))")
    emit("bestarm.spearman_H_mcs_excl_pi", float(ex_H_rho), n=n - 1,
         spec="bestarm.spearman_H_mcs recomputed excluding Personal Income (PI)",
         cites=["S4.3: the H-|MCS| correlation from"],
         cross_check="best_arm.csv: spearmanr(H, MCS) on series != 'PI'")
    emit("bestarm.spearman_H_mcs_excl_pi_p", float(ex_H_p), n=n - 1,
         spec="p-value for bestarm.spearman_H_mcs_excl_pi",
         cites=["S4.3: the H-|MCS| correlation from"],
         cross_check="best_arm.csv: spearmanr(H, MCS) p-value on series != 'PI'")
    emit("bestarm.spearman_H_mcs_excl_pi_drop_pct", H_drop_pct,
         spec="relative drop in |Spearman rho| from bestarm.spearman_H_mcs to its excl-PI counterpart",
         cites=["S4.3: the excl-PI relative drop"],
         cross_check="round(100 * (abs(H_rho) - abs(ex_H_rho)) / abs(H_rho))")
    flush()


if __name__ == "__main__":
    main()

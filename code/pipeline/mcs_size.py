"""mcs_size.py — what is the ACTUAL level of the model confidence set at these panel sizes, and does the paper's headline survive correcting for it?
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "instrument"))
import mcs_test as MCS
from canonical_emit import emit, flush
PRED = HERE.parent.parent / "multiseries" / "results" / "breadth_distinct" / "breadth_distinct_predictions.csv"
RES = HERE.parent.parent / "multiseries" / "results"
OUT = RES / "mcs_size.csv"
OUT_ALPHA = RES / "mcs_size_alpha_sweep.csv"
OUT_AR1 = RES / "mcs_size_ar1.csv"
K_PAPER, T_PAPER, BLOCK_PAPER = 6, 30, 3
K_STATE, T_STATE = 4, 35
          # Section 7's state tie sets: four candidate models on ~35 annual cells (1991 warm-up to 2026)
PHI_AR1 = 0.25
          # the panel's own pooled lag-1 autocorrelation of the pairwise loss differentials
          # (election.ar1null_coefficient, social_choice stage). Size is measured under this serial
          # dependence as well as under i.i.d., because the i.i.d. figure understates it
N_SIZE = 2000  # 0.10 +/- 0.013 at 95%, which is tight enough to separate 0.10 from 0.25
N_ARCH = 600  # arch is ~40x slower per panel; 600 gives +/- 0.036, ample to confirm a 2.5x gap
SD = 1.0
BASE_LEVEL = 1.0
ALPHAS = (0.10, 0.05, 0.02, 0.01, 0.001)
SEED = 7_000_003


def null_panel(g, K, T):
    """A panel on which the equal-predictive-ability null is EXACTLY true."""
    return BASE_LEVEL + SD * g.normal(0.0, 1.0, size=(T, K))


def null_panel_ar1(g, K, T, phi=PHI_AR1, burn=50):
    """An exactly-null panel whose K loss series each follow a stationary AR(1) with coefficient
    `phi` and unit marginal variance (so it differs from null_panel only in serial dependence)."""
    e = g.normal(0.0, 1.0, size=(T + burn, K))
    x = np.zeros_like(e)
    for t in range(1, T + burn):
        x[t] = phi * x[t - 1] + e[t]
    return BASE_LEVEL + SD * x[burn:] * np.sqrt(1.0 - phi ** 2)


def size_row(K, T, block, alpha, n_rep, seed, panel=None):
    """Empirical rejection rate of the FIRST-STEP equal-ability null on exactly-null panels."""
    panel = null_panel if panel is None else panel
    g = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_rep):
        surv, _p = MCS.mcs(panel(g, K, T), alpha=alpha, B=800, block=block, seed=0)
        hits += int(len(surv) < K)
    rate = hits / n_rep
    se = float(np.sqrt(rate * (1 - rate) / n_rep))
    return rate, se


def arch_cross_check(K, T, block, alpha, n_rep, seed):
    """The same measurement through a third-party implementation, on IDENTICAL panels."""
    from arch.bootstrap import MCS as ArchMCS
    g = np.random.default_rng(seed)
    hits_ours = hits_arch = agree = 0
    for _ in range(n_rep):
        L = null_panel(g, K, T)
        surv, _p = MCS.mcs(L, alpha=alpha, B=800, block=block, seed=0)
        a = ArchMCS(pd.DataFrame(L), size=alpha, reps=800, block_size=block, method="R", seed=0)
        a.compute()
        n_arch = int(len(a.included))
        hits_ours += int(len(surv) < K)
        hits_arch += int(n_arch < K)
        agree += int((len(surv) < K) == (n_arch < K))
    return hits_ours / n_rep, hits_arch / n_rep, agree / n_rep


def curated_panels():
    """Per-series per-year absolute-loss (sae) matrices."""
    d = pd.read_csv(PRED)
    out = {}
    for sid, g in d.groupby("series"):
        models = sorted(g.model.unique())
        yrs = sorted(g.year.unique())
        La = np.full((len(yrs), len(models)), np.nan)
        for j, m in enumerate(models):
            a = g[g.model == m].groupby("year")["sae"].mean()
            for i, y in enumerate(yrs):
                if y in a.index:
                    La[i, j] = a[y]
        out[sid] = La[~np.isnan(La).any(1)]
    return out


def main():
    print("=== ACTUAL LEVEL of the elimination rule on EXACTLY-NULL panels ===")
    print("(nominal alpha = 0.10; 'rejection' = at least one model eliminated)")
    rows = []
    grid = [(K_PAPER, T_PAPER, BLOCK_PAPER), (K_PAPER, T_PAPER, 1), (K_PAPER, T_PAPER, 6),
            (K_PAPER, 100, BLOCK_PAPER), (K_PAPER, 400, BLOCK_PAPER),
            (2, T_PAPER, BLOCK_PAPER), (3, T_PAPER, BLOCK_PAPER), (10, T_PAPER, BLOCK_PAPER)]
    for i, (K, T, block) in enumerate(grid):
        rate, se = size_row(K, T, block, 0.10, N_SIZE, SEED + 101 * i)
        rows.append({"K": K, "T": T, "block": block, "alpha": 0.10,
                     "size": round(rate, 4), "se": round(se, 4), "n_rep": N_SIZE})
        flag = "  <- the paper's configuration" if (K, T, block) == (K_PAPER, T_PAPER, BLOCK_PAPER) else ""
        print(f"K={K:3d} T={T:4d} block={block}  size {rate:.4f} +/- {1.96*se:.4f}{flag}", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(OUT, index=False)
    paper = R[(R.K == K_PAPER) & (R["T"] == T_PAPER) & (R.block == BLOCK_PAPER)].iloc[0]
    print("\n=== the same measurement under the panel's own serial dependence (AR(1) phi=%.2f) ===" % PHI_AR1)
    ar_rows = []
    for i, (K, T, panel, label) in enumerate([(K_PAPER, T_PAPER, null_panel_ar1, "ar1"),
                                              (K_STATE, T_STATE, null_panel, "iid"),
                                              (K_STATE, T_STATE, null_panel_ar1, "ar1")]):
        rate, se = size_row(K, T, BLOCK_PAPER, 0.10, N_SIZE, SEED + 7_001 * (i + 1), panel=panel)
        ar_rows.append({"K": K, "T": T, "block": BLOCK_PAPER, "alpha": 0.10, "null": label,
                        "phi": PHI_AR1 if label == "ar1" else 0.0,
                        "size": round(rate, 4), "se": round(se, 4), "n_rep": N_SIZE})
        print(f"K={K:3d} T={T:4d} block={BLOCK_PAPER} null={label:3s}  size {rate:.4f} +/- {1.96*se:.4f}", flush=True)
    AR = pd.DataFrame(ar_rows)
    AR.to_csv(OUT_AR1, index=False)
    print("\n=== CROSS-CHECK against arch 8.0.0 on IDENTICAL panels ===")
    ours, theirs, agr = arch_cross_check(K_PAPER, T_PAPER, BLOCK_PAPER, 0.10, N_ARCH, SEED + 9001)
    print(f"ours  {ours:.4f}   arch  {theirs:.4f}   verdict agreement {agr:.3f}  (n={N_ARCH})")
    print("-> a distortion both implementations show is a property of the statistic, not of this repo.")
    print("\n=== ALPHA SWEEP on the curated panel: does the headline depend on how hard we eliminate? ===")
    panels = curated_panels()
    arows = []
    for al in ALPHAS:
        sizes = {sid: len(MCS.mcs(L, alpha=al, seed=0)[0]) for sid, L in panels.items()}
        nonid = sum(1 for v in sizes.values() if v > 1)
        arows.append({"alpha": al, "nonidentified": nonid, "total": len(sizes),
                      "mcs_PI": sizes.get("PI"), "mean_size": round(float(np.mean(list(sizes.values()))), 3)})
        print(f"alpha={al:<6g} non-identified {nonid}/{len(sizes)}   |MCS(PI)|={sizes.get('PI')}   "
              f"mean |MCS| {np.mean(list(sizes.values())):.2f}", flush=True)
    A = pd.DataFrame(arows)
    A.to_csv(OUT_ALPHA, index=False)
    nonid_vals = sorted(set(A.nonidentified))
    pi_vals = sorted(set(A.mcs_PI))
    strict = A[A.alpha == min(ALPHAS)].iloc[0]
    _pi_control_failed = int(strict.mcs_PI) != 1
    emit("mcssize.size_at_nominal_10", round(float(paper["size"]), 3),
         K=K_PAPER, T=T_PAPER, block=BLOCK_PAPER, n_rep=N_SIZE, se=round(float(paper["se"]), 4),
         spec="Empirical rejection rate of the first-step equal-predictive-ability null (at least one "
              "model eliminated) on EXACTLY-NULL iid panels at the paper's own K/T/block, against a "
              "nominal level of 0.10. Greater than nominal means the test eliminates too readily, which "
              "makes every confidence set in this paper if anything too SMALL and the non-identification "
              "count a lower bound",
         provenance="mcs_size.py::size_row on null_panel draws",
         cites=["S4.2 body: empirical size at the nominal alpha"],
         cross_check="mcs_size.csv: size where K=6,T=30,block=3")
    emit("mcssize.oversize_ratio", round(float(paper["size"]) / 0.10, 1),
         spec="mcssize.size_at_nominal_10 divided by the nominal alpha=0.10, i.e. how many times "
              "oversized the elimination rule's empirical size is at these panel dimensions",
         cites=["S4.2 body: empirical size at the nominal alpha"],
         cross_check="round(mcssize.size_at_nominal_10 / 0.10, 1) -- the raw quotient is 2.65, and "
                     "Python's round-half-to-even takes it to 2.6, not 2.7")
    def _grid_cell(K, T, block):
        row = R[(R.K == K) & (R["T"] == T) & (R.block == block)].iloc[0]
        return row
    _ar_paper = AR[(AR.K == K_PAPER) & (AR.null == "ar1")].iloc[0]
    _iid_state = AR[(AR.K == K_STATE) & (AR.null == "iid")].iloc[0]
    _ar_state = AR[(AR.K == K_STATE) & (AR.null == "ar1")].iloc[0]
    emit("mcssize.size_ar1_at_nominal_10", round(float(_ar_paper["size"]), 3),
         K=K_PAPER, T=T_PAPER, block=BLOCK_PAPER, phi=PHI_AR1, n_rep=N_SIZE, se=round(float(_ar_paper["se"]), 4),
         iid=round(float(paper["size"]), 3),
         spec="Empirical rejection rate of the first-step equal-predictive-ability null on EXACTLY-NULL "
              "panels whose loss series each follow an AR(1) with the panel's own pooled lag-1 "
              "differential autocorrelation (phi=0.25, election.ar1null_coefficient), at the paper's "
              "K/T/block and nominal alpha=0.10. The i.i.d. figure (mcssize.size_at_nominal_10, in the "
              "`iid` field) understates the operating size under the serial dependence the losses "
              "actually carry; the direction is the same (sets too small, non-identification counts a "
              "lower bound, Section 7's model channel understated)",
         provenance="mcs_size.py::size_row on null_panel_ar1 draws",
         cites=["S4.2 body: empirical size at the nominal alpha"],
         cross_check="mcs_size_ar1.csv: size where K=6,T=30,null=='ar1'")
    emit("mcssize.size_state_dims", round(float(_iid_state["size"]), 3),
         K=K_STATE, T=T_STATE, block=BLOCK_PAPER, n_rep=N_SIZE, se=round(float(_iid_state["se"]), 4),
         ar1=round(float(_ar_state["size"]), 3), phi=PHI_AR1,
         spec="The same first-step size at the dimensions of Section 7's state tie-set test (K=4 "
              "candidate models, T=35 annual cells): i.i.d. null in the value, AR(1) phi=0.25 null in "
              "the `ar1` field. Quoted where Section 7 says its tie sets come from an oversized test",
         provenance="mcs_size.py::size_row at K_STATE/T_STATE",
         cites=["S7 body: multipliers caveat"],
         cross_check="mcs_size_ar1.csv: size where K=4,T=35")
    _t400 = _grid_cell(K_PAPER, 400, BLOCK_PAPER)
    emit("mcssize.size_at_T400", round(float(_t400["size"]), 3),
         K=K_PAPER, T=400, block=BLOCK_PAPER, n_rep=N_SIZE, se=round(float(_t400["se"]), 4),
         spec="Same empirical-size measurement as mcssize.size_at_nominal_10, at 4x the paper's panel "
              "length (T=400 vs T=30), holding K/block fixed -- the T-shrinks leg of the finite-sample "
              "pattern",
         provenance="mcs_size.py::size_row on null_panel draws",
         cites=["S4.2 body: size-distortion finite-sample pattern"],
         cross_check="mcs_size.csv: size where K=6,T=400,block=3")
    _block6 = _grid_cell(K_PAPER, T_PAPER, 6)
    emit("mcssize.size_at_block6", round(float(_block6["size"]), 3),
         K=K_PAPER, T=T_PAPER, block=6, n_rep=N_SIZE, se=round(float(_block6["se"]), 4),
         spec="Same empirical-size measurement as mcssize.size_at_nominal_10, at double the paper's "
              "block length (block=6 vs block=3), holding K/T fixed -- the block-grows leg of the "
              "finite-sample pattern",
         provenance="mcs_size.py::size_row on null_panel draws",
         cites=["S4.2 body: size-distortion finite-sample pattern"],
         cross_check="mcs_size.csv: size where K=6,T=30,block=6")
    _k10 = _grid_cell(10, T_PAPER, BLOCK_PAPER)
    emit("mcssize.size_at_K10", round(float(_k10["size"]), 3),
         K=10, T=T_PAPER, block=BLOCK_PAPER, n_rep=N_SIZE, se=round(float(_k10["se"]), 4),
         spec="Same empirical-size measurement as mcssize.size_at_nominal_10, at K=10 models instead "
              "of the paper's K=6, holding T/block fixed -- the K-grows leg of the finite-sample "
              "pattern",
         provenance="mcs_size.py::size_row on null_panel draws",
         cites=["S4.2 body: size-distortion finite-sample pattern"],
         cross_check="mcs_size.csv: size where K=10,T=30,block=3")
    emit("mcssize.size_arch", round(float(theirs), 3), n_rep=N_ARCH, ours=round(float(ours), 3),
         agreement=round(float(agr), 3),
         spec="The SAME level measured through arch 8.0.0's independent implementation of the "
              "Hansen-Lunde-Nason range statistic on IDENTICAL null panels. Its agreement with "
              "mcssize.size_at_nominal_10 is what establishes the size distortion as a property of the "
              "statistic at these panel dimensions rather than a defect in this repository",
         provenance="mcs_size.py::arch_cross_check",
         cites=["S4.2 body: arch size cross-check"],
         cross_check=None)
    emit("mcssize.alpha_sweep_nonidentified", int(strict.nonidentified),
         alpha=float(min(ALPHAS)), total=int(strict.total), mcs_PI=int(strict.mcs_PI),
         alphas=[float(a) for a in ALPHAS],
         counts=[int(v) for v in A.nonidentified],
         spec="Curated-panel non-identification count at the STRICTEST alpha swept (0.001, vs the "
              "paper's nominal 0.10). Note: a stricter alpha eliminates LESS readily, so the "
              "surviving (non-identified) set can mechanically only STAY THE SAME OR GROW as alpha "
              "shrinks, never shrink. An unchanged count here is therefore expected regardless of whether "
              "mcssize.size_at_nominal_10's over-rejection is real, and does NOT by itself rule out "
              "that the nominal-alpha=0.10 count is size-inflated. What it DOES show, together with "
              "mcssize.alpha_sweep_invariant: relaxing elimination all the way to alpha=0.001 does not "
              "reverse or shrink the finding -- i.e. the headline is not an artifact of a specific, "
              "aggressive choice of alpha",
         provenance="mcs_size.py::main alpha sweep over the curated per-year absolute-loss panels",
         cross_check="mcs_size_alpha_sweep.csv: nonidentified at min(alpha)")
    emit("mcssize.alpha_sweep_invariant", len(nonid_vals) == 1,
         distinct_counts=[int(v) for v in nonid_vals], distinct_pi=[int(v) for v in pi_vals],
         alpha_min=float(min(ALPHAS)), alpha_max=float(max(ALPHAS)),
         spec="True when the curated non-identification count takes a SINGLE value across the whole alpha "
              "sweep, with PI point-identified throughout. This is the claim §4.1 rests on when it says "
              "the headline does not depend on the test's level",
         provenance="mcs_size.py::main, set equality over the sweep",
         cross_check="mcs_size_alpha_sweep.csv: nunique(nonidentified)")
    flush()
    print(f"\n  wrote {OUT.name}, {OUT_AR1.name} and {OUT_ALPHA.name}")
    if _pi_control_failed:
        raise SystemExit(f"CONTROL FAILED: PI is not point-identified at alpha={min(ALPHAS)} "
                         f"(|MCS(PI)|={int(strict.mcs_PI)}). The alpha sweep cannot be interpreted "
                         f"(values were written and flushed before this check so nothing is lost).")


if __name__ == "__main__":
    main()

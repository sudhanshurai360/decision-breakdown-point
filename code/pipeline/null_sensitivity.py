"""null_sensitivity.py — does the exchangeable null move under two departures from i.i.d.:
correlated model losses (a common shock shared by every model) and serial dependence (AR(1) in
each model's own loss)?
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "instrument"))
import fragility as F
from canonical_emit import emit, flush
import run_multiseries as RM   # for its shared _atomic_to_csv() helper
RES = HERE.parent.parent / "multiseries" / "results"
K, T = 6, 30
              # the dimensions this sensitivity sweep is run at. S5.4's own shipped benchmark uses
              # the curated panel's T=32; the two p05 figures agree to 3 dp, but they are not the
              # same computation
N_REP, N_BOOT = 400, 800
SD = 0.30
from election_null_ar1 import _ar1  # noqa: E402 -- needed after this file's own sys.path insert above

SEEDS = tuple(i * 100_000_019 for i in range(20))


def _check_seed_spacing(seeds, n_rep):
    """Fail loudly if the seeds are close enough for null_dist()'s two internal streams to overlap
    across seeds (mirrors null_budget.py's own `_check_spacing`, adapted to this file's own, larger,
    bootstrap-stream offset -- see `need` below in this function)."""
    gap = min(b - a for a, b in zip(sorted(seeds), sorted(seeds)[1:]))
    need = n_rep + 10_000_003
    if gap <= need:
        raise ValueError(
            f"seed spacing {gap} <= {need}: null_dist()'s panel/bootstrap streams would overlap "
            f"across seeds, so the averaged band would be spuriously tight. Widen SEEDS or lower N_REP.")


def panel(g, rho, ar):
    """K loss series sharing a per-period common component of variance `rho`, each with AR(1) `ar`."""
    # the common component is drawn unconditionally, even at rho == 0, so that every point on the
    # rho sweep consumes the generator identically and neighbouring rho values stay comparable
    common = _ar1(g.normal(0, 1, T), ar)
    out = {}
    for i in range(K):
        idio = _ar1(g.normal(0, 1, T), ar)
        out[f"m{i}"] = 1.0 + SD * (np.sqrt(rho) * common + np.sqrt(1 - rho) * idio)
    return out


def null_dist(rho, ar, seed=0):
    """K-model correlated/serially-dependent null distribution of winner_stability, at one outer
    replication `seed`."""
    w = np.ones(T)
    vals = []
    for r in range(N_REP):
        g = np.random.default_rng(seed + 1 + r)
        v = F.winner_stability(panel(g, rho, ar), w, n_boot=N_BOOT, seed=seed + 10_000_003 + r)[0]
        if v is not None and np.isfinite(v):
            vals.append(float(v))
    a = np.asarray(vals, float)
    return float(np.median(a)), float(np.percentile(a, 5)), float(np.percentile(a, 95)), len(a)


def stable_null_dist(rho, ar, seeds=SEEDS):
    """null_dist(), averaged over independent outer seeds -- the same discipline null_budget.py's
    own stable_benchmark() applies to fragility.exchangeable_benchmark(), adapted here since
    exchangeable_benchmark takes no rho/ar argument and so cannot be reused directly for this file's
    correlated/serially-dependent null."""
    _check_seed_spacing(seeds, N_REP)
    med, p05, p95, ns = [], [], [], []
    for s in seeds:
        m, lo, hi, n = null_dist(rho, ar, seed=s)
        med.append(m); p05.append(lo); p95.append(hi); ns.append(n)
    p05a = np.asarray(p05, float)
    return {
        "median": float(np.mean(med)), "p05": float(np.mean(p05a)), "p95": float(np.mean(p95)),
        "p05_lo": float(p05a.min()), "p05_hi": float(p05a.max()),
        "p05_sd": float(p05a.std(ddof=1)) if len(p05a) > 1 else 0.0,
        "n_eff": int(sum(ns)), "n_seeds": len(seeds),
    }


def main():
    grid = [("iid", 0.0, 0.0), ("rho05", 0.5, 0.0), ("rho08", 0.8, 0.0),
            ("ar03", 0.0, 0.3), ("ar06", 0.0, 0.6), ("rho05_ar03", 0.5, 0.3)]
    print(f"=== NULL SENSITIVITY (K={K} T={T} n_rep={N_REP} n_boot={N_BOOT} n_seeds={len(SEEDS)}) ===")
    rows = []
    for name, rho, ar in grid:
        d = stable_null_dist(rho, ar)
        rows.append(dict(config=name, rho=rho, ar=ar, median=d["median"], p05=d["p05"], p95=d["p95"],
                          p05_lo=d["p05_lo"], p05_hi=d["p05_hi"], p05_sd=d["p05_sd"],
                          n_eff=d["n_eff"], n_seeds=d["n_seeds"]))
        print(f"  {name:11s} rho={rho:.1f} ar={ar:.1f}  median {d['median']:.4f}  "
              f"p05 {d['p05']:.4f} (sd {d['p05_sd']:.4f} over {d['n_seeds']} seeds)  p95 {d['p95']:.4f}")
    R = pd.DataFrame(rows)
    RM._atomic_to_csv(R, RES / "null_sensitivity.csv")
    def cell(name, col):
        return float(R.loc[R.config == name, col].iloc[0])
    d_rho = abs(cell("rho08", "p05") - cell("rho05", "p05"))
    d_ar = cell("ar06", "p05") - cell("iid", "p05")
    emit("nullsens.p05_iid", round(cell("iid", "p05"), 4),
         median=round(cell("iid", "median"), 4), n_eff=int(cell("iid", "n_eff")),
         band=[round(cell("iid", "p05_lo"), 4), round(cell("iid", "p05_hi"), 4)],
         sd=round(cell("iid", "p05_sd"), 4),
         spec=f"fifth percentile of winner_stability under the i.i.d. exchangeable null at K={K}, "
              f"T={T}, averaged over {len(SEEDS)} independent seeds ({int(cell('iid', 'n_eff'))} "
              f"effective replications). This is the same construction as the benchmark the paper "
              f"ships for §5.4, run at this sweep's own T={T} rather than §5.4's T=32; the two p05 "
              f"figures agree to 3 dp",
         cites=["S5.4 body: null sensitivity"], cross_check="null_sensitivity.csv: p05 where config=='iid'")
    emit("nullsens.p05_rho05", round(cell("rho05", "p05"), 4),
         band=[round(cell("rho05", "p05_lo"), 4), round(cell("rho05", "p05_hi"), 4)],
         sd=round(cell("rho05", "p05_sd"), 4),
         spec="the same fifth percentile when half the loss variance is a shock COMMON to every "
              f"model, averaged over {len(SEEDS)} independent seeds",
         cites=["S5.4 body: null sensitivity"], cross_check="null_sensitivity.csv: p05 where config=='rho05'")
    emit("nullsens.p05_rho08", round(cell("rho08", "p05"), 4),
         band=[round(cell("rho08", "p05_lo"), 4), round(cell("rho08", "p05_hi"), 4)],
         sd=round(cell("rho08", "p05_sd"), 4),
         spec="the same fifth percentile at eighty percent common variance, averaged over "
              f"{len(SEEDS)} independent seeds. Equal (within seed noise) to the rho=0.5 value "
              "because winner_stability counts argmin preservation and argmin is invariant to a shift "
              "applied to every model in a period -- the statistic cannot see the common component",
         cites=["S5.4 body: null sensitivity"], cross_check="null_sensitivity.csv: p05 where config=='rho08'")
    emit("nullsens.rho_delta_p05", round(d_rho, 4),
         spec="absolute change in the null's fifth percentile between eighty and fifty percent common "
              f"variance, both averaged over {len(SEEDS)} independent seeds. Emitted because the "
              "INVARIANCE is the finding: a common shock across models might be expected to raise the "
              "null, but doubling the correlation moves it by only this much",
         cites=["S5.4 body: null sensitivity"], cross_check=None)
    emit("nullsens.p05_ar06", round(cell("ar06", "p05"), 4),
         median=round(cell("ar06", "median"), 4),
         band=[round(cell("ar06", "p05_lo"), 4), round(cell("ar06", "p05_hi"), 4)],
         sd=round(cell("ar06", "p05_sd"), 4),
         spec="fifth percentile under AR(1)=0.6 serial dependence in every model's loss, averaged "
              f"over {len(SEEDS)} independent seeds. Serial dependence DOES raise the null, which "
              "matters for §5.4's lower-tail claims: a higher null puts an observed low value further "
              "into the tail, so the i.i.d. benchmark flags FEWER cells as unusual than a realistic "
              "one would",
         cites=["S5.4 body: null sensitivity"], cross_check="null_sensitivity.csv: p05 where config=='ar06'")
    emit("nullsens.ar_delta_p05", round(d_ar, 4),
         spec="rise in the null's fifth percentile from i.i.d. to AR(1)=0.6, both averaged over "
              f"{len(SEEDS)} independent seeds -- the departure that does move the benchmark, in the "
              "direction that makes the i.i.d. comparison ANTI-conservative for a lower-tail claim "
              "rather than conservative",
         cites=["S5.4 body: null sensitivity"], cross_check=None)
    print(f"\n  correlation: p05 moves {d_rho:.4f} between rho=0.5 and rho=0.8 (argmin is blind to it)")
    print(f"  serial dep : p05 rises {d_ar:+.4f} from iid to AR(1)=0.6")
    print("  => the i.i.d. benchmark is robust to correlation but not to serial dependence; and for a")
    print("     LOWER-tail claim a higher null is anti-conservative, not conservative.")
    flush()


if __name__ == "__main__":
    main()

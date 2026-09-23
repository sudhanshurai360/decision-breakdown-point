"""null_budget.py — a calibrated null with enough replications behind it to be a THRESHOLD.
"""
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "instrument"))
import fragility as F
SEEDS = tuple(i * 10_000_019 for i in range(20))   # 20 seeds spaced > N_REP apart so per-replicate streams never overlap
N_REP = 1000        # per seed; 20 seeds => 20000 effective replications
N_BOOT = 800         # per replicate


def _check_spacing(seeds, n_rep):
    """Fail loudly if the seeds are close enough for their random streams to overlap."""
    gap = min(b - a for a, b in zip(sorted(seeds), sorted(seeds)[1:]))
    need = n_rep + 1_000_003  # the bootstrap offset is the larger of the two strides
    if gap <= need:
        raise ValueError(
            f"seed spacing {gap} <= {need}: exchangeable_benchmark's streams would overlap across "
            f"seeds, so the replications would not be independent and the reported band would be "
            f"spuriously tight. Widen SEEDS or lower n_rep.")


def stable_benchmark(K, T, n_rep=N_REP, n_boot=N_BOOT, seeds=SEEDS, block=None):
    """Exchangeable null at (K, T), averaged over independent seeds."""
    _check_spacing(seeds, n_rep)
    med, p05, p95 = [], [], []
    for s in seeds:
        m, lo, hi = F.exchangeable_benchmark(K, T, n_rep=n_rep, n_boot=n_boot, seed=s, block=block)
        med.append(m); p05.append(lo); p95.append(hi)
    p05 = np.asarray(p05, float)
    p95 = np.asarray(p95, float)
    return {
        "median": float(np.mean(med)),
        "p05": float(np.mean(p05)),
        "p95": float(np.mean(p95)),
        "p05_lo": float(p05.min()),
        "p05_hi": float(p05.max()),
        "p05_sd": float(p05.std(ddof=1)),
        "p95_lo": float(p95.min()),
        "p95_hi": float(p95.max()),
        "p95_sd": float(p95.std(ddof=1)),
        "n_eff": int(len(seeds) * n_rep),
        "n_seeds": len(seeds),
    }


def stable_stability(L, w, n_boot=10_000, seeds=SEEDS, block=None):
    """Observed winner_stability averaged over independent bootstrap seeds."""
    v = [F.winner_stability(L, w, n_boot=n_boot, block=block, seed=s)[0] for s in seeds]
    v = np.asarray(v, float)
    return {"value": float(v.mean()), "lo": float(v.min()), "hi": float(v.max()),
            "sd": float(v.std(ddof=1)), "n_seeds": len(seeds), "n_boot": n_boot}


def band(sd, n_seeds, conf=0.95):
    """Half-width of a `conf` interval for a mean of `n_seeds` draws, using the Student-t quantile."""
    from statistics import NormalDist
    try:
        from scipy.stats import t as _t
        q = float(_t.ppf(0.5 + conf / 2, df=max(1, n_seeds - 1)))
    except Exception:
        q = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306,
             10: 2.262, 11: 2.228, 12: 2.201, 13: 2.179, 14: 2.160, 15: 2.145, 16: 2.131,
             17: 2.120, 18: 2.110, 19: 2.101, 20: 2.093, 21: 2.086}.get(
                 n_seeds, NormalDist().inv_cdf(0.5 + conf / 2))
    return q * sd / (n_seeds ** 0.5)


def verdict(obs, null, obs_band=0.0, null_band=0.0):
    """'fragile' / 'not_fragile' / 'undetermined'."""
    if obs + obs_band < null - null_band:
        return "fragile"
    if obs - obs_band > null + null_band:
        return "not_fragile"
    return "undetermined"

#!/usr/bin/env python3
"""
metric_consistency.py — the same metric NAME must mean the same COMPUTATION in every producer.
"""
import hashlib
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "code" / "pipeline"))
sys.path.insert(0, str(ROOT / "code" / "instrument"))

PRED = ROOT / "multiseries" / "results" / "breadth_distinct" / "breadth_distinct_predictions.csv"

COVERED = ["social_choice", "election_null_ar1"]
N_COVERED_PRODUCERS = 2          # len(COVERED) today
N_COMPARED_NAMES = 2             # metric names ("mase", "rmsse") with >=2 producers exposing them
                                  # today -- i.e. the count that actually got compared, not just seen

COVERED_PRODUCERS_HASH = "6279a7d9ee0659fbda0ea4d31467e63ba1ce09eb50e50b9942cacc0d41515cb5"
COMPARED_NAMES_HASH = "c35b51b4d95a5d48fa90bcf98dc7343faaf9cfb1881248d8d61b57270aba162e"

UNCOVERED = {
    "mcs_test.py":       'gm["sae"].mean(); gm["msse"].mean(); sqrt(mean(msse)) -- MSSE and '
                          'RMSSE, both computed inline, no METRICS mapping',
    "mcs_robustness.py":  'same pair as mcs_test.py',
    "combination_benchmark.py": '["sae"].mean() only; no msse aggregation',
    "mcs_power.py":       '["sae"].mean() only; no msse aggregation',
    "walkforward.py":     'METRIC_COL = {"mase": "sae", "rmsse": "msse"} + agg_metric()\'s '
                          'conditional sqrt for "rmsse" -- structurally the same shape this gate '
                          'exists to police, not yet adapted to this file\'s standalone-dict-plus-'
                          'function shape (no module-level METRICS attribute for _load_metrics() '
                          'to find)',
}


def _load_metrics():
    out = {}
    for mod in COVERED:
        m = __import__(mod)
        if not hasattr(m, "METRICS"):
            raise RuntimeError(f"{mod} is listed as COVERED but exposes no METRICS mapping")
        out[mod] = m.METRICS
    return out


def check_synthetic(metrics) -> list[str]:
    """Adversarial vectors."""
    bad = []
    cases = {
        "dispersed":   pd.Series([0.25, 4.0, 9.0, 16.0]),
        "with_zero":   pd.Series([0.0, 1.0, 4.0]),
        "tiny_spread": pd.Series([1.0, 1.0000001]),
        "wide":        pd.Series([1e-6, 1e3]),
    }
    names = sorted({n for M in metrics.values() for n in M})
    for name in names:
        for cname, vec in cases.items():
            vals = {}
            for mod, M in metrics.items():
                if name not in M:
                    continue
                col, agg = M[name]
                vals[mod] = (col, float(agg(vec)))
            cols = {c for c, _ in vals.values()}
            if len(cols) > 1:
                bad.append(f"metric '{name}' reads different COLUMNS across producers: "
                           f"{ {m: c for m, (c, _) in vals.items()} }")
            nums = [v for _, v in vals.values()]
            if len(nums) > 1 and not all(abs(x - nums[0]) < 1e-12 for x in nums):
                bad.append(f"metric '{name}' on {cname}: aggregators DISAGREE — "
                           f"{ {m: round(v, 10) for m, (_, v) in vals.items()} }")
    return bad


def check_real_panel(metrics) -> list[str]:
    """The same comparison on the frozen loss matrix, per series and model — the object the paper's
    curated results are actually built from."""
    if not PRED.exists():
        return [f"frozen loss matrix missing: {PRED}"]
    d = pd.read_csv(PRED)
    bad = []
    names = sorted({n for M in metrics.values() for n in M})
    for name in names:
        producers = {m: M[name] for m, M in metrics.items() if name in M}
        if len(producers) < 2:
            continue
        panels = {}
        for mod, (col, agg) in producers.items():
            panels[mod] = (d.groupby(["series", "model", "year"])[col]
                            .apply(agg).rename(mod))
        merged = pd.concat(panels.values(), axis=1)
        ref = merged.columns[0]
        for other in merged.columns[1:]:
            diff = (merged[ref] - merged[other]).abs()
            isna_mismatch = merged[ref].isna() != merged[other].isna()
            bad_mask = (diff > 1e-12) | isna_mismatch
            n_bad = int(bad_mask.sum())
            if n_bad:
                worst_idx = diff[bad_mask].idxmax() if diff[bad_mask].notna().any() else merged.index[bad_mask][0]
                worst = merged.loc[worst_idx]
                bad.append(f"metric '{name}': {ref} vs {other} differ on {n_bad} of {len(diff)} "
                           f"(series, model, year) cells; worst at {worst_idx} "
                           f"({ref}={worst[ref]:.6f}, {other}={worst[other]:.6f})")
    return bad


def main() -> int:
    metrics = _load_metrics()
    print("=== METRIC CONSISTENCY — same name, same computation, across producers ===")
    for mod, M in metrics.items():
        print(f"  {mod:16} exposes: { {n: c for n, (c, _) in M.items()} }")

    bad = check_synthetic(metrics) + check_real_panel(metrics)

    all_names = sorted({n for M in metrics.values() for n in M})
    compared_names = sorted(n for n in all_names if sum(1 for M in metrics.values() if n in M) >= 2)
    n_compared = len(compared_names)
    print(f"\n  producers compared: {len(metrics)}   metric names: {all_names}   "
          f"names actually compared (>=2 producers each): {n_compared}")
    if bad:
        print(f"\n  \033[31mINCONSISTENT\033[0m ({len(bad)}):")
        for b in bad:
            print(f"    {b}")
    else:
        print("  every shared metric name computes identically in every covered producer.")

    print(f"\n  NOT COVERED by this gate ({len(UNCOVERED)}) — inline aggregation, no named mapping:")
    for where, note in UNCOVERED.items():
        print(f"    {where:32} {note}")
    print("  A producer joins this check by exposing METRICS = {name: (column, aggregator)}.")

    ratchet_ok = True
    for name, want, got in (("N_COVERED_PRODUCERS", N_COVERED_PRODUCERS, len(metrics)),
                             ("N_COMPARED_NAMES", N_COMPARED_NAMES, n_compared)):
        if want != got:
            print(f"\n  RATCHET INCONSISTENT — {name} says {want}, this run gives {got}")
            ratchet_ok = False

    for name, want, got in (
        ("COVERED_PRODUCERS_HASH", COVERED_PRODUCERS_HASH,
         hashlib.sha256(",".join(sorted(COVERED)).encode()).hexdigest()),
        ("COMPARED_NAMES_HASH", COMPARED_NAMES_HASH,
         hashlib.sha256(",".join(compared_names).encode()).hexdigest()),
    ):
        if want != got:
            print(f"\n  RATCHET INCONSISTENT — {name} says '{want}', this run's actual set "
                  f"hashes to '{got}'")
            ratchet_ok = False

    good = not bad and ratchet_ok
    print(f"\n  {'PASS' if good else 'FAIL'}  {len(bad)} inconsistency(ies), "
          f"ratchet {'ok' if ratchet_ok else 'INCONSISTENT'}")
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())

"""
run_multiseries.py — walk-forward driver: one run per series, then --combine.
"""
import json
import shutil
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT  = HERE.parent.parent                 # <repo root>/code/<subdir>/ -> <repo root>
MULTI = ROOT / "multiseries"               # data + results live HERE, not under code/
INSTR = ROOT / "code" / "instrument"       # shared estimator

RESULTS = MULTI / "results"
MANIFEST = MULTI / "data" / "raw" / "breadth_manifest_distinct.json"   # the curated panel's own selection log
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(INSTR))
import models_freq as MF          # noqa: E402
import walkforward as WF          # noqa: E402
import fragility as F             # noqa: E402

# env overrides (for fast sanity checks; canonical run uses the defaults)
import os
MODELS = [n.strip() for n in os.environ.get(
    "MS_MODELS", "seasonal_naive,ar1,ets,theta,sarima,dhr").split(",") if n.strip()]
SHOCK_YEARS = [2008, 2009, 2020, 2021]      # pre-committed (GFC + COVID); anti-cherry-pick
EVAL_START = int(os.environ.get("MS_EVAL_START", "1990"))
EVAL_END = int(os.environ.get("MS_EVAL_END", "2025"))
M = 12
N_BOOT = int(os.environ.get("MS_NBOOT", "2000"))


def _complete_years(L: dict, years):
    """Keep only years where EVERY model has a finite loss value (mase or rmsse; the diagnostic
    needs complete rows)."""
    keep = [i for i, _ in enumerate(years) if all(np.isfinite(L[m][i]) for m in L)]
    return keep


def _atomic_to_csv(df: pd.DataFrame, out: Path, **to_csv_kwargs) -> None:
    """Write `df` to `out` atomically: temp file + os.replace(), with the destination's existing
    mode preserved (Path.replace() does not carry over permission bits, and the temp file is
    created at the process's default umask)."""
    to_csv_kwargs.setdefault("index", False)
    tmp = out.with_suffix(out.suffix + ".tmp")
    df.to_csv(tmp, **to_csv_kwargs)
    if out.exists():
        shutil.copymode(out, tmp)
    tmp.replace(out)


def run_series(sid: str) -> tuple:
    """Run the walk-forward ONCE (predictions), then score the SAME predictions under each metric (mase, rmsse)
    -> one fragility row per (series, metric)."""
    mdls = MF.build(MODELS, m=M)
    detail = WF.run(sid, mdls, eval_start=EVAL_START, eval_end=EVAL_END, m=M)
    if detail is None or detail.empty:
        return [], None
    rows = []
    for metric in ("mase", "rmsse"):
        agg = WF.agg_metric(detail, metric)
        L, w, years = WF.to_fragility_inputs(agg)
        keep = _complete_years(L, years)
        L2 = {m: L[m][keep] for m in L}; w2 = w[keep]; yrs = [years[i] for i in keep]
        r = F.fragility(L2, w2, shock_periods=SHOCK_YEARS, labels=yrs, n_boot=N_BOOT, seed=0)
        rows.append({
            "series": sid, "metric": metric, "n_years": len(yrs), "year_min": min(yrs), "year_max": max(yrs),
            "pooled_winner": r["pooled_winner"], "per_period_winner": r["per_period_winner"],
            "runner_up": r["runner_up"], "k_star": r["k_star"], "k_over_T": round(r["k_over_T"], 3),
            "reversal": r["reversal"], "responsible": ";".join(map(str, r["responsible"])),
            "shock_coincidence": r["shock_coincidence"], "concentration": round(r["concentration"], 3),
            "winner_stability": round(r["winner_stability"], 3), "fragile": r["fragile"],
        })
    return rows, detail


PARTS = RESULTS / "parts"


def combine_scope_warnings(found_series, manifest_path=MANIFEST):
    """Compare the series `combine()` is about to merge against the curated panel's own
    pre-committed selection log (breadth_manifest_distinct.json's `passed` list)."""
    if not manifest_path.exists():
        return []
    expected = set(json.loads(manifest_path.read_text()).get("passed", []))
    if not expected:
        return []
    found = set(found_series)
    warnings = []
    extra = found - expected
    missing = expected - found
    if extra:
        warnings.append(f"{len(extra)} series being combined are NOT in the curated panel's manifest "
                         f"({manifest_path.name}): {sorted(extra)} -- possible stale/leftover parts "
                         f"from an unrelated run sitting in {PARTS}")
    if missing:
        warnings.append(f"{len(missing)} curated-panel series are MISSING from the parts being "
                         f"combined: {sorted(missing)} -- this combine is a PARTIAL scope, not the "
                         f"full {len(expected)}-series panel the manifest expects")
    return warnings


def combine():
    """Assemble per-series part files (written by parallel single-series runs) into the canonical outputs."""
    fparts = sorted(PARTS.glob("*_fragility.csv"))
    pparts = sorted(PARTS.glob("*_predictions.csv"))
    if not fparts:
        print("no parts to combine"); return
    if not pparts:
        print("no prediction parts to combine (fragility parts exist, but zero predictions parts) "
              "-- refusing to write a partial, unpaired output; rerun the affected series' full "
              "write before combining")
        return
    found_series = sorted({p.name.replace("_fragility.csv", "") for p in fparts})
    pred_series = sorted({p.name.replace("_predictions.csv", "") for p in pparts})
    if found_series != pred_series:
        only_frag = sorted(set(found_series) - set(pred_series))
        only_pred = sorted(set(pred_series) - set(found_series))
        print(f"  SCOPE WARNING: fragility parts and predictions parts cover DIFFERENT series sets "
              f"-- fragility-only: {only_frag}, predictions-only: {only_pred}. Likely an interrupted "
              f"per-series write; rerun the affected series before trusting the combined output.")
    for w in combine_scope_warnings(found_series):
        print(f"  SCOPE WARNING: {w}")
    frag = pd.concat([pd.read_csv(p) for p in fparts], ignore_index=True).sort_values(["series", "metric"])
    _atomic_to_csv(frag, RESULTS / "multiseries_fragility.csv")
    preds = pd.concat([pd.read_csv(p) for p in pparts], ignore_index=True)
    _atomic_to_csv(preds, RESULTS / "multiseries_predictions.csv")
    print("\n=== COMBINED MULTISERIES FRAGILITY TABLE (per metric) ===")
    print(frag.to_string(index=False))
    print(f"\nwrote {RESULTS/'multiseries_fragility.csv'} (+ multiseries_predictions.csv) from "
          f"{len(fparts)} series parts: {found_series}")


def main():
    unknown_flags = [a for a in sys.argv[1:] if a.startswith("--") and a != "--combine"]
    if unknown_flags:
        print(f"  UNKNOWN FLAG(S): {unknown_flags} -- refusing to silently ignore a possible typo "
              f"(a mistyped series name here, e.g. '--UNRATE' instead of 'UNRATE', would otherwise "
              f"be silently dropped and fall back to ALL series). Recognized: --combine, or bare "
              f"series filenames as positional args.", file=sys.stderr)
        sys.exit(2)
    if "--combine" in sys.argv:
        combine(); return
    wanted = [a for a in sys.argv[1:] if not a.startswith("--")] or [p.name for p in WF.RAW.glob("*_alfred_vintages.parquet")]
    sids = [str(s).replace("_alfred_vintages.parquet", "") for s in wanted]
    PARTS.mkdir(parents=True, exist_ok=True)

    for sid in sids:
        f = WF.RAW / f"{sid}_alfred_vintages.parquet"
        if not f.exists():
            print(f"  [skip] {sid}: no vintages parquet"); continue
        print(f"\n=== {sid} ===")
        srows, detail = run_series(sid)
        if not srows:
            print(f"  [skip] {sid}: no eval rows"); continue
        # write per-series parts immediately (parallel-safe: unique sid per file; partial progress preserved)
        _atomic_to_csv(pd.DataFrame(srows), PARTS / f"{sid}_fragility.csv")
        _atomic_to_csv(detail, PARTS / f"{sid}_predictions.csv")
        for row in srows:
            print(f"  [{row['metric']:5s}] pooled={row['pooled_winner']:14s} per-period={row['per_period_winner']:14s} "
                  f"k*={row['k_star']} (k/T={row['k_over_T']})  reversal={row['reversal']}  "
                  f"fragile={row['fragile']}  responsible={row['responsible']}")
        print(f"  wrote parts/{sid}_fragility.csv (+ predictions)")


if __name__ == "__main__":
    main()

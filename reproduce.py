#!/usr/bin/env python3
"""
reproduce.py — one command to regenerate + reverify every analysis output of the paper, deterministically.
"""
import os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable                                            # run sub-scripts with the SAME interpreter

REQUIRED_PACKAGES = ["numpy", "pandas", "scipy", "statsmodels", "fredapi", "arch"]


def _preflight_check():
    missing = []
    for pkg in REQUIRED_PACKAGES:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        sys.exit(
            f"FATAL: this interpreter ({PY}) is missing required package(s): {', '.join(missing)}.\n"
            f"reproduce.py refuses to run every downstream stage under a wrong/incomplete "
            f"environment -- several stages silently degrade (not crash) their forecasts when a "
            f"dependency is missing, which can silently corrupt a cited number. See REPRODUCE.md "
            f"for the correct venv setup."
        )
    frozen_caches = [
        ROOT / "multiseries" / "data" / "fredmd" / "fredmd_api.parquet",
        ROOT / "multiseries" / "data" / "state_ur.parquet",
        ROOT / "multiseries" / "data" / "unrate_vintages.parquet",
        ROOT / "multiseries" / "data" / "open_llm.parquet",
    ]
    missing_caches = [str(p.relative_to(ROOT)) for p in frozen_caches if not p.exists()]
    if missing_caches:
        sys.exit(
            f"FATAL: {len(missing_caches)} frozen fetched-data cache(s) are missing: "
            f"{', '.join(missing_caches)}.\n"
            f"These are pinned (FROZEN_ATOM entry 3) so re-scoring is deterministic offline -- their "
            f"producers (fred_md_sweep.py/state_eb_decision.py/sahm_evans_realtime.py/ml_witness.py) "
            f"would otherwise silently refetch live data on a missing cache, which need not match "
            f"what originally shipped (FRED/ALFRED/HF continue to revise data over time). Restore "
            f"the missing file(s) from git before running the pipeline; do not let a stage refetch "
            f"them as a substitute."
        )
    for pkg in REQUIRED_PACKAGES:
        mod = sys.modules.get(pkg) or __import__(pkg)
        print(f"  {pkg}=={getattr(mod, '__version__', '?')}", flush=True)
    print(f"  MS_FIT_TIMEOUT={os.environ.get('MS_FIT_TIMEOUT', '25 (default)')}", flush=True)

FROZEN_ATOM = [
    ("results/extension_scoreboard_1988-2025.csv", "UI flagship NEURAL scores (nbeats/nhits/tft) — GPU-trained on a separate machine, SHA-pinned in REPRODUCE.md"),
    ("multiseries/results/breadth_distinct/breadth_distinct_predictions.csv",
     "12-series predictions (classical tier), byte-identical to multiseries/results/multiseries_predictions.csv "
     "as produced by walkforward.py/run_multiseries.py (CPU, local). Frozen to preserve a fixed evaluation "
     "lineage across the analysis passes built on it; protected by git-tracking + no writer, NOT by an active "
     "hash check. Last regenerated 2026-09-19 (`run_multiseries.py DGORDER BUSINV PI UNEMPLOY` + `--combine`; "
     "sha256 c12e984d3ede22f74baf3a94e94d05acd4fc3cb32c8552419ae70578f50f3cc3). Regenerating it "
     "is NOT a reproduce.py stage: run those two commands and copy multiseries_predictions.csv here."),
    ("multiseries/data/fredmd/fredmd_api.parquet + multiseries/data/{state_ur,unrate_vintages,open_llm}.parquet "
     "+ multiseries/data/raw/*.parquet (32 per-series ALFRED vintages, the merged state_ur_alfred_vintages.parquet, "
     "and 51 state-UR parts) + "
     "multiseries/data/raw/{breadth_manifest,breadth_manifest_distinct,lineage}.json",
     "fetched-data caches and their selection/provenance logs (FRED/ALFRED/HF) — git-tracked (~4 MB) and "
     "pinned for correctness, not only determinism: FRED/ALFRED continue to revise and add vintages over "
     "time, so a re-fetch need not reproduce the recorded values."),
    ("results/contrast_exhibit.csv",
     "Flagship real-time-vs-revised window contrast, read by data_config.py; no producer script in this "
     "repository. Same protection as the entries above: git-tracked, no current writer, no active "
     "hash check."),
]

# (label, script relpath, output to freshness-check or None, tier)  — order = rough dependency / speed
PIPELINE = [
    ("k* object unit tests (greedy==brute force)", "code/instrument/test_fragility.py", None, "fast"),
    ("per-series Model Confidence Set", "code/pipeline/mcs_test.py", "multiseries/results/mcs_test.csv", "fast"),
    ("is non-identification just COVID? (placebo-controlled)", "code/pipeline/regime_robustness.py", "multiseries/results/regime_robustness.csv", "med"),
    ("social-choice / election reframe",                  "code/pipeline/social_choice.py", "multiseries/results/social_choice.csv", "med"),
    ("best-arm reframe (conceptual)",                     "code/pipeline/best_arm.py", None, "fast"),
    ("shape-not-magnitude mechanism",                     "code/pipeline/shape_not_magnitude.py", "multiseries/results/shape_not_magnitude.csv", "fast"),
    ("flagship k*/MCS (uses frozen neural scoreboard)",   "code/instrument/flagship_fragility.py", None, "med"),
    ("M4 submissions audit (unique champion)",            "code/pipeline/m4_submissions_audit.py", "multiseries/results/m4_submissions_leaderboard.csv", "med"),
    ("M4 champion-vs-runner-up margins",                   "code/pipeline/m4_margins.py", "multiseries/results/m4_margins.csv", "fast"),
    ("M4 real per-series k*/winner_stability",             "code/pipeline/m4_perseries_kstar.py", "multiseries/results/m4_perseries_kstar.csv", "slow"),
    ("M4 per-series winner-flip audit (5 standard forecasters)", ("code/pipeline/m_audit.py", "--emit"), "multiseries/results/m_audit_Monthly.csv", "slow"),
    ("M4 density / crowding curve",                     "code/pipeline/density_audit.py", "multiseries/results/m4_density_curve.csv", "med"),
    ("arena crowding (M4 + Open-LLM, same op + null)",    "code/pipeline/arena_crowding.py", None, "med"),
    ("ML-leaderboard witness (Open-LLM)",                 "code/pipeline/ml_witness.py", None, "med"),
    ("MCS discriminant-power sim",                        "code/pipeline/mcs_power.py", None, "med"),
    ("real-time Sahm/Evans (UNRATE ALFRED vintages)",     "code/pipeline/sahm_evans_realtime.py", None, "med"),
    ("FRED-MD population sweep",                  "code/pipeline/fred_md_sweep.py", "multiseries/results/fred_md_sweep.csv", "slow"),
    ("k* split-half predictive-persistence check",        "code/pipeline/kstar_persistence.py", "multiseries/results/kstar_persistence.csv", "slow"),
    ("MCS/k* bootstrap-robustness grid (+ FRED-MD band)","code/pipeline/mcs_robustness.py", "multiseries/results/mcs_monthly_resolution.csv", "slow"),
    ("curated-panel/flagship membership manifest (data.*)", "code/pipeline/data_config.py", None, "slow"),
    ("state-EB decision projection",                      "code/pipeline/state_eb_decision.py", "multiseries/results/state_eb_decision.csv", "slow"),
    # --- each of the following emits a canonical fragment, which the gate stages at the end merge and re-check.
    ("realised |t| beside k*",                            "code/pipeline/realised_t.py", "multiseries/results/realised_t.csv", "fast"),
    ("model-set similarity audit",                        "code/pipeline/model_set_audit.py", "multiseries/results/model_set_audit.csv", "slow"),

    ("equal-weight combination benchmark (S4.4)",         "code/pipeline/combination_benchmark.py", "multiseries/results/combination_benchmark.csv", "med"),
    ("null sensitivity (correlation / serial dependence)","code/pipeline/null_sensitivity.py", "multiseries/results/null_sensitivity.csv", "med"),
    ("election reframe null under empirical AR(1)",       "code/pipeline/election_null_ar1.py", "multiseries/results/election_null_ar1.csv", "slow"),
    ("flagship exhibit null under empirical AR(1)",       "code/pipeline/flagship_null_ar1.py", "results/flagship_null_ar1.csv", "slow"),
    ("Section 5 permutation-null power check",             "code/pipeline/social_choice_power.py", "multiseries/results/social_choice_power.csv", "med"),
    ("MCS actual level + alpha sweep",                    "code/pipeline/mcs_size.py", "multiseries/results/mcs_size.csv", "slow"),
    ("MCS circular-bootstrap check",                       "code/pipeline/mcs_circular_check.py", "multiseries/results/mcs_circular_check.csv", "slow"),
    ("state EB: initial-release vs revised vintages",   "code/pipeline/state_eb_vintage.py", "multiseries/results/state_eb_vintage.csv", "med"),
    ("dMCS decision-loss MCS + straddling",               "code/pipeline/dmcs.py", "multiseries/results/dmcs.csv", "slow"),

    ("canonical: merge fragments",                        "code/gates/canonical.py", "paper/canonical.json", "fast"),
    ("canonical: independent re-derivation gate",         ("code/gates/canonical.py", "--check"), None, "fast"),
    ("canonical: selftest (negative control)",            "code/gates/selftest_canonical.py", None, "fast"),
    ("acceptance: pipeline vs recorded values",           "code/gates/acceptance.py", None, "fast"),
]


def manifest():
    print("=== FROZEN INPUT ATOM (NOT regenerated — GPU-trained scores / external fetches) ===")
    for p, note in FROZEN_ATOM:
        print(f"  [frozen] {p}\n           {note}")
    print("\n=== REGENERATED PIPELINE (CPU-deterministic, downstream of the frozen atom) ===")
    for label, script, out, tier in PIPELINE:
        sc = script if isinstance(script, (list, tuple)) else (script,)
        print(f"  [{tier:>4}] {' '.join(sc):<48} -> {out or '(prints only)'}")


def run(fast):
    _preflight_check()
    started = time.time(); rows = []
    logdir = ROOT / "logs" / f"reproduce_{time.strftime('%Y%m%d_%H%M')}"
    logdir.mkdir(parents=True, exist_ok=True)
    print(f"  per-stage logs -> {logdir.relative_to(ROOT)}/")
    for i, (label, script, out, tier) in enumerate(PIPELINE, 1):
        if fast and tier == "slow":
            rows.append((label, script, "SKIP", "-", "-")); continue
        t0 = time.time()
        sc = script if isinstance(script, (list, tuple)) else (script,)
        r = subprocess.run([PY, str(ROOT / sc[0]), *sc[1:]], cwd=str(ROOT),
                           capture_output=True, text=True)
        dt = time.time() - t0
        suffix = ("_" + sc[1].strip("-")) if len(sc) > 1 else ""
        (logdir / f"{i:02d}_{Path(sc[0]).stem}{suffix}.log").write_text(
            f"$ {' '.join(sc)}\n[exit {r.returncode}, {dt:.0f}s]\n--- STDOUT ---\n{r.stdout}"
            f"\n--- STDERR ---\n{r.stderr}")
        ok = r.returncode == 0
        fresh = "n/a"
        if out is not None:
            f = ROOT / out
            fresh = "fresh" if (f.exists() and f.stat().st_mtime >= t0) else ("STALE" if f.exists() else "MISSING")
        status = "PASS" if ok and fresh in ("fresh", "n/a") else "FAIL"
        rows.append((label, script, status, f"{dt:.0f}s", fresh))
        if not ok:
            print(f"  !! {' '.join(sc)} exit {r.returncode}\n{r.stderr[-800:]}")
    print("\n=== REPRODUCE REPORT ===")
    print(f"  {'status':<6}{'time':>7}  {'output':<8} script")
    for label, script, status, dt, fresh in rows:
        sc = script if isinstance(script, (list, tuple)) else (script,)
        print(f"  {status:<6}{dt:>7}  {fresh:<8} {' '.join(sc)}")
    npass = sum(1 for x in rows if x[2] == "PASS"); nrun = sum(1 for x in rows if x[2] != "SKIP")
    print(f"\n  {npass}/{nrun} stages PASS ({time.time()-started:.0f}s total). Frozen atom untouched; all else regenerated.")
    return all(x[2] in ("PASS", "SKIP") for x in rows)


if __name__ == "__main__":
    _KNOWN_FLAGS = {"--list", "--fast"}
    _unknown = [a for a in sys.argv[1:] if a not in _KNOWN_FLAGS]
    if _unknown:
        print(f"reproduce.py: unrecognized argument(s) {_unknown} -- valid flags are "
              f"{sorted(_KNOWN_FLAGS)}", file=sys.stderr)
        sys.exit(2)
    if "--list" in sys.argv:
        manifest()
    else:
        ok = run(fast="--fast" in sys.argv)
        sys.exit(0 if ok else 1)

"""differential_test.py — the pipeline copy and the shipped package must compute the same thing.
"""
import argparse
import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent                      # <repo>/code/gates/ -> <repo>
PIPELINE_COPY = ROOT / "code" / "instrument" / "fragility.py"


def _load(path, name):
    """Import a module from an explicit path, under its own name."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_package_copy():
    """Import the installed package's fragility module (its relative imports need a real
    package import, not a file-path load)."""
    try:
        from selection_fragility._internals import fragility as mod
    except ImportError as e:
        raise ImportError(
            "could not import selection_fragility._internals -- is the package installed? "
            "Run: pip install selection-fragility==1.0.3 (the version requirements-lock.txt and "
            "REPRODUCE.md both pin for this gate)"
        ) from e
    return mod


def _panels(n, rng):
    """Varied panels: sizes, scales, near-ties, a dominant model, and degenerate edges."""
    out = []
    for i in range(n):
        K = int(rng.integers(2, 8))
        T = int(rng.integers(5, 40))
        kind = i % 5
        if kind == 0:                                   # generic
            M = rng.normal(1.0, 0.3, (T, K))
        elif kind == 1:                                 # near-tie
            M = rng.normal(1.0, 0.02, (T, K))
        elif kind == 2:                                 # one clear winner
            M = rng.normal(1.0, 0.3, (T, K)); M[:, 0] -= 0.5
        elif kind == 3:                                 # one pivotal shock period
            M = rng.normal(1.0, 0.1, (T, K)); M[T // 2, 1:] += 5.0
        else:                                           # heterogeneous variances
            M = np.column_stack([rng.normal(1.0, s, T) for s in rng.uniform(0.05, 1.0, K)])
        M = np.abs(M) + 1e-9                            # losses are non-negative
        L = {f"m{j}": M[:, j].astype(float) for j in range(K)}
        w = np.ones(T) if i % 2 else rng.uniform(0.5, 1.5, T)
        out.append((L, w))
    return out


def _same(a, b, tol=1e-12):
    """Deep equality."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k], tol) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a, float), np.asarray(b, float)
        return a.shape == b.shape and np.allclose(a, b, rtol=0, atol=tol, equal_nan=True)
    if isinstance(a, float) or isinstance(b, float):
        try:
            fa, fb = float(a), float(b)
        except (TypeError, ValueError):
            return a == b
        if np.isnan(fa) and np.isnan(fb):
            return True
        return abs(fa - fb) <= tol
    return a == b


# (name, builds the positional args from a panel). Seeded calls get the SAME seed on both sides --
# an unseeded comparison of a bootstrap would fail for reasons that say nothing about the code.
CASES = [
    ("pooled_winner",          lambda L, w: (L, w)),
    ("per_period_winner",      lambda L, w: (L,)),
    ("condorcet_winner",       lambda L, w: (L,)),
    ("condorcet_status",       lambda L, w: (L,)),
    ("decision_breakdown",     lambda L, w: (L, w)),
    ("surprise_concentration", lambda L, w: (L,)),
    ("winner_stability",       lambda L, w: (L, w)),
    ("fragility",              lambda L, w: (L, w)),
]
SEEDED = {"winner_stability": dict(n_boot=200, seed=0),
          "fragility":        dict(n_boot=200, seed=0)}

KNOWN_DIVERGENT = {
    "fragility": {
        "fragile": "package v1.0 deliberately retired this field (returns None without an "
                   "explicit benchmark=) after finding it INVERTED under the exchangeable null; "
                   "see the installed package's fragility.py, its 'fragile =' assignment",
        "pooled_runner_up": "package v1.0 routes this through pooled_winner()'s degenerate-tie "
                   "floor; the frozen pipeline copy still uses a raw, un-floored min and can "
                   "disagree on a near-tied panel under a weight-scale rescaling",
    },
}
N_KNOWN_DIVERGENT_FIELDS = 2   # total (function, field) pairs across all of KNOWN_DIVERGENT

N_SHARED_FUNCTIONS = 12

KNOWN_SCHEMA_DRIFT = {
    "fragility": {
        "pipeline_only": frozenset({
            "concentration_hi", "concentration_lo", "n_binding_tied", "per_period_tied",
            "responsible_by_opponent", "reversal_ambiguous",
        }),
        "package_only": frozenset({
            "binding_opponent", "binding_opponents", "concentration_range", "degenerate_margin",
            "plurality_winners", "screen",
        }),
    },
}
N_KNOWN_SCHEMA_DRIFT_KEYS = 12  # total pipeline_only + package_only keys across all entries above

KNOWN_DIVERGENT_HASH = "77eb75e8775dd5722eda4e56a6fdbbea690f17a53b4cd033b2b70a1c91744e6e"
SHARED_FUNCTIONS_HASH = "3d920f81d1f4282fd8dc5f6aa1387fa5e67c66380fb8389e97f0ca56869eda7b"
SCHEMA_DRIFT_KEYS_HASH = "7839da2d99f9d3d4c77b6dff3d723dfa8faa9bebe627bc1aec0c9f1b48cc3a4d"


def _dict_field_diff(name, ra, rb):
    """For a dict-shaped (ra, rb) pair -> (live_mismatches, known_divergent, pipeline_only_keys,
    package_only_keys)."""
    ka, kb = set(ra), set(rb)
    shared = ka & kb
    known = KNOWN_DIVERGENT.get(name, {})
    live, div = [], []
    for k in sorted(shared):
        if _same(ra[k], rb[k]):
            continue
        (div if k in known else live).append(k)
    return live, div, sorted(ka - kb), sorted(kb - ka)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", type=int, default=300)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    if not PIPELINE_COPY.exists():
        print(f"  FAIL  missing copy: {PIPELINE_COPY.relative_to(ROOT)}")
        return 2

    pipe = _load(PIPELINE_COPY, "_diff_pipeline_fragility")
    pkg = _load_package_copy()

    shared = [n for n, _ in CASES if hasattr(pipe, n) and hasattr(pkg, n)]
    absent = [n for n, _ in CASES if n not in shared]

    # breakdown_number takes two loss vectors, not a panel -- handled separately below.
    rng = np.random.default_rng(20260824)
    panels = _panels(a.panels, rng)

    checked = 0
    mismatch = {}
    compared = {}
    known_divergent = {}   # name -> {field: count} -- disclosed, does not fail the gate
    schema_drift = {}      # name -> (extra_a, extra_b), first occurrence only, informational
    errors = {}
    for name in shared:
        build = dict(CASES)[name]
        kw = SEEDED.get(name, {})
        for i, (L, w) in enumerate(panels):
            args = build(L, w)
            try:
                ra = getattr(pipe, name)(*args, **kw)
            except Exception as e:
                errors.setdefault(name, []).append(f"pipeline panel {i}: {type(e).__name__}: {e}")
                continue
            try:
                rb = getattr(pkg, name)(*args, **kw)
            except Exception as e:
                errors.setdefault(name, []).append(f"package panel {i}: {type(e).__name__}: {e}")
                continue
            checked += 1
            compared[name] = compared.get(name, 0) + 1
            if isinstance(ra, dict) and isinstance(rb, dict):
                # Field-level diff rather than a single opaque _same(ra, rb): KNOWN_DIVERGENT
                # is keyed by (function, field), so only a per-field comparison can tell a
                # documented divergence apart from a live mismatch.
                live, div, extra_a, extra_b = _dict_field_diff(name, ra, rb)
                if live:
                    mismatch[name] = mismatch.get(name, 0) + 1
                    if a.verbose and mismatch[name] == 1:
                        print(f"    first LIVE mismatch in {name} (panel {i}), field(s) {live}:")
                        for k in live:
                            print(f"      {k}: pipeline={ra[k]!r}  package={rb[k]!r}")
                for k in div:
                    known_divergent.setdefault(name, {})
                    known_divergent[name][k] = known_divergent[name].get(k, 0) + 1
                if (extra_a or extra_b) and name not in schema_drift:
                    schema_drift[name] = (extra_a, extra_b)
                known_drift = KNOWN_SCHEMA_DRIFT.get(name, {})
                unexpected_a = sorted(set(extra_a) - known_drift.get("pipeline_only", frozenset()))
                unexpected_b = sorted(set(extra_b) - known_drift.get("package_only", frozenset()))
                if unexpected_a or unexpected_b:
                    errors.setdefault(f"{name} schema drift", []).append(
                        f"panel {i}: undocumented extra key(s) not in KNOWN_SCHEMA_DRIFT -- "
                        f"pipeline-only={unexpected_a} package-only={unexpected_b}. If this is a "
                        f"genuine, understood divergence, add it to KNOWN_SCHEMA_DRIFT and bump "
                        f"N_KNOWN_SCHEMA_DRIFT_KEYS; otherwise this is an unreviewed schema change "
                        f"(rename, typo, dropped/added field) that must be investigated.")
            elif not _same(ra, rb):
                mismatch[name] = mismatch.get(name, 0) + 1
                if a.verbose and mismatch[name] == 1:
                    print(f"    first mismatch in {name} (panel {i}):")
                    print(f"      pipeline: {str(ra)[:160]}")
                    print(f"      package : {str(rb)[:160]}")

    # breakdown_number(la, lb, w) -- the pairwise primitive k* is built on.
    if hasattr(pipe, "breakdown_number") and hasattr(pkg, "breakdown_number"):
        for i, (L, w) in enumerate(panels):
            ms = sorted(L)
            la, lb = L[ms[0]], L[ms[1]]
            try:
                ra, rb = pipe.breakdown_number(la, lb, w), pkg.breakdown_number(la, lb, w)
            except Exception as e:
                errors.setdefault("breakdown_number", []).append(f"panel {i}: {type(e).__name__}: {e}")
                continue
            checked += 1
            compared["breakdown_number"] = compared.get("breakdown_number", 0) + 1
            if not _same(ra, rb):
                mismatch["breakdown_number"] = mismatch.get("breakdown_number", 0) + 1
        shared.append("breakdown_number")

    if hasattr(pipe, "winner_stability") and hasattr(pkg, "winner_stability"):
        for i, (L, w) in enumerate(panels):
            T = len(w)
            block = max(1, T // 3)
            try:
                kw = dict(n_boot=200, seed=0, block=block)
                ra = pipe.winner_stability(L, w, **kw)
                rb = pkg.winner_stability(L, w, **kw)
            except Exception as e:
                errors.setdefault("winner_stability[block]", []).append(f"panel {i}: {type(e).__name__}: {e}")
                continue
            checked += 1
            compared["winner_stability[block]"] = compared.get("winner_stability[block]", 0) + 1
            if not _same(ra, rb):
                mismatch["winner_stability[block]"] = mismatch.get("winner_stability[block]", 0) + 1
        shared.append("winner_stability[block]")

    if hasattr(pipe, "exchangeable_benchmark") and hasattr(pkg, "exchangeable_benchmark"):
        for K, T in [(2, 12), (3, 20), (6, 30), (6, 32), (7, 38), (10, 25)]:
            try:
                kw = dict(n_rep=20, n_boot=100, seed=0)
                ra = pipe.exchangeable_benchmark(K, T, **kw)
                rb = pkg.exchangeable_benchmark(K, T, **kw)
            except Exception as e:
                errors.setdefault("exchangeable_benchmark", []).append(f"K={K},T={T}: {type(e).__name__}: {e}")
                continue
            checked += 1
            compared["exchangeable_benchmark"] = compared.get("exchangeable_benchmark", 0) + 1
            if not _same(ra, rb):
                mismatch["exchangeable_benchmark"] = mismatch.get("exchangeable_benchmark", 0) + 1
                if a.verbose and mismatch["exchangeable_benchmark"] == 1:
                    print(f"    first mismatch in exchangeable_benchmark (K={K}, T={T}):")
                    print(f"      pipeline: {ra}")
                    print(f"      package : {rb}")
        shared.append("exchangeable_benchmark")

    K1_FUNCS = ("pooled_winner", "winner_stability", "decision_breakdown", "surprise_concentration")
    known_k1_divergent = {}
    if all(hasattr(pipe, n) and hasattr(pkg, n) for n in K1_FUNCS):
        k1_unexpected = []
        for i in range(20):
            rng1 = np.random.default_rng(9_000_001 + i)
            T = int(rng1.integers(5, 40))
            L = {"m0": np.abs(rng1.normal(1.0, 0.3, T)) + 1e-9}
            w = np.ones(T) if i % 2 else rng1.uniform(0.5, 1.5, T)
            for name in K1_FUNCS:
                args = (L,) if name == "surprise_concentration" else (L, w)
                kw = SEEDED.get(name, {})
                pipe_raised = pkg_raised = False
                try:
                    getattr(pipe, name)(*args, **kw)
                except Exception:
                    pipe_raised = True
                try:
                    getattr(pkg, name)(*args, **kw)
                except Exception:
                    pkg_raised = True
                checked += 1
                if pkg_raised and not pipe_raised:
                    known_k1_divergent[name] = known_k1_divergent.get(name, 0) + 1
                elif pipe_raised != pkg_raised:
                    # the OPPOSITE asymmetry (pipeline raises, package doesn't) or any other shape
                    # is NOT the documented divergence -- a real, unexplained difference.
                    k1_unexpected.append(f"{name} panel {i}: pipeline_raised={pipe_raised} "
                                          f"package_raised={pkg_raised}")
        if k1_unexpected:
            errors.setdefault("K=1 comparison", []).extend(k1_unexpected)
        missing = [n for n in K1_FUNCS if known_k1_divergent.get(n, 0) == 0]
        if missing:
            errors.setdefault("K=1 comparison", []).append(
                f"expected package-only-raises divergence never observed for: {missing}")
        shared.append("K=1 comparison")

    total_mm = sum(mismatch.values())
    print(f"  functions compared : {len(shared)}  ({', '.join(sorted(shared))})")
    if absent:
        print(f"  NOT in both copies : {', '.join(absent)}")
    print(f"  comparisons        : {checked}")
    for n in sorted(mismatch):
        print(f"    LIVE MISMATCH  {n}: {mismatch[n]} of {compared.get(n, a.panels)}")
    for n in sorted(known_divergent):
        for k, c in sorted(known_divergent[n].items()):
            print(f"    known-divergent  {n}.{k}: {c} of {compared.get(n, a.panels)} -- {KNOWN_DIVERGENT[n][k]}")
    for n in sorted(known_k1_divergent):
        print(f"    known-divergent  K=1.{n}: {known_k1_divergent[n]} of 20 -- package raises "
              f"(v1.0 K<2 guard), frozen pipeline copy does not; confirmed inert against every "
              f"real producer in this repo")
    for n, (extra_a, extra_b) in sorted(schema_drift.items()):
        print(f"    schema drift  {n}: pipeline-only={extra_a}  package-only={extra_b}")
    for n in sorted(errors):
        print(f"    ERROR     {n}: {len(errors[n])} (first: {errors[n][0][:110]})")

    n_known = sum(len(fields) for fields in KNOWN_DIVERGENT.values())
    ratchet_ok = n_known == N_KNOWN_DIVERGENT_FIELDS
    if not ratchet_ok:
        print(f"\n  RATCHET INCONSISTENT — N_KNOWN_DIVERGENT_FIELDS says {N_KNOWN_DIVERGENT_FIELDS}, "
              f"KNOWN_DIVERGENT now has {n_known}. A new exemption was added without bumping the "
              f"ratchet in the same edit -- confirm it's genuinely justified (a real citation to the "
              f"redesign's own source, like the existing 'fragile' entry), then bump the constant.")

    # Closes N_KNOWN_DIVERGENT_FIELDS' swap-exploit gap: a same-count substitution of WHICH
    # (function, field) pair is exempted changes this hash even when the count stays satisfied.
    known_divergent_hash = hashlib.sha256(
        ",".join(sorted(f"{fn}.{field}" for fn, fields in KNOWN_DIVERGENT.items() for field in fields)).encode()
    ).hexdigest()
    if known_divergent_hash != KNOWN_DIVERGENT_HASH:
        print(f"\n  RATCHET INCONSISTENT — KNOWN_DIVERGENT_HASH says '{KNOWN_DIVERGENT_HASH}', the "
              f"actual exempted (function, field) set hashes to '{known_divergent_hash}'")
        ratchet_ok = False

    coverage_ok = len(shared) == N_SHARED_FUNCTIONS
    if not coverage_ok:
        absent_hint = ("see 'NOT in both copies' above" if absent else
                        "none of CASES is missing, so check whether one of the 4 special-cased "
                        "comparisons (breakdown_number, winner_stability[block], "
                        "exchangeable_benchmark, K=1 comparison) silently stopped running instead")
        print(f"\n  RATCHET INCONSISTENT — N_SHARED_FUNCTIONS says {N_SHARED_FUNCTIONS}, this run "
              f"actually compared {len(shared)} ({', '.join(sorted(shared))}). Either a function "
              f"silently dropped out of one copy (rename, typo, dropped export -- exactly the "
              f"failure mode this file exists to catch, {absent_hint}) or a new one was "
              f"legitimately added -- confirm which, then update the constant.")

    # Closes N_SHARED_FUNCTIONS' swap-exploit gap: a same-count substitution of WHICH functions are
    # actually compared changes this hash even when the count stays satisfied.
    shared_functions_hash = hashlib.sha256(",".join(sorted(shared)).encode()).hexdigest()
    if shared_functions_hash != SHARED_FUNCTIONS_HASH:
        print(f"\n  RATCHET INCONSISTENT — SHARED_FUNCTIONS_HASH says '{SHARED_FUNCTIONS_HASH}', "
              f"the actual compared-function set hashes to '{shared_functions_hash}'")
        coverage_ok = False

    n_known_schema_drift = sum(
        len(v.get("pipeline_only", ())) + len(v.get("package_only", ()))
        for v in KNOWN_SCHEMA_DRIFT.values()
    )
    schema_drift_ratchet_ok = n_known_schema_drift == N_KNOWN_SCHEMA_DRIFT_KEYS
    if not schema_drift_ratchet_ok:
        print(f"\n  RATCHET INCONSISTENT — N_KNOWN_SCHEMA_DRIFT_KEYS says {N_KNOWN_SCHEMA_DRIFT_KEYS}, "
              f"KNOWN_SCHEMA_DRIFT now has {n_known_schema_drift}. A new exemption was added without "
              f"bumping the ratchet in the same edit -- confirm it's genuinely justified, then bump "
              f"the constant.")

    # Closes N_KNOWN_SCHEMA_DRIFT_KEYS' swap-exploit gap: a same-count substitution of WHICH keys
    # are exempted changes this hash even when the count stays satisfied.
    schema_drift_keys_hash = hashlib.sha256(",".join(sorted(
        f"{name}.{side}:{key}"
        for name, v in KNOWN_SCHEMA_DRIFT.items()
        for side in ("pipeline_only", "package_only")
        for key in v.get(side, ())
    )).encode()).hexdigest()
    if schema_drift_keys_hash != SCHEMA_DRIFT_KEYS_HASH:
        print(f"\n  RATCHET INCONSISTENT — SCHEMA_DRIFT_KEYS_HASH says '{SCHEMA_DRIFT_KEYS_HASH}', "
              f"the actual exempted key set hashes to '{schema_drift_keys_hash}'")
        schema_drift_ratchet_ok = False

    ok = (total_mm == 0 and not errors and ratchet_ok and coverage_ok
          and schema_drift_ratchet_ok)
    print(f"  {'PASS' if ok else 'FAIL'}  pipeline copy == released package (differential)"
          f"   {total_mm}/{checked} live mismatches")
    if not ok:
        print("  NOTE: the released package (v1.0) redesigned several fields, and the instrument")
        print("        copy is FROZEN, so divergence may be INTENDED. Read the redesign's own inline")
        print("        comment in the installed package's fragility.py before 'fixing' either side;")
        print("        if the divergence is genuinely intended, add it to KNOWN_DIVERGENT above")
        print("        rather than re-investigating it every run.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

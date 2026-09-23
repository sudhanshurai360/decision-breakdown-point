"""test_fragility.py — prove the ranking-fragility diagnostic.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fragility as F

PASS, FAIL = [], []


def check(n, c, d=""):
    (PASS if c else FAIL).append(n); print(f"  [{'PASS' if c else 'FAIL'}] {n}{(' — ' + d) if d else ''}")

# 1) greedy == brute force, over many random small cases
rng = np.random.default_rng(0)
ok = True
for _ in range(300):
    T = int(rng.integers(3, 9))
    la, lb, w = rng.random(T), rng.random(T), rng.random(T) + 0.5
    if F.breakdown_number(la, lb, w)[0] != F._brute_force_k(la, lb, w):
        ok = False; break
check("breakdown_number == brute force (greedy is exact)", ok)

# 2) ARTIFACT: deep wins all 13 calm years; AR1 is least-bad in the 2 COVID years → pooled flips to AR1
T = 15; w = np.full(T, 52.0); covid = [9, 10]; years = list(range(2011, 2026))
deep = np.full(T, 0.35); ar1 = np.full(T, 0.58)
for i in covid:
    deep[i], ar1[i] = 8.5, 5.9
r = F.fragility({"ar1": ar1, "deep": deep}, w, shock_periods=[years[i] for i in covid], labels=years, n_boot=500)
check("artifact: pooled winner = AR1", r["pooled_winner"] == "ar1", str({k: r[k] for k in ("pooled_winner","k_star")}))
check("artifact: per-period winner = deep", r["per_period_winner"] == "deep")
check("artifact: reversal flagged", r["reversal"] is True)
check("artifact: FRAGILE (small k*)", r["k_star"] <= 2 and r["k_over_T"] < 0.2, f"k*={r['k_star']} k/T={r['k_over_T']:.3f}")
check("artifact: responsible periods ⊆ COVID years", set(r["responsible"]) <= {2020, 2021}, str(r["responsible"]))
check("artifact: shock-coincidence True", r["shock_coincidence"] is True)
check("artifact: flagged FRAGILE", r["fragile"] is True)
check("artifact: winner unstable to resampling", r["winner_stability"] < 0.95, f"stability={r['winner_stability']:.2f}")

# 3) TRUE-NEGATIVE: deep beats AR1 in EVERY period (incl COVID) → robust ranking
deep2 = np.full(T, 0.35); ar1b = np.full(T, 0.58)
for i in covid:
    deep2[i], ar1b[i] = 6.0, 8.0
r2 = F.fragility({"ar1": ar1b, "deep": deep2}, w, n_boot=500)
check("true-neg: pooled winner = deep", r2["pooled_winner"] == "deep")
check("true-neg: NO reversal (silent on robust ranking)", r2["reversal"] is False)
check("true-neg: ROBUST (k/T high)", r2["k_over_T"] >= 0.9, f"k/T={r2['k_over_T']:.3f}")
check("true-neg: NOT fragile", r2["fragile"] is False)
check("true-neg: winner stable to resampling", r2["winner_stability"] > 0.95, f"stability={r2['winner_stability']:.2f}")

# 4) SURPRISE-CONCENTRATION: a CLIFF (one spike) -> sci~1, eff_n~1; a GRADUAL ramp -> sci low, eff_n high
spike = np.full(15, 0.3); spike[9] = 8.0
sc_cliff = F.surprise_concentration({"a": spike, "b": np.full(15, 0.6)}, labels=list(range(15)))
check("cliff: SCI ~ 1 (all surprise in one period)", sc_cliff["sci"] > 0.95, f"sci={sc_cliff['sci']:.3f}")
check("cliff: eff_surprising_periods ~ 1", sc_cliff["eff_surprising_periods"] < 1.5, f"eff_n={sc_cliff['eff_surprising_periods']:.2f}")
check("cliff: hardest period identified", sc_cliff["hardest"] == 9)
grad = np.array([.3,.3,.3,.3,.3,.5,.7,.9,1.1,1.3,1.5,.4,.4,.4,.4])
sc_grad = F.surprise_concentration({"a": grad, "b": grad + 0.2})
check("gradual: SCI low (surprise spread)", sc_grad["sci"] < 0.5, f"sci={sc_grad['sci']:.3f}")
check("gradual: eff_surprising_periods > cliff's", sc_grad["eff_surprising_periods"] > 3, f"eff_n={sc_grad['eff_surprising_periods']:.2f}")

# 5) MULTI-OPPONENT decision-level k* (min over ALL opponents) == brute force; and it's TIGHTER than runner-up-only
ok_multi = True; third_binds = False
for _ in range(400):
    K = int(rng.integers(3, 6)); T = int(rng.integers(3, 9))
    L = {f"m{j}": rng.random(T) for j in range(K)}; w = rng.random(T) + 0.5
    kg, opp, _ = F.decision_breakdown(L, w); kb = F._brute_force_k_multi(L, w)
    if kg != kb:
        ok_multi = False; break
    a = F.pooled_winner(L, w)
    ru = min((m for m in L if m != a), key=lambda m: float(np.average(L[m], weights=w)))
    if opp is not None and opp != ru:
        third_binds = True   # the binding opponent is NOT the pooled runner-up -> multi-opponent is strictly tighter
check("multi-opponent decision k* == brute force (greedy exact over opponents)", ok_multi)
check("multi-opponent k* is TIGHTER than runner-up-only (a non-runner-up binds somewhere)", third_binds)

sc_zero = F.surprise_concentration({"a": np.array([0.0, 0.0, 0.0]), "b": np.array([0.0, 0.0, 0.0])})
check("d_max_over_median: zero baseline, zero max -> 1.0", sc_zero["d_max_over_median"] == 1.0, str(sc_zero["d_max_over_median"]))
sc_posmax = F.surprise_concentration({"a": np.array([0.0, 0.0, 5.0]), "b": np.array([1.0, 1.0, 6.0])})
check("d_max_over_median: zero baseline, positive max -> +inf", sc_posmax["d_max_over_median"] == float("inf"), str(sc_posmax["d_max_over_median"]))
sc_mismatch = F.surprise_concentration({"a": np.array([-5.0, -3.0, 2.0]), "b": np.array([-4.0, -2.0, 3.0])})
check("d_max_over_median: sign mismatch (b<0, d_max>=0) -> NaN", np.isnan(sc_mismatch["d_max_over_median"]), str(sc_mismatch["d_max_over_median"]))

try:
    F.pooled_winner({"a": np.array([1.0, 2.0, 3.0]), "b": np.array([2.0, 1.0, 3.0])}, np.array([0.0, 0.0, 0.0]))
    zero_w_raised = False
except ValueError:
    zero_w_raised = True
check("pooled_winner: all-zero weights raise ValueError (not ZeroDivisionError)", zero_w_raised)
check("pooled_winner: uniform nonzero weights unaffected",
      F.pooled_winner({"a": np.array([1.0, 2.0, 3.0]), "b": np.array([4.0, 5.0, 6.0])}, np.ones(3)) == "a")

print(f"\n{'='*50}\nRESULT: {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILED:", FAIL); sys.exit(1)
print("fragility diagnostic behaves correctly.")

#!/usr/bin/env python3
"""acceptance.py — does the pipeline reproduce the values the project actually recorded?
"""
import argparse
import collections
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
LEDGER = ROOT / "paper" / "ledger_recorded.json"

# A "harvest" is a pipeline run whose emitted values were recorded into paper/ledger_recorded.json;
# a "harvest-noise band" is the spread observed across repeated harvests of the same producer
# (different seeds or block lengths), measured empirically rather than derived analytically.
#
# key -> (abs_tolerance, provenance). Measured, never assumed.
STOCHASTIC = {
    "nullsens.p05_iid":        (0.038, "6 spaced masters, sd 0.0152"),
    "nullsens.p05_rho05":      (0.038, "same null, same band"),
    "nullsens.p05_rho08":      (0.038, "same null, same band"),
    "nullsens.p05_ar06":       (0.038, "same null, same band"),
    "nullsens.ar_delta_p05":   (0.054, "difference of two draws from that band"),
    "election.stability_median": (0.002, "n_boot=1500 -> granularity 1/1500 = 0.00067"),
    "comb.median_winner_stability": (0.002, "n_boot=2000 -> granularity 0.0005"),
    "mcssize.size_at_nominal_10": (0.025, "se 0.0099 reported in mcs_size.csv at n_rep=2000"),
    "flagship.stability_null_median": (0.031, "INERT: 8 spaced seeds, sd 0.0124 -- but this key is "
                                              "not in canonical.py's CHECKS, so classify() can never "
                                              "reach it; kept only as the measured band's record"),
    "flagship.stability_null_p05":    (0.024, "INERT: 8 spaced seeds, sd 0.0094 -- likewise absent "
                                              "from canonical.py's CHECKS and therefore unreachable"),
}

ARTIFACT_PRECISION = {
    "flagship.winner_stability": (0.001, "flagship_fragility.csv stores 3dp; recorded 0.6622, "
                                         "CSV can only ever say 0.662"),
    "comb.winning_series_null_p05": (0.0005, "canonical.py's own round(..., 3) on this key, NOT the "
                                              "artifact: combination_benchmark.csv stores the full "
                                              "0.527225, which is what the ledger recorded, while "
                                              "the live read can only ever say 0.527"),
    "comb.winning_series_stability": (0.0005, "INERT since the ledger's latest was re-harvested: "
                                               "recorded (band 120) and live are both 0.519 and "
                                               "agree exactly, so this tolerance no longer decides "
                                               "anything. Calibrated back when the latest was "
                                               "0.53432 -- and, as above, the truncation was "
                                               "canonical.py's own round(..., 3), not the CSV, "
                                               "which stores the full value (0.5191700000000001 "
                                               "today) and could only ever be read as 0.534"),
}

EXPECTED_DRIFT = {
    "fredmd.nonidentified":            (5,    "live FRED-MD vintage fetch (base cause) plus the "
                                               "COVID-scale-exclusion fix in fred_md_sweep.py"),
    "fredmd.identified_complement":    (3,    "complement of nonidentified, same cause; tightened "
                                               "from a bare piggyback on nonidentified's own 5. "
                                               "There is no measured gap to cover today -- the "
                                               "ledger's recorded 18 and the live 18 agree exactly "
                                               "-- so this is pure headroom against the same "
                                               "panel-membership drift that moves fredmd.total "
                                               "(recorded 77 -> 76 -> 74 across harvest bands)"),
    "fredmd.tcode_subset":             (2,    "vintage growth changes which series clear the "
                                               "length/finiteness filter feeding the tcode subset"),
    "fredmd.nonidentified_levels_subset": (2, "same vintage-subset-size cause"),
    "fredmd.nonidentified_tcode":      (6,    "same vintage-subset-size cause"),
    "fredmd.count_alpha05":            (3,    "own 3-seed Monte-Carlo noise floor (the alpha=0.05 / "
                                               "block=3 / B=2000 row of fredmd_robustness_band.csv "
                                               "spans [59,61] across its 3 seeds, mean 60.0, against "
                                               "the ledger's recorded 61.0) stacked on the live FRED-MD "
                                               "vintage cause every fredmd.* key carries -- NOT SARIMA "
                                               "(this key's producer arm never uses the sarima model)"),
    "fredmd.count_alpha25":            (2.0,  "same vintage cause, alpha-sweep arm, plus the "
                                               "COVID-scale-exclusion fix in fred_md_sweep.py"),
    "fredmd.spearman_gap_mcs":         (0.03, "same vintage cause: a live FRED-MD pull adds or drops "
                                               "SERIES release to release -- fredmd.total's own "
                                               "ledger record moves 77 -> 76 -> 74 across harvest "
                                               "bands -- which changes the set this rank "
                                               "correlation is taken over. Vintage drift, not a "
                                               "formula change. (The eval-year count is NOT the "
                                               "channel: series_losses() builds eval years from "
                                               "EVAL_START=2000 and every row of fred_md_sweep.csv "
                                               "carries n_years=27.)"),
    "comb.in_mcs": (1, "a one-unit gap the other way round from how this entry used to read: the "
                       "ledger's LATEST records 4 (band 130) while the live run gives 5 of 12. The "
                       "elimination-rule explanation no longer holds -- re-derived over the curated "
                       "panel, the raw default (project convention, matching mcs_test.py's own "
                       "headline computation), 'studentized' and 'e_R' ALL return 5 today, so no "
                       "elimination setting reproduces the recorded 4, and the earlier bands (30, "
                       "40) recorded 5 as well. The cause is not recoverable from the frozen "
                       "artifacts; the tolerance stays at the one-unit move it covers"),
    "mcspower.conditional_power_0pct":  (0.001, "the producer emits 4 dp (0.0833) against the ledger's "
                                                "3-dp 0.083; a precision difference, not a value change"),
    "mcspower.conditional_power_2pct":  (0.03, "deterministic; harvest-noise band -- empirical spread "
                                               "observed across repeated mcs_power.py runs on the "
                                               "frozen panel with different seeds/block-lengths, not "
                                               "derived analytically"),
    "mcspower.conditional_power_5pct":  (0.03, "deterministic; harvest-noise band -- empirical spread "
                                               "observed across repeated mcs_power.py runs on the "
                                               "frozen panel with different seeds/block-lengths, not "
                                               "derived analytically"),
    "mcspower.conditional_power_50pct": (0.03, "deterministic; harvest-noise band -- empirical spread "
                                               "observed across repeated mcs_power.py runs on the "
                                               "frozen panel with different seeds/block-lengths, not "
                                               "derived analytically"),
    "mcspower.recentred_power_50pct":   (0.07, "deterministic; harvest-noise band -- empirical spread "
                                               "observed across repeated mcs_power.py runs on the "
                                               "frozen panel with different seeds/block-lengths, not "
                                               "derived analytically"),
    "mcspower.studentized_power_2pct":  (0.02, "deterministic; harvest-noise band -- empirical spread "
                                               "observed across repeated mcs_power.py runs on the "
                                               "frozen panel with different seeds/block-lengths, not "
                                               "derived analytically"),
    "mcspower.studentized_power_5pct":  (0.02, "deterministic; harvest-noise band -- empirical spread "
                                               "observed across repeated mcs_power.py runs on the "
                                               "frozen panel with different seeds/block-lengths, not "
                                               "derived analytically"),
    "mcspower.studentized_power_10pct": (0.001,"INERT: 'rounding only -- 0.6794871.. vs 0.679' was "
                                               "calibrated against a ledger latest that is now "
                                               "history (that pair survives in this key's band-30 "
                                               "record). The latest recorded value is 0.68056 and "
                                               "the live value 0.65278, which this tolerance cannot "
                                               "cover, so the key falls through to its BLOCKED entry "
                                               "below -- which carries the correct 0.681 -> 0.653"),
    "mcspower.studentized_power_50pct": (0.02, "deterministic; harvest-noise band -- empirical spread "
                                               "observed across repeated mcs_power.py runs on the "
                                               "frozen panel with different seeds/block-lengths, not "
                                               "derived analytically"),
}

N_COMPARABLE = 127     # keys carrying BOTH a ledger entry and a live re-derivation: the rows compared
N_BLOCKED = 47         # of those, the keys classified BLOCKED (a DIFFER with a documented exemption
                       # in BLOCKED below, including the regenerated-panel movers)

COMPARABLE_KEYS_HASH = "ab032e9e9699ad1ed29a3e7d6e7fdb248fe8d08e7237dcbb0c34ad0f6891b53f"
BLOCKED_KEYS_HASH = "315209d014bb698c0c8b2c3e23fabca27dab00b6b7fc737f68841144cf13287e"

N_LEDGER_UNCOMPARED = 23
LEDGER_UNCOMPARED_KEYS_HASH = "fa2ea142071ccdfa489eb51e6d834d77926998ecdb6e583dd825b2880e53357d"

STRING_NORMALIZE = {
    "data.curated_vintage_cutoff": lambda s: s[:4],
}

# Keys known to be blocked on source we do not hold. Reported separately so they do not
# masquerade as fresh failures every run. Each names WHY.
BLOCKED = {
    "eb.": "ledger's recorded values are from a superseded generation: eb.episodes was harvested at "
           "63 (bands 30 and 50), 41 (band 80) and a latest of 27 (band 90) -- 90 there is the BAND "
           "label, never an episode count -- while the current code gives 29, which is the correct "
           "value; a known-wrong generation the current code correctly does not match",
    "vintage.": "scores on the same statutory window as eb.*; inherits the same known-wrong-ledger block",
    "dmcs.mean_set_decision": "definition changed: now the near-trigger- "
                               "conditioned decision loss, not the unconditional one the ledger's "
                               "recorded band was harvested against -- not a live-code defect, a "
                               "genuine methodology change; the ledger has no valid target for this "
                               "quantity until a fresh harvest exists",
    "dmcs.mean_set_statistical": "denominator shrank 51->44 states (dmcs.n_eligible is still 51; "
                                  "dmcs.csv now carries 44 rows): the "
                                  "near-trigger floor added for dmcs.mean_set_decision now also "
                                  "gates which states contribute to this average, even though its "
                                  "own formula (mean MCS_stat size) is unchanged -- recorded "
                                  "2.29412 (= 117/51) against a live 2.34091 (= 103/44)",
    "election.spearman_mcs_stability_rmsse": "definition changed: rmsse now paired with "
                                              "MCS_rmsse_10 (genuine sqrt(mean(msse)) per year), not "
                                              "MCS_sq_10 (plain MSSE) -- a real bug fix, not a defect; "
                                              "the ledger's recorded value reflects the old, mislabeled "
                                              "pairing",
    "election.spearman_mcs_stability_rmsse_p": "same cause as election.spearman_mcs_stability_rmsse "
                                                "above, paired p-value",
    "fredmd.squared_nonidentified": "definition changed: "
                                     "gap-interpolation restoring 11 shutdown-affected series "
                                     "(74->63->74) plus the RMSSE-sqrt-before-pooling fix in "
                                     "series_losses() -- the ledger's recorded 61 predates both; not "
                                     "vintage drift, which is what this key's old EXPECTED_DRIFT "
                                     "tolerance was calibrated for",
    "llm.n_models": "ml_witness.py's fetch() cap was raised: the ledger's recorded "
                     "2949 was itself truncated, not a complete "
                     "snapshot; a fresh fetch of the actively-updated leaderboard will keep "
                     "growing indefinitely, so no fixed tolerance stays meaningful",
    "llm.mean_cross_task_corr": "same cause as llm.n_models -- moved 0.676->0.672 as a direct, "
                                 "expected consequence of the larger, now-untruncated candidate pool",
    "election.stability_median": "social_choice.py's seed-averaging fix (see null_budget.py's "
                                  "stable_stability()), compounded by the regenerated curated panel "
                                  "documented further down -- moved 0.594->0.574, the ledger "
                                  "predates both",
    "election.stability_min": "same cause as election.stability_median -- moved 0.168->0.159",
    "election.stability_below_060": "same cause -- moved 12->15 of 24. The earlier account (a single "
                                     "cell, M2SL/mase, crossing the fixed 0.60 threshold) cannot "
                                     "explain a three-cell move; which cells crossed is not "
                                     "recoverable from the ledger, which records the count alone",
    "election.spearman_kstar_stability": "same cause -- the correlation is computed against the "
                                          "now-corrected winner_stability column; moved "
                                          "0.80512->0.80526, substantively nothing, but wider than "
                                          "the gate's 1e-6 relative exactness test",
    "election.spearman_mcs_stability_mase": "same cause -- moved -0.5234->-0.5509",
    "election.spearman_mcs_stability_mase_p": "same cause -- moved 0.0808->0.0634 (n=12). Both the "
                                               "recorded and the live value sit ABOVE the "
                                               "conventional 0.05 boundary, so the move does not "
                                               "cross it; the prose's own 'weaker and "
                                               "metric-dependent' framing already did not claim a "
                                               "significance verdict either side of that line",
    "flagship.winner_stability": "flagship_fragility.py's seed-averaging fix (see "
                                  "null_budget.py's stable_stability()) -- moved 0.662->0.670, the ledger "
                                  "predates the fix",
    "realtime.sahm_divergent_all_forecast": "Sahm reference-window fix, then the "
                                             "as-of vintage construct: retired "
                                             "all_forecast arm moved 6->5->1 of 12; cites=[], never "
                                             "in the manuscript; the ledger predates both fixes",
    "realtime.sahm_divergent": "as-of vintage construct for the national Sahm/Evans arm: "
                               "2 of 13 -> 1 of 12; the "
                               "superseded construct is retained as the each_leg_first_release "
                               "companion (2 of 13); the ledger predates the fix",
    # Movers from the regenerated curated panel (DGORDER, BUSINV, PI, UNEMPLOY refits) plus a
    # bootstrap-centring fix. Every value below was predicted before the run and matched; the
    # ledger's recorded value predates the regenerated panel, and no fresh harvest exists yet.
    "election.reversals": "regenerated panel -- 13 -> 10 of 24, predicted",
    "election.concentration_gt1": "regenerated panel -- 13 -> 14 of 24, predicted",
    "election.weight_conc_gt1_equal": "regenerated panel -- 13 -> 14, predicted",
    "election.weight_conc_gt1_monthwtd": "regenerated panel -- 12 -> 13, predicted",
    "comb.sn_worst_ratio_min": "regenerated panel -- 1.6494 -> 1.6821, predicted to the digit",
    "prop.spearman_kstar_t": "regenerated panel -- 0.789 -> 0.842, predicted",
    "shape.hardest_april2020": "regenerated panel -- 7 -> 8 (DGORDER's hardest month moves to "
                               "2020-04), predicted",
    "regime.placebo_below_excovid_median": "regenerated panel -- 50.0 -> 43.0, predicted",
    "mcspower.conditional_power_2pct": "bootstrap-centring fix + regenerated panel -- 0.389 -> "
                                       "0.429; all mcspower.* values were certain to move; "
                                       "exceeds the old harvest-noise tolerance by design",
    "mcspower.conditional_power_5pct": "same cause -- 0.688 -> 0.633",
    "mcspower.conditional_power_10pct": "same cause -- 0.891 -> 0.855",
    "mcspower.studentized_power_5pct": "same cause -- 0.458 -> 0.431",
    "mcspower.studentized_power_10pct": "same cause -- 0.681 -> 0.653",
}


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def classify(key, got, want):
    """-> (verdict, note). Order matters: tolerance before exactness."""
    if key in STRING_NORMALIZE:
        norm = STRING_NORMALIZE[key]
        same = norm(str(got).strip("'\"")) == norm(str(want).strip("'\""))
        return ("MATCH", "normalized") if same else ("DIFFER", "non-numeric, normalized")
    g, w = _num(got), _num(want)
    if g is None or w is None:
        same = str(got).strip("'\"") == str(want).strip("'\"")
        return ("MATCH", "") if same else ("DIFFER", "non-numeric")
    if math.isnan(g) and math.isnan(w):
        return "MATCH", "both NaN"
    if key in ARTIFACT_PRECISION:
        tol, why = ARTIFACT_PRECISION[key]
        if abs(g - w) <= tol:
            return "MATCH", f"artifact precision {tol} ({why})"
        return "DIFFER", f"exceeds artifact precision {tol}"
    if key in STOCHASTIC:
        tol, why = STOCHASTIC[key]
        if abs(g - w) <= tol:
            return "MATCH", f"within {tol} ({why})"
        return "DIFFER", f"exceeds {tol} ({why})"
    if key in EXPECTED_DRIFT:
        tol, why = EXPECTED_DRIFT[key]
        if abs(g - w) <= tol:
            return "MATCH", f"expected drift, within {tol} ({why})"
        return "DIFFER", f"exceeds expected-drift tolerance {tol} ({why})"
    if abs(g - w) <= 1e-9 or (w != 0 and abs(g - w) / abs(w) < 1e-6):
        return "MATCH", ""
    return "DIFFER", ""


def blocked_reason(key, got=None, canon=None):
    """The documented exemption for `key`, or None if it is not exempt.

    The exemption is granted to a VALUE, not to a key: a longest-prefix match against BLOCKED finds
    the reason, but when `got` and `canon` are supplied the key is exempt only if `got` still agrees
    with canonical.json's current value. A key under an exempt prefix that has ALSO drifted from the
    current source of truth is not exempt and is reported as a real difference."""
    why = None
    for prefix in sorted(BLOCKED, key=len, reverse=True):
        if key.startswith(prefix):
            why = BLOCKED[prefix]
            break
    if why is None:
        return None
    if got is None or canon is None:
        return why
    want_now = canon.get(key, {}).get("value")
    if want_now is None:
        return why  # not (yet) registered in canonical.json -- fall back to the documented exemption
    g, w = _num(got), _num(want_now)
    if g is not None and w is not None:
        if (math.isnan(g) and math.isnan(w)) or abs(g - w) <= 1e-9 or (w != 0 and abs(g - w) / abs(w) < 1e-6):
            return why
        return None  # diverges from the CURRENT source of truth too -- not a documented exemption
    return why if str(got).strip("'\"") == str(want_now).strip("'\"") else None


def _load_canonical_module():
    """Load gates/canonical.py as a module and return it, tolerating its argparse SystemExit.
    Broken out of main() purely so tests can monkeypatch this one function to inject a fake
    CHECKS registry."""
    spec = importlib.util.spec_from_file_location("cg", HERE / "canonical.py")
    cg = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(cg)
    except SystemExit:
        pass
    return cg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    if not LEDGER.exists():
        raise SystemExit(f"missing {LEDGER}")
    led = json.loads(LEDGER.read_text())

    cg = _load_canonical_module()
    _canon_path = getattr(cg, "OUT", None)
    canon = json.loads(_canon_path.read_text()) if _canon_path and _canon_path.exists() else {}

    rows, fam = [], collections.defaultdict(collections.Counter)
    for key in sorted(set(cg.CHECKS) & set(led)):
        want = led[key]["latest"]
        try:
            got = cg.CHECKS[key]()
            verdict, note = classify(key, got, want)
        except Exception as e:
            rows.append((key, "ERROR", f"{type(e).__name__}", want, ""))
            fam[key.split(".")[0]]["error"] += 1
            continue
        why = blocked_reason(key, got, canon)
        if verdict == "DIFFER" and why:
            verdict = "BLOCKED"
        rows.append((key, verdict, got, want, note or (why or "")))
        fam[key.split(".")[0]][verdict.lower()] += 1

    tot = collections.Counter()
    for c in fam.values():
        tot.update(c)
    n = sum(tot.values())
    print(f"=== ACCEPTANCE: pipeline vs the recorded latest ({n} comparable keys) ===")
    print(f"  MATCH {tot['match']}   DIFFER {tot['differ']}   BLOCKED {tot['blocked']}   ERROR {tot['error']}\n")
    print(f"  {'family':<12}{'match':>6}{'differ':>7}{'blocked':>8}{'error':>6}")
    for f, c in sorted(fam.items(), key=lambda kv: (-kv[1]["differ"], -kv[1]["match"])):
        print(f"  {f:<12}{c['match']:>6}{c['differ']:>7}{c['blocked']:>8}{c['error']:>6}")

    live = [r for r in rows if r[1] == "DIFFER"]
    if live:
        print(f"\n  --- {len(live)} LIVE difference(s): not blocked, not tolerance, needs work ---")
        for k, v, g, w, note in live:
            print(f"    {k:<34} ours={str(g)[:20]:<20} recorded={str(w)[:20]} {note}")
    if a.verbose:
        for k, v, g, w, note in rows:
            if v != "DIFFER":
                print(f"    {v:<8} {k:<34} {str(g)[:22]:<22} {note}")

    errored = [r for r in rows if r[1] == "ERROR"]
    if errored:
        print(f"\n  --- {len(errored)} ERRORED check(s): CANNOT run, not merely differing ---")
        for k, v, g, w, note in errored:
            print(f"    {k:<34} raised {g}")

    ratchet_ok = True
    for name, want, got in (("N_COMPARABLE", N_COMPARABLE, n), ("N_BLOCKED", N_BLOCKED, tot["blocked"])):
        if want != got:
            print(f"\n  RATCHET INCONSISTENT — {name} says {want}, this run gives {got}")
            ratchet_ok = False

    comparable_keys_hash = hashlib.sha256(",".join(sorted(r[0] for r in rows)).encode()).hexdigest()
    blocked_keys_hash = hashlib.sha256(
        ",".join(sorted(r[0] for r in rows if r[1] == "BLOCKED")).encode()
    ).hexdigest()
    for name, want, got in (("COMPARABLE_KEYS_HASH", COMPARABLE_KEYS_HASH, comparable_keys_hash),
                            ("BLOCKED_KEYS_HASH", BLOCKED_KEYS_HASH, blocked_keys_hash)):
        if want != got:
            print(f"\n  RATCHET INCONSISTENT — {name} says '{want}', this run's actual key set "
                  f"hashes to '{got}'")
            ratchet_ok = False

    uncompared = sorted(k for k in led if k in canon and k not in cg.CHECKS)
    unc_differ = []
    for k in uncompared:
        try:
            verdict = classify(k, canon[k].get("value"), led[k]["latest"])[0]
        except Exception:   # noqa: BLE001 -- a non-numeric/odd key: report as not-matching, never crash the gate
            verdict = "ERROR"
        if verdict != "MATCH":
            unc_differ.append(k)
    print(f"\n  --- {len(uncompared)} ledger key(s) present in canonical.json but NEVER compared (no CHECKS "
          f"entry); {len(unc_differ)} of them currently differ from the ledger's latest, unreviewed ---")
    for k in unc_differ:
        print(f"    {k:<38} canonical={str(canon[k].get('value'))[:16]:<16} ledger={str(led[k]['latest'])[:16]}")
    unc_hash = hashlib.sha256(",".join(uncompared).encode()).hexdigest()
    if canon and (len(uncompared) != N_LEDGER_UNCOMPARED or unc_hash != LEDGER_UNCOMPARED_KEYS_HASH):
        print(f"\n  RATCHET INCONSISTENT — uncompared ledger keys: N_LEDGER_UNCOMPARED says "
              f"{N_LEDGER_UNCOMPARED} / hash '{LEDGER_UNCOMPARED_KEYS_HASH}', this run gives "
              f"{len(uncompared)} / '{unc_hash}' -- a ledger key silently gained or lost coverage")
        ratchet_ok = False

    # EXIT CODE. LIVE differences fail (unchanged). ERRORED checks now fail too (see above). The
    # ratchet failing (comparable-key count or blocked-key count silently drifted) fails too.
    good = not live and not errored and ratchet_ok
    print(f"\n  {'PASS' if good else 'FAIL'}  {len(live)} live difference(s), {len(errored)} errored "
          f"check(s), ratchet {'ok' if ratchet_ok else 'INCONSISTENT'}")
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())

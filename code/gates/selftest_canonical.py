"""selftest_canonical.py — prove `canonical.py --check` can actually FAIL.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
import canonical as C  # noqa: E402 -- needed after sys.path.insert, for C._digest below

CANON = ROOT / "paper" / "canonical.json"
GATE = HERE / "canonical.py"
SIDECAR = CANON.with_suffix(".json.selftest_backup")
SENTINEL = 987654
KNOWN_GENUINE_CHECK_KEY = "mcspower.n_trials"
KNOWN_UNREGISTERED_KEY = "arena.llm_flip_null_max_pool"


def _gate_check():
    """Run the gate. Returns True if it PASSES."""
    r = subprocess.run([sys.executable, str(GATE), "--check"],
                       cwd=ROOT, capture_output=True, text=True)
    return r.returncode == 0, r.stdout + r.stderr


def _recover_interrupted() -> bool:
    """Returns True iff it is safe for main() to proceed (no sidecar found, or a found sidecar
    restored and verified byte-identical)."""
    if SIDECAR.exists():
        print("  ! sidecar present at startup — an earlier run was interrupted mid-mutation.")
        sidecar_bytes = SIDECAR.read_bytes()
        shutil.copy2(SIDECAR, CANON)
        if CANON.read_bytes() == sidecar_bytes:
            SIDECAR.unlink()
            print("  ! canonical.json restored from sidecar before starting.")
        else:
            print("  ! WARNING: restore from sidecar did NOT verify byte-identical -- the sidecar "
                  "has been PRESERVED, not deleted. Do not delete it manually until "
                  f"{CANON.relative_to(ROOT)} is confirmed correct.")
            return False
    return True


def main():
    if not CANON.exists():
        print(f"  SKIP  {CANON.relative_to(ROOT)} does not exist — run canonical.py first")
        return 2
    if not _recover_interrupted():
        print("  ABORT  a leftover sidecar from an earlier interrupted run did not restore "
              "byte-identical -- fix that manually before running this script again.")
        return 2

    original = CANON.read_bytes()
    doc = json.loads(original)
    payload = [k for k in doc if not k.startswith("_")]
    if not payload:
        print("  SKIP  canonical.json has no values to mutate")
        return 2

    passed_clean, out = _gate_check()
    if not passed_clean:
        print("  ABORT  the gate does not pass on the UNMUTATED file, so a later FAIL would prove")
        print("         nothing. Fix that first. Gate said:")
        for l in out.strip().split("\n")[:6]:
            print("           " + l.strip())
        return 2

    # Each mutation must make the gate FAIL. A mutation the gate misses is a hole in the gate.
    key = sorted(payload)[0]
    numeric = [k for k in payload if isinstance(doc[k].get("value"), (int, float))
               and not isinstance(doc[k].get("value"), bool)]
    mutations = [
        ("planted sentinel in a value",
         lambda d: d[numeric[0]].__setitem__("value", SENTINEL) if numeric else None),
        ("value perturbed by 1e-3",
         lambda d: d[numeric[0]].__setitem__("value", float(d[numeric[0]]["value"]) + 1e-3) if numeric else None),
        ("value replaced with a string",
         lambda d: d[key].__setitem__("value", "not-a-number")),
        ("a value deleted outright",
         lambda d: d.pop(key)),
        ("hash re-stamped to a lie",
         lambda d: d["_meta"].__setitem__("hash", "0" * 64)),
    ]
    if not numeric:
        mutations = [m for m in mutations if "value" not in m[0] or "string" in m[0] or "deleted" in m[0]]

    if KNOWN_GENUINE_CHECK_KEY not in payload or KNOWN_GENUINE_CHECK_KEY not in C.CHECKS:
        print(f"  ABORT  KNOWN_GENUINE_CHECK_KEY={KNOWN_GENUINE_CHECK_KEY!r} is missing from "
              f"canonical.json and/or canonical.CHECKS -- the CHECKS-only mutation below cannot "
              f"run. Pick a new known-independent (non-readback) key and update this constant.")
        return 2

    def _hash_consistent_corrupt(d):
        old = float(d[KNOWN_GENUINE_CHECK_KEY]["value"])
        # +5% or +1, whichever is larger -- well outside _close()'s max(1e-9, |b|*1e-6) tolerance
        d[KNOWN_GENUINE_CHECK_KEY]["value"] = old + max(1.0, abs(old) * 0.05)
        d["_meta"]["hash"] = C._digest(d)
    mutations.append((
        f"{KNOWN_GENUINE_CHECK_KEY} corrupted, hash re-stamped to match (CHECKS-only test)",
        _hash_consistent_corrupt,
    ))

    def _hash_consistent_companion_corrupt(d):
        d[KNOWN_GENUINE_CHECK_KEY]["n_panels"] = 999999   # a companion field, not `value`
        d["_meta"]["hash"] = C._digest(d)
    mutations.append((
        f"{KNOWN_GENUINE_CHECK_KEY}.n_panels corrupted (companion field, not value), hash "
        f"re-stamped to match",
        _hash_consistent_companion_corrupt,
    ))

    if KNOWN_UNREGISTERED_KEY not in payload or KNOWN_UNREGISTERED_KEY in C.CHECKS:
        print(f"  ABORT  KNOWN_UNREGISTERED_KEY={KNOWN_UNREGISTERED_KEY!r} is missing from "
              f"canonical.json and/or now HAS a CHECKS[] entry -- the SENTINEL-scan-only mutation "
              f"below cannot run as designed. Pick a new key that is present in canonical.json but "
              f"absent from canonical.CHECKS, and update this constant.")
        return 2

    def _hash_consistent_sentinel_leak(d):
        d[KNOWN_UNREGISTERED_KEY]["value"] = SENTINEL
        d["_meta"]["hash"] = C._digest(d)
    mutations.append((
        f"{KNOWN_UNREGISTERED_KEY} (unregistered) set to SENTINEL, hash re-stamped to match "
        f"(SENTINEL-scan-only test)",
        _hash_consistent_sentinel_leak,
    ))

    def _hash_consistent_unregistered_delete(d):
        d.pop(KNOWN_UNREGISTERED_KEY)
        d["_meta"]["hash"] = C._digest(d)
    mutations.append((
        f"{KNOWN_UNREGISTERED_KEY} (unregistered) deleted outright, hash re-stamped to match "
        f"(N_VALUES-ratchet-only test)",
        _hash_consistent_unregistered_delete,
    ))

    RENAMED_PROBE_KEY = "audit.selftest_renamed_key_probe_do_not_use"

    def _hash_consistent_key_rename(d):
        assert RENAMED_PROBE_KEY not in d, f"{RENAMED_PROBE_KEY!r} unexpectedly already exists"
        d[RENAMED_PROBE_KEY] = d.pop(KNOWN_UNREGISTERED_KEY)
        d["_meta"]["hash"] = C._digest(d)
    mutations.append((
        f"{KNOWN_UNREGISTERED_KEY} renamed to a synthetic key (same payload count), hash "
        f"re-stamped to match (VALUES_KEYS_HASH-ratchet-only test)",
        _hash_consistent_key_rename,
    ))

    caught = missed = 0
    try:
        shutil.copy2(CANON, SIDECAR)          # external restore point BEFORE any mutation
        for label, mutate in mutations:
            d = json.loads(original)
            mutate(d)
            CANON.write_text(json.dumps(d, indent=2, sort_keys=True, default=str) + "\n")
            ok, _ = _gate_check()
            if ok:
                missed += 1
                print(f"    MISSED   the gate PASSED with: {label}")
            else:
                caught += 1
                print(f"    caught   {label}")
    finally:
        CANON.write_bytes(original)
        restored_identical = CANON.read_bytes() == original
        if restored_identical:
            if SIDECAR.exists():
                SIDECAR.unlink()
        elif SIDECAR.exists():
            print(f"  WARNING  canonical.json restore did not verify byte-identical -- "
                  f"{SIDECAR.relative_to(ROOT)} was preserved (NOT deleted) for manual/next-run "
                  f"recovery. Do not delete it until paper/canonical.json is confirmed correct.")

    print(f"  {caught}/{caught + missed} planted defects caught")
    print(f"  restored byte-identical: {restored_identical}")
    good = missed == 0 and restored_identical
    print(f"  {'PASS' if good else 'FAIL'}  the gate fails when the results are wrong")
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())

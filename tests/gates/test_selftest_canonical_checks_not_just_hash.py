"""Regression test: selftest_canonical.py's
"N/N planted defects caught" score used to prove nothing about whether canonical.py's
independent re-derivations (CHECKS) actually work. All 5 original mutations left
doc["_meta"]["hash"] stale (computed from the pre-mutation payload), so every one of them was
caught via the cheap `hash_ok = False` path alone -- even if every CHECKS entry were deleted, or
replaced with a pure readback of the stored value, this file would still have reported a perfect
score.

The fix added one more mutation that corrupts a KNOWN-GENUINE, non-readback registered check
(mcspower.n_trials, which reads breadth_distinct_predictions.csv directly rather than
canonical.json's own stored value) and re-stamps _meta.hash to match the corruption -- exactly
what a real corrupted producer CSV, re-merged, would look like. This isolates CHECKS as the only
thing that can still catch it.

Verified by hand (not re-verified by this test, since it requires mutating the real
code/gates/canonical.py source on disk and shelling out -- too invasive for routine CI): swapping
mcspower.n_trials's CHECKS entry for a pure `json.loads(OUT.read_text())[...]` readback made this
specific new mutation MISSED (5/6, FAIL) while the clean file scores 6/6 (PASS) -- confirming the
new mutation actually discriminates real re-derivation from readback, not just another way to
trip the hash check.
"""
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
CANON = ROOT / "paper" / "canonical.json"
SCRIPT = ROOT / "code" / "gates" / "selftest_canonical.py"

pytestmark = pytest.mark.skipif(not CANON.exists(), reason="paper/canonical.json not available")


def _run():
    r = subprocess.run([sys.executable, str(SCRIPT)], cwd=ROOT, capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def test_selftest_includes_a_hash_consistent_mutation():
    """The mutation list must contain at least one entry whose label says it re-stamped the hash
    to match -- i.e. a mutation that can ONLY be caught by CHECKS, not by the cheap hash-mismatch
    path every other mutation relies on."""
    rc, out = _run()
    assert rc == 0, f"selftest_canonical.py did not pass cleanly:\n{out}"
    assert "hash re-stamped to match" in out
    assert "MISSED" not in out


def test_selftest_restores_canonical_json_byte_identical():
    before = CANON.read_bytes()
    rc, out = _run()
    after = CANON.read_bytes()
    assert rc == 0
    assert before == after
    assert "restored byte-identical: True" in out


def test_selftest_includes_a_companion_field_mutation():
    """Every mutation before this one
    corrupted `value`; none corrupted a COMPANION field (n/of/band/...) while re-stamping the hash
    to match. Before the matching fix in canonical.py (_mcspower_n_panels(), the new
    companion-field check in check()), that exact mutation shape was a live, reproducible blind
    spot -- no CHECKS[] entry re-derived anything but `value`, so it passed cleanly. This asserts
    the mutation exists, is labeled correctly, and is actually caught (not MISSED).

    The expected score below tracks the mutation list: each of the SENTINEL-scan-only, the
    N_VALUES-ratchet-only and the VALUES_KEYS_HASH-ratchet-only mutations (each with its own test
    in this file) is a distinct defect class, and adding one raises the expected N/N rather than
    signalling a regression."""
    rc, out = _run()
    assert rc == 0, f"selftest_canonical.py did not pass cleanly:\n{out}"
    assert "n_panels corrupted (companion field, not value), hash re-stamped to match" in out
    assert "MISSED" not in out
    assert "10/10 planted defects caught" in out


def test_selftest_includes_a_values_keys_hash_ratchet_only_mutation():
    """canonical.py gained VALUES_KEYS_HASH/
    CHECKED_KEYS_HASH ratchets, closing the swap-exploit live-demonstrated
    (removing one real check and adding an unrelated passthrough for a different key leaves
    N_CHECKED's COUNT unchanged). No prior mutation here isolates the new hash specifically -- this
    one renames an unregistered key to a synthetic one, keeping the payload COUNT identical (so
    N_VALUES stays satisfied and no other mutation's mechanism fires), so only VALUES_KEYS_HASH can
    catch it. Asserts the mutation exists, is labeled correctly, and is actually caught (not
    MISSED)."""
    rc, out = _run()
    assert rc == 0, f"selftest_canonical.py did not pass cleanly:\n{out}"
    assert "renamed to a synthetic key (same payload count), hash re-stamped to match " \
           "(VALUES_KEYS_HASH-ratchet-only test)" in out
    assert "MISSED" not in out


def test_selftest_includes_a_sentinel_scan_only_mutation():
    """Mapping all 7 pre-existing
    mutations to canonical.py::check()'s own code paths found only 3 genuinely independent
    detection mechanisms -- 5 of 7 mutations share one (a stale/mismatched `_meta.hash`). check()'s
    standalone SENTINEL-scan loop (which runs over every key/field regardless of CHECKS
    registration, added after an incident where a sentinel was left live in an UNREGISTERED key)
    was never isolated by any prior mutation. This one plants SENTINEL into an unregistered key's
    `value` and re-stamps the hash to match, so neither the hash check nor any CHECKS[] entry
    (which never looks at an unregistered key) can catch it -- only the SENTINEL scan can. Asserts
    the mutation exists, is labeled correctly, and is actually caught (not MISSED)."""
    rc, out = _run()
    assert rc == 0, f"selftest_canonical.py did not pass cleanly:\n{out}"
    assert "(unregistered) set to SENTINEL, hash re-stamped to match (SENTINEL-scan-only test)" in out
    assert "MISSED" not in out


def test_selftest_includes_an_n_values_ratchet_only_mutation():
    """canonical.py::check()'s N_VALUES/N_CHECKED ratchet (the
    `for name, want, got in (("N_VALUES", ...), ("N_CHECKED", ...))` loop) was never isolated by
    any prior mutation -- mutation 4 ("a value deleted outright") shrinks the payload the same way,
    but does not re-stamp the hash, so it is caught by the cheap stale-hash path first. This
    mutation deletes an UNREGISTERED key and re-stamps the hash to match the smaller payload, so
    neither the hash check, CHECKS[] re-derivation, the companion-field check, nor the SENTINEL
    scan can catch it -- only the N_VALUES half of the ratchet (a hardcoded module-level int in
    canonical.py's own source, which no JSON-only mutation can make agree with a shrunk payload)
    can. Asserts the mutation exists, is labeled correctly, and is actually caught (not MISSED)."""
    rc, out = _run()
    assert rc == 0, f"selftest_canonical.py did not pass cleanly:\n{out}"
    assert "(unregistered) deleted outright, hash re-stamped to match (N_VALUES-ratchet-only test)" in out
    assert "MISSED" not in out


def test_missing_known_genuine_key_aborts_loudly(monkeypatch, capsys):
    """A review of the fix above flagged that if KNOWN_GENUINE_CHECK_KEY ever silently
    vanished from the payload or canonical.CHECKS (a future rename, or a swap for a readback), the
    original guard would just silently drop the extra mutation with zero explanation -- the exact
    "verification laundering the defects it existed to catch" failure mode this whole file's own
    docstring says it exists to prevent. Fixed by aborting loudly (exit 2) instead. This test
    exercises that path in-process (not via subprocess, since we need to monkeypatch the constant)
    -- safe because the abort happens before any file mutation or sidecar creation."""
    import selftest_canonical as ST

    monkeypatch.setattr(ST, "KNOWN_GENUINE_CHECK_KEY", "not.a.real.key.anywhere")
    before = CANON.read_bytes()
    rc = ST.main()
    out = capsys.readouterr().out
    after = CANON.read_bytes()

    assert rc == 2
    assert "ABORT" in out
    assert "not.a.real.key.anywhere" in out
    assert before == after  # no mutation, no sidecar left behind
    assert not ST.SIDECAR.exists()


def test_sidecar_only_deleted_when_restore_verified_identical():
    """main()'s `finally` block used to unlink SIDECAR unconditionally, even when
    `restored_identical` came back False -- deleting the only external recovery artifact for
    paper/canonical.json at the exact moment it would be needed (e.g. a concurrent process touching
    canonical.json in that window). Simulating the actual restore-failure condition safely requires
    faking a read-back mismatch without ever leaving the real paper/canonical.json genuinely
    corrupted, which is exactly the invasive-to-test-safely shape this file's own module docstring
    already accepts hand-verification for (see its "Verified by hand... too invasive for routine
    CI" precedent). This test instead locks in the SOURCE structure of the fix -- the unlink call
    must be gated behind `restored_identical`, not run unconditionally -- so a future edit
    reverting the guard fails this test immediately rather than only being caught by luck the next
    time the race actually fires."""
    import inspect
    import selftest_canonical as ST

    src = inspect.getsource(ST.main)
    finally_block = src[src.index("finally:"):]
    unlink_pos = finally_block.index("SIDECAR.unlink()")
    # rfind, NOT rindex: rindex RAISES on a miss, so `!= -1` below would be unreachable.
    guard_pos = finally_block.rfind("if restored_identical", 0, unlink_pos)
    assert guard_pos != -1, (
        "SIDECAR.unlink() must be reached only through an `if restored_identical` branch -- an "
        "unconditional or wrongly-scoped unlink would delete the recovery sidecar even when the "
        "restore did not verify byte-identical"
    )
    assert "WARNING" in finally_block and "preserved" in finally_block, (
        "the restore-failed branch must explicitly warn that the sidecar was preserved, not just "
        "silently skip deleting it"
    )


def test_recover_interrupted_only_deletes_sidecar_when_restore_verified_identical():
    """The SAME unconditional-unlink-after-copy bug the test above locks in for main()'s `finally`
    block also existed in `_recover_interrupted()` -- the startup path that restores
    canonical.json from a leftover sidecar found at process start -- a different code path in the
    same file, confirmed LIVE-REACHABLE during testing. Same source-structure verification as the
    sibling test above (mutating the real sidecar-recovery race safely isn't feasible for routine
    CI, per this file's own established precedent)."""
    import inspect
    import selftest_canonical as ST

    src = inspect.getsource(ST._recover_interrupted)
    # rindex/rindex: take the LAST unlink call and the nearest `if ` preceding it, so the guard
    # matched is the one actually controlling that call rather than some earlier branch.
    unlink_pos = src.rindex("SIDECAR.unlink()")
    guard_pos = src.rindex("if ", 0, unlink_pos)
    assert "CANON.read_bytes() ==" in src[guard_pos:unlink_pos], (
        "SIDECAR.unlink() in _recover_interrupted() must be reached only after verifying the "
        "restored CANON bytes match the sidecar -- an unconditional unlink would delete the "
        "recovery artifact even when shutil.copy2 silently failed to land byte-identical"
    )
    assert "WARNING" in src and "PRESERVED" in src, (
        "the restore-not-verified branch must explicitly warn that the sidecar was preserved"
    )


def test_main_actually_aborts_when_recover_interrupted_reports_failure(monkeypatch, tmp_path):
    """The first version of the fix above had a
    real, confirmed gap -- _recover_interrupted() printed a WARNING on a not-verified restore but
    returned None, and main() never checked the return value, so it carried on regardless. Its own
    try/finally block then unconditionally did `shutil.copy2(CANON, SIDECAR)` (overwriting
    whatever _recover_interrupted() had just "preserved") and unlinked SIDECAR the moment
    `restored_identical` came back True -- which it necessarily would, since that comparison reads
    CANON against `original`, itself read from the same file _recover_interrupted() left in a
    not-verified-correct state. Net effect: the "has been PRESERVED" promise printed to the user
    was false by the time main() finished. Fixed by having main() check _recover_interrupted()'s
    return value and abort immediately.

    A plain `shutil.copy2(SIDECAR, CANON)` on a local filesystem always lands byte-identical in a
    single-process test, so the failure branch can't be triggered by just giving CANON/SIDECAR
    different starting content (the copy overwrites CANON with SIDECAR's bytes before the
    comparison runs) -- confirmed by hand while writing this test. `shutil.copy2` itself is
    monkeypatched to simulate a genuine partial/corrupted copy (the real-world case this guard
    exists for: a concurrent writer racing in between, or a truncated write) by writing DIFFERENT
    bytes than the sidecar, so the verification correctly fails."""
    import selftest_canonical as ST

    fake_canon = tmp_path / "canonical.json"
    fake_sidecar = tmp_path / "canonical.json.selftest_backup"
    fake_canon.write_bytes(b'{"_meta": {}, "a": {"value": 1}}')
    fake_sidecar.write_bytes(b'{"_meta": {}, "a": {"value": 999}}')

    def _corrupted_copy(src, dst):
        Path(dst).write_bytes(b"not what the sidecar actually contains")

    monkeypatch.setattr(ST, "ROOT", tmp_path)  # CANON.relative_to(ROOT) in the print statements needs this
    monkeypatch.setattr(ST, "CANON", fake_canon)
    monkeypatch.setattr(ST, "SIDECAR", fake_sidecar)
    monkeypatch.setattr(ST.shutil, "copy2", _corrupted_copy)

    rc = ST.main()

    assert rc == 2, "main() must abort (not proceed to mutate canonical.json) when the sidecar restore didn't verify"
    assert fake_sidecar.exists(), "the sidecar must still exist -- main() must not reach the code that would delete it"
    assert fake_canon.read_bytes() != fake_sidecar.read_bytes(), (
        "sanity check on the fixture itself: the simulated corrupted copy must genuinely leave "
        "CANON different from SIDECAR, or this test isn't exercising the failure path at all"
    )

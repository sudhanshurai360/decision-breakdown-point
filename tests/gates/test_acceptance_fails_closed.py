"""Regression test: acceptance.py had 3 vacuous-pass paths.

  1. A check that raised (`verdict="ERROR"`) was counted but never gated the exit code -- a check
     that cannot even RUN is not "a gap in the registry," it is exactly what breaks first when a
     producer's CSV schema, column name, or CHECKS signature changes underneath this gate. The
     same fail-open shape as canonical.py's `unchk` bug.
  2. No floor on the comparable-key count (`len(set(cg.CHECKS) & set(led))`): a shrinking
     intersection silently reduces `n` toward zero while `live`/`errored` both stay empty, printing
     "0 comparable keys" and PASS.
  3. No floor on the BLOCKED count specifically: a regression could be hidden by adding a new
     prefix/key to the BLOCKED dict rather than fixing it -- "a floor is satisfied by deleting a
     check," applied here to "satisfied by exempting a check."

Fixed by: gating the exit code on `errored` too, and adding an exact-equality ratchet
(N_COMPARABLE, N_BLOCKED) mirroring canonical.py's own N_VALUES/N_CHECKED discipline.

`acceptance.py`'s `main()` was refactored to break `_load_canonical_module()` out as its own
top-level function specifically so these tests can inject a fake CHECKS registry via monkeypatch,
without ever touching the real code/gates/canonical.py file on disk.

A follow-up review of the fix above found two further issues in the same rewrite,
both fixed here too: (a) the rewrite accidentally DROPPED the --verbose row-printing block instead
of keeping it alongside the new errored/ratchet printing, silently regressing --verbose to a
declared-but-unread dead flag; (b) classify() was called OUTSIDE the try/except wrapping
cg.CHECKS[key](), so a bug in classify() itself crashed the whole script with a raw traceback
instead of producing a clean ERROR row for just that one key.
"""
import io
import json
from contextlib import redirect_stdout
from types import SimpleNamespace

import pytest

import acceptance as A

pytestmark = pytest.mark.skipif(not A.LEDGER.exists(), reason="ledger_recorded.json not available")


def _real_ledger_keys(n=3):
    led = json.loads(A.LEDGER.read_text())
    return list(led.keys())[:n], led


def _run_main_with_fake_cg(monkeypatch, checks):
    monkeypatch.setattr(A, "_load_canonical_module", lambda: SimpleNamespace(CHECKS=checks))
    monkeypatch.setattr("sys.argv", ["acceptance.py"])
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = A.main()
    return rc, buf.getvalue()


def test_clean_run_still_passes():
    """The refactor into _load_canonical_module() must not change real-run behavior. Runs the
    real (non-monkeypatched) script as a subprocess, exercising the actual
    code/gates/canonical.py on disk."""
    import subprocess
    import sys
    r = subprocess.run([sys.executable, str(A.HERE / "acceptance.py")],
                        cwd=A.ROOT, capture_output=True, text=True)
    assert r.returncode == 0
    assert "PASS" in r.stdout
    assert "ratchet ok" in r.stdout


def test_errored_check_fails_closed(monkeypatch):
    """The exact scenario described above: one CHECKS entry raises. Before the fix this was
    silently absorbed into tot['error'], printed only under --verbose, exit 0. After the fix it
    must FAIL."""
    keys, led = _real_ledger_keys(3)

    def boom():
        raise RuntimeError("simulated breakage")

    checks = {k: boom for k in keys}
    rc, out = _run_main_with_fake_cg(monkeypatch, checks)

    assert rc == 1
    assert "FAIL" in out
    assert "ERRORED check(s): CANNOT run" in out
    # main() stores only type(e).__name__, not str(e) -- the exception message itself is not
    # surfaced, a real (if minor) debuggability gap, but not a fail-open one: the type name alone
    # is enough to distinguish "errored" from every other verdict, which is all this test needs.
    assert "RuntimeError" in out


def test_shrunk_comparable_intersection_fails_closed(monkeypatch):
    """The second gap: a shrinking set(cg.CHECKS) & set(led) intersection used to print
    a smaller 'N comparable keys' header and PASS if nothing else was DIFFER/ERROR. The ratchet
    must now catch this even when every one of the (fewer) keys MATCHes."""
    keys, led = _real_ledger_keys(3)
    checks = {k: (lambda k=k: led[k]["latest"]) for k in keys}  # trivially matches every key
    rc, out = _run_main_with_fake_cg(monkeypatch, checks)

    assert rc == 1
    assert "FAIL" in out
    assert "RATCHET INCONSISTENT — N_COMPARABLE" in out
    # and specifically NOT because anything differed or errored -- isolating the ratchet as the
    # thing that caught it
    assert "0 live difference(s), 0 errored check(s)" in out


def test_verbose_flag_still_prints_non_differ_rows(monkeypatch):
    """A follow-up review of the fix caught that the rewrite of main()'s tail accidentally
    DROPPED the --verbose row-printing block instead of keeping it alongside the new
    errored/ratchet printing -- silently regressing --verbose from 'prints every non-DIFFER row'
    to a declared-but-unread dead flag. Restored; this locks it in."""
    keys, led = _real_ledger_keys(2)
    checks = {k: (lambda k=k: led[k]["latest"]) for k in keys}  # both MATCH

    monkeypatch.setattr(A, "_load_canonical_module", lambda: SimpleNamespace(CHECKS=checks))
    monkeypatch.setattr("sys.argv", ["acceptance.py", "--verbose"])
    buf = io.StringIO()
    with redirect_stdout(buf):
        A.main()
    out = buf.getvalue()

    detail_lines = [l for l in out.splitlines() if l.strip().startswith("MATCH")]
    for k in keys:
        assert any(k in l for l in detail_lines), f"{k!r} not printed under --verbose: {out}"


def test_classify_exception_becomes_error_row_not_a_crash(monkeypatch):
    """A follow-up review of the fix also found classify() was called OUTSIDE the
    try/except wrapping cg.CHECKS[key](), so a bug in classify() (or STRING_NORMALIZE) crashed
    the whole script with a raw traceback instead of producing a clean ERROR row for just that
    key. Widened the try/except to cover classify() too."""
    keys, led = _real_ledger_keys(2)
    checks = {k: (lambda k=k: led[k]["latest"]) for k in keys}

    def boom_classify(key, got, want):
        raise ValueError("simulated classify() bug")

    monkeypatch.setattr(A, "classify", boom_classify)
    rc, out = _run_main_with_fake_cg(monkeypatch, checks)

    assert rc == 1  # fails closed (via ERROR + ratchet), not an uncaught traceback
    assert "ERRORED check(s): CANNOT run" in out
    assert "ValueError" in out


def test_ratchet_ok_on_real_repo_state():
    """A quieter guard than the two scenario tests above: confirms the CURRENT real repo state
    still satisfies both ratchets, so N_COMPARABLE/N_BLOCKED don't silently drift out of sync with
    a real, legitimate change to CHECKS or the BLOCKED dict without this test flagging it."""
    cg = A._load_canonical_module()
    led = json.loads(A.LEDGER.read_text())
    comparable = sorted(set(cg.CHECKS) & set(led))
    n = len(comparable)
    assert n == A.N_COMPARABLE, (
        f"comparable-key count drifted to {n} -- if this is a legitimate change (new CHECKS "
        f"entry, ledger update), bump N_COMPARABLE in the same edit; if not, something shrank "
        f"the intersection unexpectedly"
    )

    # N_BLOCKED, re-derived the same way main() does (its own classify()/blocked_reason()), so the
    # BLOCKED half of "both ratchets" is genuinely read here and cannot drift unnoticed.
    canon_path = getattr(cg, "OUT", None)
    canon = json.loads(canon_path.read_text()) if canon_path and canon_path.exists() else {}
    blocked = 0
    for key in comparable:
        try:
            got = cg.CHECKS[key]()
            verdict, _note = A.classify(key, got, led[key]["latest"])
        except Exception:
            continue          # an ERROR row, counted by main() as 'error', never as 'blocked'
        if verdict == "DIFFER" and A.blocked_reason(key, got, canon):
            blocked += 1
    assert blocked == A.N_BLOCKED, (
        f"BLOCKED-key count drifted to {blocked} -- a regression must not be hidden by adding a "
        f"new prefix/key to the BLOCKED dict; if this is a legitimate change, bump N_BLOCKED (and "
        f"BLOCKED_KEYS_HASH) in the same edit"
    )


def test_a_same_count_comparable_key_swap_is_caught_by_the_hash_even_though_the_count_ratchet_would_miss_it(
    monkeypatch,
):
    """Regression test: closes the same swap-exploit shape already found and fixed in
    canonical.py's own ratchets. N_COMPARABLE/N_BLOCKED are exact-equality on COUNT
    alone, so replacing the comparable-key set with a different, same-size set -- one key drops
    out, a different one enters -- would satisfy N_COMPARABLE while changing what's actually being
    verified. COMPARABLE_KEYS_HASH must catch it even when the count ratchet cannot."""
    keys, led = _real_ledger_keys(4)
    a, b, c, d = keys

    import hashlib
    hash_v1 = hashlib.sha256(",".join(sorted((a, b, c))).encode()).hexdigest()
    empty_blocked_hash = hashlib.sha256(b"").hexdigest()
    monkeypatch.setattr(A, "N_COMPARABLE", 3)
    monkeypatch.setattr(A, "N_BLOCKED", 0)
    monkeypatch.setattr(A, "COMPARABLE_KEYS_HASH", hash_v1)
    monkeypatch.setattr(A, "BLOCKED_KEYS_HASH", empty_blocked_hash)

    checks_v1 = {k: (lambda k=k: led[k]["latest"]) for k in (a, b, c)}
    rc1, out1 = _run_main_with_fake_cg(monkeypatch, checks_v1)
    assert rc1 == 0, f"fixture assumption: the first 3-key set must cleanly pass: {out1}"

    # Same COUNT (3), different member (d instead of c) -- the count-only ratchet cannot see this.
    checks_v2 = {k: (lambda k=k: led[k]["latest"]) for k in (a, b, d)}
    rc2, out2 = _run_main_with_fake_cg(monkeypatch, checks_v2)

    assert rc2 == 1
    assert "FAIL" in out2
    assert "RATCHET INCONSISTENT — COMPARABLE_KEYS_HASH" in out2
    # and specifically NOT because N_COMPARABLE's own count check caught it
    assert "RATCHET INCONSISTENT — N_COMPARABLE" not in out2

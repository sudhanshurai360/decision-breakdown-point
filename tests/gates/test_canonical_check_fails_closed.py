"""Regression test: `canonical.py --check`
used to fail OPEN: pointing RES/FLAG_RES/ROOT_RES at
nonexistent paths collapsed machine-checked coverage from 171 to ~1 (every other CHECKS entry
raising FileNotFoundError, silently counted as neither ok nor bad) while `check()` still printed
PASS and returned exit 0. This is the same failure mode that once let a deleted `results/`
directory pass green ("159 ok / 10 uncheckable, still PASS").

The fix in canonical.py has two parts, both covered here:
  1. the exact-equality ratchet (N_VALUES/N_CHECKED) was re-enabled after being disabled to None
     during a corpus rebuild;
  2. `unchk` (registered checks that raised, distinct from `unregistered` = never registered) now
     gates `good`, not just `bad`.

A follow-up found one more live gap in
the same function: the SENTINEL-detection loop printed a loud "*** SENTINEL LEFT LIVE ***" warning
but never touched `good`. A sentinel value landing on one of the UNREGISTERED keys (N_VALUES minus
N_CHECKED, 217 today -- no machine check, so `bad`/`unchk` never see it) -- with the stored
`_meta.hash` recomputed to match, e.g. by
re-running `merge()` over a corrupted fragment -- printed the warning and still returned PASS/exit
0. Fixed by tracking `sentinel_found` and adding `and not sentinel_found` to `good`. Covered by
`test_sentinel_on_unregistered_key_fails_closed` below.
"""
import json

import pytest

import canonical as C

pytestmark = pytest.mark.skipif(not C.OUT.exists(), reason="paper/canonical.json not available")


def test_ratchet_is_enabled_not_disabled():
    """Guards against a repeat of the regression above: the ratchet silently
    left at None after a corpus rebuild, which defeats the N_VALUES/N_CHECKED INCONSISTENT check
    entirely (want is None => skipped)."""
    assert C.N_VALUES is not None
    assert C.N_CHECKED is not None


def test_ratchet_matches_current_canonical_json():
    doc = json.loads(C.OUT.read_text())
    payload = C._payload(doc)
    registered = sum(1 for k in C.CHECKS if k in payload)
    assert len(payload) == C.N_VALUES
    assert registered == C.N_CHECKED


def test_clean_check_passes(capsys):
    rc = C.check()
    out = capsys.readouterr().out
    assert rc == 0
    assert "PASS" in out
    assert "uncheckable 0" in out


def test_vanished_inputs_fail_closed(monkeypatch, capsys, tmp_path):
    """The exact scenario that demonstrates the bug: point every
    input-resolving path constant at a location with nothing under it. Before the fix this
    printed PASS / returned 0. After the fix it must print FAIL / return 1, and every check that
    could not run must be visible in the printed output rather than silently dropped."""
    nonexistent = tmp_path / "nonexistent"
    monkeypatch.setattr(C, "RES", nonexistent)
    monkeypatch.setattr(C, "FLAG_RES", nonexistent)
    monkeypatch.setattr(C, "ROOT_RES", nonexistent)

    rc = C.check()
    out = capsys.readouterr().out

    assert rc == 1
    assert "FAIL" in out
    assert "PASS" not in out
    # the registered-check count collapses toward the handful of checks that don't touch a CSV
    # under RES/FLAG_RES/ROOT_RES, and the rest must show up as UNCHECKABLE, not vanish quietly
    assert "uncheckable" in out
    assert "UNCHECKABLE" in out
    # interpolated from the live constant, not hardcoded: a literal would silently stop guarding
    # the day N_CHECKED changes.
    assert f"ok {C.N_CHECKED}" not in out


def test_sentinel_on_unregistered_key_fails_closed(monkeypatch, capsys, tmp_path):
    """The gap found in the vanished-inputs fix above: a sentinel value
    on an UNREGISTERED key (one with no machine check at all) is invisible to `bad`/`unchk`, so a
    hash-consistent corruption there must be caught by the sentinel scan itself, not fall through
    to a printed warning plus PASS/exit 0.

    Mutates an isolated tmp_path COPY of paper/canonical.json, never the real file -- a plain
    try/finally around a write to
    the real C.OUT risks leaving the actual published-numbers file corrupted on a hard kill
    (SIGKILL/OOM/timeout) between the write and the finally block. Reusing the same
    monkeypatch(C, ...) pattern as test_vanished_inputs_fail_closed above removes that risk
    category entirely rather than merely mitigating it."""
    doc = json.loads(C.OUT.read_text())
    payload = C._payload(doc)
    unregistered = [k for k in payload if k not in C.CHECKS]
    assert unregistered, "test needs at least one unregistered key to plant the sentinel on"
    key = unregistered[0]

    doc[key]["value"] = C.SENTINEL
    doc["_meta"]["hash"] = C._digest(doc)  # hash-consistent corruption, same as merge() would produce
    tmp_out = tmp_path / "canonical.json"
    tmp_out.write_text(json.dumps(doc, indent=2))
    monkeypatch.setattr(C, "OUT", tmp_out)

    rc = C.check()
    out = capsys.readouterr().out

    assert "SENTINEL LEFT LIVE" in out
    assert rc == 1
    assert "FAIL" in out
    assert "PASS" not in out

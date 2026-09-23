"""Regression test: `blocked_reason()`
used to be a pure key-PREFIX match ("eb.", "vintage.", ...) -- ANY value under an exempted prefix,
right or wrong, was classified BLOCKED, because the exemption was granted to the key, not to any
specific value. Demonstrated live before the fix: monkeypatching `CHECKS['eb.episodes']` to return
a fabricated regressed value (35, vs the real, current, correct 29) and rerunning the unmodified
classification logic still reported BLOCKED, 0 live differences -- exactly the class of bug
acceptance.py itself exists to catch (a real incident: the pipeline ran green while
state_eb_decision.py silently emitted a retracted number).

The documented reason these keys are exempt is real and correct (the ledger's OWN recorded "latest"
value for eb.*/vintage.* is harvested from a known-wrong, retracted pre-rebuild generation) -- but
that's a reason to stop comparing against the stale LEDGER, not a reason to stop comparing at all.
Fixed: `blocked_reason(key, got, canon)` now also requires the fresh `got` to match
`paper/canonical.json`'s own current, independently-verified value for that key. Matching it keeps
the documented exemption (expected, understood divergence from the stale ledger); NOT matching it
means the fresh value diverges from BOTH the stale ledger AND today's verified source of truth --
which is real news -- so it now falls through to a genuine DIFFER instead of being silently exempted.
"""
import json

import pytest

import acceptance as A

CANON_PATH = A.ROOT / "paper" / "canonical.json"
_CANON = json.loads(CANON_PATH.read_text()) if CANON_PATH.exists() else {}


def _blocked_canonical_keys() -> list[str]:
    """Every canonical key that A.BLOCKED actually exempts, DERIVED from A.BLOCKED rather than
    hand-listed. BLOCKED mixes two entry shapes: dotted PREFIX entries ('eb.', 'vintage.'), which
    exempt every canonical key beneath them, and exact-key entries (the other 29). The previous
    version of this file hand-listed three keys while its docstring claimed to cover "every family
    in BLOCKED" -- 'election.' alone has 12 exact entries, not the "two" the docstring named."""
    out = []
    for entry in sorted(A.BLOCKED):
        if entry.endswith("."):
            out.extend(sorted(k for k in _CANON if k.startswith(entry)))
        elif entry in _CANON:
            out.append(entry)
    return sorted(set(out))


BLOCKED_CANONICAL_KEYS = _blocked_canonical_keys()


@pytest.fixture
def canon():
    if not CANON_PATH.exists():
        pytest.skip("paper/canonical.json not available")
    return _CANON


def test_blocked_reason_still_exempts_the_real_current_correct_value(canon):
    """Sanity check: today's actual, correct eb.episodes value (whatever canonical.json currently
    says) must still be exempted -- this is the documented, deliberate, understood divergence from
    the known-wrong ledger, not a regression to chase."""
    got = canon["eb.episodes"]["value"]
    why = A.blocked_reason("eb.episodes", got, canon)
    assert why is not None, "the genuinely current, correct value must still be BLOCKED (exempted)"


def test_blocked_reason_no_longer_exempts_a_fabricated_regressed_value(canon):
    """THE CONFIRMED REGRESSION, locked in: a fabricated wrong value for a BLOCKED-prefixed key
    (simulating a future silent regression in state_eb_decision.py, the exact incident class this
    whole gate exists to catch) must NOT be silently exempted just because its key starts with
    'eb.'. It must diverge from canonical.json's own current value and fall through to a real
    DIFFER."""
    real = canon["eb.episodes"]["value"]
    fabricated = real + 6  # a plausible-looking but wrong regressed count
    why = A.blocked_reason("eb.episodes", fabricated, canon)
    assert why is None, (
        f"fabricated regressed value {fabricated} (real value is {real}) was still exempted as "
        f"BLOCKED -- the value-aware check did not fire"
    )


def test_the_derived_blocked_key_list_is_not_vacuous():
    """Non-vacuity guard for the parametrize below: if BLOCKED's entry SHAPE ever changes (e.g. all
    prefixes replaced by regexes) the derivation above would silently yield an empty list and the
    per-key test would generate zero cases while still reporting green."""
    assert CANON_PATH.exists(), "paper/canonical.json must be present for this file to mean anything"
    assert len(BLOCKED_CANONICAL_KEYS) >= len(A.BLOCKED), (
        f"BLOCKED has {len(A.BLOCKED)} entries but only {len(BLOCKED_CANONICAL_KEYS)} canonical "
        f"keys were derived from it -- the derivation has stopped matching BLOCKED's entry shape"
    )
    # every BLOCKED entry must be represented by at least one derived key
    unrepresented = [e for e in A.BLOCKED
                     if not any(k == e or (e.endswith(".") and k.startswith(e))
                                for k in BLOCKED_CANONICAL_KEYS)]
    assert unrepresented == [], (
        f"BLOCKED entries with no canonical key behind them: {unrepresented} -- either the key was "
        f"renamed/removed from canonical.json and its exemption is now dead, or the derivation is wrong"
    )


@pytest.mark.parametrize("blocked_key", BLOCKED_CANONICAL_KEYS)
def test_blocked_reason_value_awareness_applies_to_every_exempted_key(canon, blocked_key):
    """The fix must not be scoped to just eb.episodes: EVERY key BLOCKED actually exempts -- the
    full list derived from A.BLOCKED, prefix entries expanded over canonical.json -- must get the
    same value-aware treatment. The correct current value stays exempt; a fabricated one does not."""
    real = canon[blocked_key]["value"]
    if not isinstance(real, (int, float)) or isinstance(real, bool):
        pytest.skip(f"{blocked_key} is not a plain numeric value, fabrication scenario doesn't apply")
    if isinstance(real, float) and real != real:      # NaN: _num()'s NaN==NaN branch, no fabrication
        pytest.skip(f"{blocked_key} is NaN, fabrication scenario doesn't apply")
    fabricated = real + max(abs(real) * 0.5, 1.0)
    why_real = A.blocked_reason(blocked_key, real, canon)
    why_fabricated = A.blocked_reason(blocked_key, fabricated, canon)
    assert why_real is not None, f"{blocked_key}: current correct value must still be exempted"
    assert why_fabricated is None, f"{blocked_key}: fabricated value {fabricated} was still exempted"


def test_blocked_reason_falls_back_to_prefix_only_when_canon_not_supplied():
    """Backward-compatible call shape: blocked_reason(key) with no got/canon must still return the
    documented reason text (used by anything that only needs the exemption's rationale, e.g. a
    future doc-generation pass, not the live classification path)."""
    assert A.blocked_reason("eb.episodes") is not None
    assert A.blocked_reason("not_a_blocked_key.anything") is None


def test_clean_real_run_still_all_blocked_pass():
    """The value-aware fix must not turn today's real, correct pipeline into a false DIFFER --
    exercises the actual code/gates/canonical.py + real canonical.json on disk, as a subprocess."""
    import subprocess
    import sys

    r = subprocess.run([sys.executable, str(A.HERE / "acceptance.py")],
                        cwd=A.ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "PASS" in r.stdout
    assert "0 live difference" in r.stdout

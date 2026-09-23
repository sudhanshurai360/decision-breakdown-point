"""Regression test. HISTORY (a past state, not today's): canonical.py's CHECKS
dict once had 176 entries while paper/canonical.json's payload matched only 171 of them -- 5 entries
were dead, silently skipped by check()'s own `if key not in payload: continue`, checking values
the producers had stopped emitting (eb.flips_far_from_trigger, eb.max_episodes_one_state,
mcspower.specificity_0pct, shape.gfc_median_x, shape.hardest_elsewhere -- confirmed via grep: none
of the 5 is referenced by any producer in code/pipeline/ or code/instrument/ anymore, and none has
an obvious renamed replacement among the live eb.*/mcspower.*/shape.* canonical keys).

Removed the 5 dead entries (and _ebs(), the helper only they used). This had two real, legitimate
downstream effects, not bugs: 2 of the 5 removed keys also carried an entry in
paper/ledger_recorded.json, so acceptance.py's N_COMPARABLE ratchet fell by 2, and both of those
happened to be "eb."-prefixed (wholesale BLOCKED), so N_BLOCKED fell by 2 as well. Both ratchets
were re-pinned in the same commit; their current values live in acceptance.py and are not restated
here, because they move for unrelated reasons too.

This test locks in that CHECKS stays exactly the size of N_CHECKED going forward -- a dead entry
reappearing (or a fresh one being silently unregistered) would fail it immediately, rather than
sitting invisibly for however long the 176-vs-171 gap sat unnoticed.
"""
import json

import pytest

import canonical as C

# NOT a module-level pytestmark: only the first test below reads canonical.json off disk.
# test_checks_size_matches_n_checked_exactly() compares two in-memory module constants and must
# still run on a checkout with no canonical.json, so the skip is applied per-test instead.
_NEEDS_CANONICAL = pytest.mark.skipif(not C.OUT.exists(),
                                      reason="paper/canonical.json not available")


@_NEEDS_CANONICAL
def test_checks_dict_has_no_dead_entries():
    doc = json.loads(C.OUT.read_text())
    payload = C._payload(doc)
    dead = sorted(k for k in C.CHECKS if k not in payload)
    assert not dead, (
        f"CHECKS has entries for keys canonical.json no longer has: {dead} -- either the producer "
        f"stopped emitting them (remove the dead CHECKS entry) or this is a genuine rename/regression "
        f"(fix the key name instead)"
    )


def test_checks_size_matches_n_checked_exactly():
    """len(CHECKS) == N_CHECKED is a slightly stronger statement than "no dead entries": it also
    catches the reverse direction (an entry silently missing from CHECKS for a key that IS in the
    payload -- N_CHECKED would then overcount registered coverage that doesn't actually exist)."""
    assert len(C.CHECKS) == C.N_CHECKED

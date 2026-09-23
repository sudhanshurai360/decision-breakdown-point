"""Regression test for a LATENT crash, the same shape as
the None-crash (see test_claims_none_valued_canonical_entry.py).

`claims.py`'s `renderings()` and `percent_renderings()` fell through every `isinstance()` branch
for a dict-valued canonical entry (top-level `value`, or a dict nested inside a list) to a final
numeric-format call, which raises `TypeError` on a dict -- and since `orphans()`/`misplaced()`
iterate over every canonical value in one pass, this would take down the ENTIRE claims.py run, not
just the one offending key.

Confirmed latent, not live: no canonical entry in `paper/canonical.json` carries a dict as its
`value` today, and no `emit()` call site passes one in that position. Dict-valued COMPANION fields
DO exist -- state_eb_decision.py emits `dates=`/`spreads_pp=` on
eb.genuine_divergent_states_detail -- which is precisely why this crash class is reachable and
worth closing. Found by static/adversarial review,
not by a real crash the way the None case was. Fixed pre-emptively anyway, matching this
project's standing practice of closing a demonstrated defect *class* wherever else it's reachable,
not just the one instance that happened to fire first.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "code" / "gates"))

import claims as C  # noqa: E402


def test_renderings_of_a_dict_recurses_into_values_not_a_crash():
    assert C.renderings({"lo": 0.1, "hi": 0.2}) == C.renderings(0.1) | C.renderings(0.2)


def test_renderings_of_a_dict_nested_in_a_list_does_not_crash():
    """A dict inside a 2-element list hits the list branch's "range" formatting
    (`f"{v[0]}-{v[1]}"`) before per-element recursion, which stringifies the dict via its own
    repr rather than crashing -- a harmless junk string in the output set, not a bug this test
    needs to eliminate, just confirmed non-crashing. The per-element recursion below it still
    contributes the dict's own real renderings alongside that junk string."""
    out = C.renderings([{"x": 5}, 7])          # must not raise
    assert C.renderings(5) <= out and C.renderings(7) <= out


def test_percent_renderings_of_a_dict_recurses_into_values_not_a_crash():
    assert C.percent_renderings({"a": 0.5}) == C.percent_renderings(0.5)


def test_orphans_handles_a_dict_valued_canonical_entry_end_to_end():
    """The full path a live occurrence would go through: a canonical entry whose `value` is a
    dict must not crash orphans() -- it should recurse into the dict's values, the same as a list."""
    canon = C._canon()
    sections = C.index_draft()
    fake_key = "__test_dict_valued_entry__"
    assert fake_key not in canon, "test fixture key collided with a real canonical key"
    canon = dict(canon)
    canon[fake_key] = {"value": {"lo": 1.23, "hi": 4.56}, "cites": []}
    # Must not raise.
    out, _n_tok, _n_benign = C.orphans(canon, sections)
    assert isinstance(out, dict)

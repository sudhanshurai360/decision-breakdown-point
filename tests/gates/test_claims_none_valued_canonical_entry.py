"""Regression test for a live crash.

`claims.py`'s own `renderings()` and `percent_renderings()` functions -- used by `orphans()` to
build the set of strings a canonical value could plausibly appear as in the prose -- fell through
every `isinstance()` branch for a `None`-valued canonical entry to a final numeric-format call
(`f"{v:g}"` / `abs(v)`), both of which raise `TypeError` on `None`.

This was a real, live crash, not a hypothetical: `llm.condorcet_winner` (added mirroring
`flagship.condorcet_winner`'s established "None when no Condorcet winner exists" convention) came
out `None` on the real Open-LLM top-30 pool (a genuine majority cycle, confirmed via
`llm.condorcet_status == "cycle"`) the first time `claims.py` ran against it, crashing the whole
gate. `flagship.condorcet_winner` has carried the same None-when-no-winner convention from early on
without ever actually triggering this path on real data -- the bug was latent, not newly
introduced, and would have crashed on the very first genuine null regardless of which key hit it.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "code" / "gates"))

import claims as C  # noqa: E402


def test_renderings_of_none_returns_empty_set_not_a_crash():
    assert C.renderings(None) == set()


def test_percent_renderings_of_none_returns_empty_set_not_a_crash():
    assert C.percent_renderings(None) == set()


def test_orphans_handles_a_none_valued_canonical_entry_end_to_end():
    """The full path the live crash actually went through: a canonical entry whose `value` is
    `None` must not crash orphans() -- it should simply contribute nothing to the known-renderings
    set (a None value has no defensible literal text to search prose for), the same as a bool."""
    canon = C._canon()
    sections = C.index_draft()
    fake_key = "__test_none_valued_entry__"
    assert fake_key not in canon, "test fixture key collided with a real canonical key"
    canon = dict(canon)
    canon[fake_key] = {"value": None, "cites": []}
    # Must not raise.
    out, _n_tok, _n_benign = C.orphans(canon, sections)
    assert isinstance(out, dict)

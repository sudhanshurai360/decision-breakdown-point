"""Regression test: claims.py's orphans()
built ONE flat, context-free `known` set from every canonical value's renderings(), including the
percent-transformed form (e.g. a value 0.9 renders as "90" too, "0.9 as a percentage"). A bare
integer in the prose that happened to share digits with some UNRELATED value's percentage
rendering was silently marked "covered" -- demonstrated: the retracted "90 episodes" figure was
invisible because eb.near_trigger_pct=0.9 renders as "90". 5 of 8 live stale numbers found by hand
were invisible for this reason.

Fixed by isolating the percent-transform into percent_renderings(), gating it in orphans() on
whether the prose token ITSELF appeared with a literal % sign (via the new tokenise_percent()) --
a percent-derived rendering can only cover a percent-context occurrence, not a bare integer.

NOTE: this closes only the percent-collision mechanism, not the broader (structural, disclosed,
not attempted here) plain-integer collision problem between two unrelated small canonical values.

SCOPE OF EACH TEST. The first three tests are deliberately UNIT-level: they pin down
percent_renderings() and tokenise_percent() as functions and make no claim about orphans(). The
last three drive the REAL C.orphans() entry point (via the _patch_body monkeypatch rig below), so
that undoing the actual gate in orphans() -- `if not covered and (tok in pct_toks or bare in
pct_toks):` -- makes them fail. They previously re-implemented that covered-check inline and so
could not detect its removal.
"""
import pytest

import claims as C

pytestmark = pytest.mark.skipif(not C._draft_exists(), reason="paper/draft_ssrn.md not available")


def _patch_body(monkeypatch, tmp_path, body: str):
    """Point claims.py's manuscript-loading machinery at a synthetic draft containing exactly
    `body`, with no note.md, so orphans()'s internal load_body() call sees a fully controlled prose
    surface instead of the real paper/draft_ssrn.md. orphans() takes the canon as an argument but
    reads the PROSE via module-level DRAFT/NOTE paths, so this rig (shared with
    tests/gates/test_claims_orphans_companion_array_allowlist.py) is the only way to drive the real
    entry point on synthetic prose."""
    draft = tmp_path / "draft.md"
    draft.write_text(f"## 1. Section\n\n{body}\n")
    monkeypatch.setattr(C, "DRAFT", draft)
    monkeypatch.setattr(C, "NOTE", tmp_path / "does_not_exist_note.md")
    monkeypatch.setattr(C, "_DRAFT_TEXT_CACHE", None)
    monkeypatch.setattr(C, "_NOTE_TEXT_CACHE", None)


def test_percent_renderings_isolates_the_v_times_100_branch():
    """percent_renderings(0.9) must reproduce the exact collision the finding demonstrated."""
    assert "90" in C.percent_renderings(0.9)
    # and the plain (non-percent) renderings() of the same value must NOT include "90" once the
    # percent-only subset is excluded -- this is what orphans() now does per-value before unioning
    assert "90" not in (C.renderings(0.9) - C.percent_renderings(0.9))


def test_percent_renderings_empty_for_non_percent_eligible_types():
    assert C.percent_renderings(90) == set()          # int -- never percent-transformed
    assert C.percent_renderings("90") == set()         # str -- percent_renderings() never transforms non-numeric types; use renderings() for the verbatim form
    assert C.percent_renderings(True) == set()         # bool
    assert C.percent_renderings(5.0) == set()           # magnitude > 1 -- outside the (1e-4, 1] band


def test_tokenise_percent_only_captures_percent_suffixed_occurrences():
    body = "The rate was 90 last year and rose to 90% this year, alongside a plain 42."
    pct = C.tokenise_percent(body)
    assert "90" in pct
    assert "42" not in pct


def test_bare_integer_no_longer_absorbed_by_an_unrelated_percent_rendering(tmp_path, monkeypatch):
    """The exact synthetic reproduction of the demonstrated bug, driven through the REAL orphans()
    entry point: a canonical corpus with ONE percentage-typed value (0.9) and a prose body
    containing the bare integer "90" with no % sign anywhere near it. Before the fix this was
    silently covered; after the fix orphans() must report it.

    (This test used to re-implement orphans()'s covered-check inline, so replacing the live gate
    `if not covered and (tok in pct_toks or bare in pct_toks):` with a bare `if not covered:` --
    i.e. undoing the fix this file exists to guard -- left it green.)"""
    canon = {
        "some.pct_key": {"value": 0.9, "producer": "test", "cites": [], "cross_check": None},
    }
    body = "This carries a real cost: 90 distinct episodes were affected."
    # preconditions for the reproduction: the token is present and is NOT in percent context
    assert "90" in set(C.tokenise(body))
    assert "90" not in C.tokenise_percent(body)

    _patch_body(monkeypatch, tmp_path, body)
    orph, _n_tok, _n_benign = C.orphans(canon)
    assert "90" in orph, "the bare '90' must NOT be absorbed by the unrelated 0.9 value's % rendering"


def test_real_corpus_still_isolates_the_percent_collision_on_a_controlled_prose_body(tmp_path, monkeypatch):
    """This test used to point
    at paper/draft_ssrn.md's own retracted '90 episodes' figure as the real-world confirmation of the
    percent-collision fix -- exactly the collision mechanism this file protects (eb.near_trigger_pct=0.9 renders
    as "90", which used to silently cover the unrelated bare "90" in the retracted episode count).
    That figure has since been corrected in the manuscript (paper/draft_ssrn.md now correctly says 29
    episodes, not 90) as part of the same review's fixes -- a GOOD outcome, not a regression, and
    exactly what this test's own prior docstring anticipated: "if this now fails... paper/draft_ssrn.md's
    prose no longer contains the retracted figure, in which case this test should be updated, not
    just skipped." Updated rather than skipped: still exercises the REAL canonical corpus
    (C._canon(), which has a real eb.near_trigger_pct=0.9 among 200+ other real values) so this
    isn't purely synthetic like test_bare_integer_no_longer_absorbed_by_an_unrelated_percent_rendering
    above, but pairs it with a CONTROLLED prose body instead of depending on draft_ssrn.md's own
    ever-changing text for a specific stale-number instance to still be present.

    Like its synthetic sibling above, this now runs through the REAL C.orphans() rather than an
    inlined copy of its covered-check."""
    canon = C._canon()
    assert canon.get("eb.near_trigger_pct", {}).get("value") == 0.9, (
        "this test depends on eb.near_trigger_pct still being 0.9 in the real corpus -- if it "
        "changed, pick a different real percent-typed value near 0.9/0.09/etc. that still "
        "renders as '90' and update this test accordingly"
    )
    body = "This carries a real cost: 90 distinct episodes were affected, unrelated to any percentage."
    assert "90" in set(C.tokenise(body))
    assert "90" not in C.tokenise_percent(body)

    _patch_body(monkeypatch, tmp_path, body)
    orph, _n_tok, _n_benign = C.orphans(canon)
    assert "90" in orph, (
        "the bare '90' must NOT be absorbed by the real corpus's eb.near_trigger_pct=0.9 percent "
        "rendering -- if this fails, the percent-collision fix has regressed against the real canonical corpus"
    )


def test_percent_context_token_still_correctly_covered(tmp_path, monkeypatch):
    """The other half of the fix, also through the real orphans(): a token that DOES appear with a
    % sign must still be covered by a matching percent rendering -- the gate must not become
    over-strict and start reporting every legitimate percentage as an orphan."""
    canon = {
        "some.pct_key": {"value": 0.9, "producer": "test", "cites": [], "cross_check": None},
    }
    body = "The rate was 90% in that period."
    assert "90" in C.tokenise_percent(body)
    assert "90" in C.percent_renderings(canon["some.pct_key"]["value"])

    _patch_body(monkeypatch, tmp_path, body)
    orph, _n_tok, _n_benign = C.orphans(canon)
    assert "90" not in orph, (
        "a percent-context occurrence must still be covered by the matching percent rendering"
    )

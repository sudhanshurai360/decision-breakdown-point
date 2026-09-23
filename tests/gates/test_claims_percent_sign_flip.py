"""Regression test: claims.py's
`tokenise()` percent branch and `tokenise_percent()` never captured a leading sign, while
`percent_renderings()` on the canonical side DOES produce signed strings for a negative value --
an asymmetry specific to the percent code path (the plain numeric path already preserved sign via
`tok.lstrip("+")`/the `[+-]?` in BARE's first alternative).

Demonstrated directly: a correctly-signed "5% increase" and a sign-flipped "-5% decrease
(CORRUPTION)" both used to tokenise to the identical `{'5'}` -- a sign-flip corruption of any
percentage-rendered claim was invisible to ORPHAN detection. Fixed by adding `[+-]?` to the percent
alternative of BARE and to tokenise_percent()'s own regex, mirroring the plain-number branch.

This test reproduces that demonstration: it would have FAILED before the fix (the
sign-flipped token tokenised to the same bare digits as the correct one, and orphans()'s own
covered-set logic would have silently accepted it) and PASSES after.
"""
import claims as C


def test_tokenise_percent_captures_a_leading_sign():
    """The core mechanical fix: the percent branch must produce distinguishable tokens for a
    positive and a sign-flipped occurrence, exactly like the plain-number branch already does."""
    assert C.tokenise_percent("5% increase") == {"5"}
    assert C.tokenise_percent("-5% decrease") == {"-5"}
    # and tokenise() itself (which shares the same BARE regex) must agree
    assert "5" in C.tokenise("5% increase")
    assert "-5" in C.tokenise("-5% decrease")
    assert "5" not in C.tokenise("-5% decrease")  # the sign is part of the token, not dropped


def test_sign_flip_corruption_is_no_longer_silently_covered():
    """The synthetic reproduction: a canonical value that renders as a
    positive percentage ("5%"), and two prose bodies -- one correctly signed, one sign-flipped
    (the corruption). Before the fix both tokenised to the bare '5' and both reported covered=True.
    After the fix, only the correctly-signed one is covered; the sign-flipped one must be reported
    as an uncovered (orphan) token."""
    canon = {
        "some.pct_key": {"value": 0.05, "producer": "test", "cites": [], "cross_check": None},
    }
    known, known_percent = set(), set()
    for v in canon.values():
        all_r = C.renderings(v["value"])
        pct_r = C.percent_renderings(v["value"])
        known.update(all_r - pct_r)
        known_percent.update(pct_r)

    def _covered(body: str) -> bool:
        toks = set(C.tokenise(body))
        pct_toks = C.tokenise_percent(body)
        assert len(toks) == 1, f"test body must produce exactly one numeric token, got {toks}"
        tok = next(iter(toks))
        bare = tok.lstrip("+")
        covered = tok in known or bare in known
        if not covered and (tok in pct_toks or bare in pct_toks):
            covered = tok in known_percent or bare in known_percent
        return covered

    correct = "Claims rose by 5% increase this quarter."
    corrupted = "Claims fell by -5% decrease this quarter, a sign flip of the figure above."

    assert _covered(correct), "the correctly-signed rendering of the canonical value must be covered"
    assert not _covered(corrupted), (
        "REGRESSION: a sign-flipped percent corruption tokenised identically to the correct value "
        "and was silently marked covered -- this is exactly the finding "
        "this test guards against"
    )


def test_percent_renderings_negative_value_is_signed_and_disjoint_from_positive():
    """percent_renderings() already produced signed strings for a negative value (this was never
    the bug) -- pin that down explicitly so a future change to that function can't silently drift
    back into symmetry with the (now-fixed) tokeniser side by accident."""
    neg = C.percent_renderings(-0.05)
    pos = C.percent_renderings(0.05)
    assert "-5" in neg
    assert "5" not in neg          # the unsigned form must NOT be a member of the negative set
    assert "5" in pos
    assert "-5" not in pos
    assert neg.isdisjoint(pos)


def _patch_body(monkeypatch, tmp_path, body: str):
    """Point claims.py's manuscript-loading machinery at a synthetic draft containing exactly
    `body`, with no note.md, so orphans()'s internal load_body() call sees a fully controlled prose
    surface instead of the real paper/draft_ssrn.md. orphans() reads the manuscript via module-level
    DRAFT/NOTE paths rather than as an argument, so this monkeypatch rig (shared with
    tests/gates/test_claims_orphans_companion_array_allowlist.py) is the only way to drive it
    end-to-end on synthetic prose."""
    draft = tmp_path / "draft.md"
    draft.write_text(f"## 1. Section\n\n{body}\n")
    monkeypatch.setattr(C, "DRAFT", draft)
    monkeypatch.setattr(C, "NOTE", tmp_path / "does_not_exist_note.md")
    monkeypatch.setattr(C, "_DRAFT_TEXT_CACHE", None)
    monkeypatch.setattr(C, "_NOTE_TEXT_CACHE", None)


# NOTE on the value: the helper-level tests above use 5% because that is the literal figure the
# original demonstration used, but a bare "5" matches claims.py's `^\d$` single-digit BENIGN rule,
# so an end-to-end `orphans()` assertion on it would pass for the wrong reason (filtered as benign
# rather than covered). 42 is two-digit, matches no BENIGN rule, and exercises exactly the same
# percent code path.
CANON_PCT = {"some.pct_key": {"value": 0.42, "producer": "test", "cites": [], "cross_check": None}}


def test_orphans_end_to_end_accepts_the_correctly_signed_percent(tmp_path, monkeypatch):
    """Half one of the end-to-end reproduction, through the REAL orphans() entry point rather than
    a hand-rolled covered() helper: a correctly-signed "42%" that renders the canonical 0.42 must
    not be reported as an orphan (the fix must not have become over-strict)."""
    _patch_body(monkeypatch, tmp_path, "Claims rose by 42% increase this quarter.")
    orph, _n_tok, _n_benign = C.orphans(CANON_PCT)
    assert orph == {}, "correctly-signed claim must not be reported as an orphan"


def test_orphans_end_to_end_detects_the_sign_flip(tmp_path, monkeypatch):
    """Half two, and the finding itself: a sign-flipped "-42%" must be reported as an orphan by the
    real orphans() call -- proof the fix is live in the actual code path the gate runs, not just in
    the helper functions. (This test previously re-implemented orphans()'s covered-check inline, so
    it went on passing when the live percent covered-check was made sign-blind; it now calls
    C.orphans.)"""
    _patch_body(monkeypatch, tmp_path, "Claims fell by -42% decrease this quarter.")
    orph, _n_tok, _n_benign = C.orphans(CANON_PCT)
    assert "-42" in orph, (
        "sign-flipped claim must be reported as an orphan -- this is the ORPHAN direction actually "
        "catching the corruption that was invisible before the fix"
    )

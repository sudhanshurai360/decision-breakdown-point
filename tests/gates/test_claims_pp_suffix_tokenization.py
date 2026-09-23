"""Regression test: claims.py's
`BARE`/`tokenise()` used to be unable to extract a number immediately followed by "pp" (percentage
points, this project's own convention -- e.g. "0.45pp") as a token at all, because the plain-number
alternative's negative lookahead `(?![\\w%])` treats the "p" of "pp" as an ordinary word character.
This silently failed `misplaced()`'s citation check for `vintage.revision_sd` (S7's "SD 0.45pp")
even though the number genuinely was in the cited section -- MISPLACED counted it as missing for
the wrong reason. Fixed by adding a third `BARE` alternative mirroring the existing "%" one.

Two follow-on gaps that fix left behind, both pinned here:
1. The fix itself had zero regression coverage -- the ONLY `claims.py` change in that batch which
   actually moved a live gate number (it closed one MISPLACED entry, so `N_MISPLACED` was
   decremented by 1 in the same commit; the current value lives in `claims.py` and is not restated
   here), unlike its two sibling fixes in the same batch (percent sign-flip, THOUS-range
   fragments), each of which got tests.
2. The new "pp" alternative reintroduced the exact thousands-separator fragment defect the SAME
   batch had just fixed for numeric ranges: THOUS's own trailing lookahead treats "p" as a word
   character, so "1,200pp" fell through THOUS entirely and got shredded into spurious ['1', '200']
   fragments by the bare alternatives. Fixed by a separate `THOUS_PP` pattern, matched and masked
   alongside THOUS in tokenise().
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "code" / "gates"))

import claims as C  # noqa: E402


def test_bare_pp_value_tokenises_to_the_bare_number():
    assert C.tokenise("the SD is 0.45pp today") == ["0.45"]
    assert C.tokenise("mean forecast bias is 0.02pp for ar1") == ["0.02"]


def test_bare_pp_value_preserves_sign():
    """tokenise() itself preserves a leading sign verbatim (both "+" and "-") -- the "+"-stripping
    normalization happens downstream, at the orphans()/misplaced() call sites, not here."""
    assert C.tokenise("a swing of -3.1pp was observed") == ["-3.1"]
    assert C.tokenise("a swing of +3.1pp was observed") == ["+3.1"]


def test_pp_suffix_does_not_spuriously_match_words_ending_in_pp():
    """An unrelated word ending in "pp" (not preceded by a digit) must never be mistaken for a
    percentage-point value -- the lookbehind requires a genuine digit sequence immediately before
    "pp", not just any occurrence of the two letters."""
    assert C.tokenise("submit the app or the approp process") == []
    assert C.tokenise("2ppm and 3ppt and 12ppb are different units entirely") == []
    assert C.tokenise("x1pp is not a real token, nor is 5pp5") == []


def test_percent_and_plain_number_tokenisation_unaffected_by_the_pp_alternative():
    """The pp alternative must not change how "%"-suffixed or bare numbers already tokenise."""
    assert C.tokenise("98% coverage") == ["98"]
    assert C.tokenise("exactly 42 units") == ["42"]
    assert C.tokenise("-5% decrease vs 5% increase") == ["-5", "5"]


def test_thousands_separated_pp_value_tokenises_as_one_token_not_fragments():
    """A thousands-separated pp value used
    to fall through THOUS entirely (its trailing lookahead also treats "p" as a word character) and
    get shredded by the bare alternatives into spurious fragment tokens. Dormant on the live corpus
    (today's largest pp value is well under 1,000) but locked in here so a future thousands-scale
    pp claim doesn't silently produce a false ORPHAN and a false MISPLACED.

    Scoped to a SINGLE comma-group (NNN,NNNpp), matching the reported case. A
    multi-comma-group number immediately followed by ANY word character (not just "pp" -- e.g.
    "1,234,567users") backtracks THOUS itself to a shorter, wrong match ("1,234") -- a separate,
    pre-existing latent defect in THOUS's own greedy-then-backtrack behavior, confirmed present
    before this pp fix and unrelated to it (reproduces with a plain trailing "."/word char,
    no "pp" involved). Out of scope for this fix; flagged for future work rather than folded in
    here, since correcting THOUS's core backtracking would be a broader, riskier change to this
    file than the narrow pp-suffix gap this test exists to pin."""
    assert C.tokenise("a thousands-scale value of 1,200pp is unusual") == ["1,200"]


def test_thousands_separated_plain_and_percent_values_unaffected():
    """Sanity check on the fixture itself: the ordinary (non-pp) thousands paths must still work
    exactly as before this fix."""
    assert C.tokenise("148,169 dollars") == ["148,169"]
    assert C.tokenise("a rate of 1,200%") == ["1,200"]


def test_vintage_revision_sd_is_no_longer_misplaced():
    """End-to-end regression for the actual motivating case: vintage.revision_sd's citation (S7's
    "SD 0.45pp") must now resolve via the real gate, not just the isolated tokenise() unit above."""
    canon = C._canon()
    sections = C.index_draft()
    bad, unknown_loc, checked, uncited = C.misplaced(canon, sections)
    assert not any(b.startswith("vintage.revision_sd") for b in bad), (
        f"vintage.revision_sd is still reported MISPLACED -- the pp-suffix fix did not resolve it "
        f"end-to-end: {[b for b in bad if b.startswith('vintage.revision_sd')]}"
    )

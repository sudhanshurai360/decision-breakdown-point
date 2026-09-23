"""Regression test for the blanket single-digit BENIGN rule (`^\\d$`) in claims.py, whose own
justification in the BENIGN table states that "corpus density alone already makes every digit 0-9
indistinguishable from an unrelated canonical value" and that the rule therefore "does not add or
remove any actual detection power, it only makes the pass-through an explicit, disclosed no-op
instead of an implicit one".

Measured directly against the live corpus, that holds: every one of 0-9 is ALREADY in the
context-free `known` set from unrelated corpus density alone, independent of the BENIGN rule --
narrowing the rule to a scoped allowlist (the fix shape used by every other BENIGN entry) would not
improve detection at all, since the same digits pass silently via incidental collision either way.
This locks in the empirical finding so it cannot silently become stale (e.g. if canonical.json ever
shrinks enough that some digit genuinely stops colliding, this test should start failing and prompt
someone to re-measure the claim, not stay green on an assumption nobody re-checks).
"""
import pytest

import claims as C


def test_every_single_digit_is_a_live_token_and_already_covered_without_the_benign_rule():
    """Asserting only that no digit shows up in orphans()'s `out` dict would be trivially satisfied
    if a digit simply never appeared as a standalone token in the prose at all (a vacuous pass, not
    proof of the actual claim). This test explicitly asserts BOTH halves: (1) each digit 0-9
    genuinely appears as a live token in the real draft_ssrn.md+note_ssrn.md body today, AND (2) with the
    BENIGN single-digit rule removed, none of them show up as orphaned -- proving the rule adds no
    actual detection power on top of what the corpus already silently absorbs, not just that the
    digits happen to be absent from the prose."""
    canon = C._canon()
    sections = C.index_draft()

    body = C._strip_covid19(C.load_body())
    live_tokens = set(C.tokenise(body))
    present_digits = live_tokens & set("0123456789")
    assert present_digits == set("0123456789"), (
        f"expected all 10 digits to appear as standalone tokens in the live manuscript; missing "
        f"{set('0123456789') - present_digits} -- if a digit genuinely stopped appearing, the "
        f"claim in this file's own module docstring needs re-measuring, not just re-asserting"
    )

    digit_rule = [pat for pat in C.BENIGN if pat[0] == r"^\d$"]
    assert len(digit_rule) == 1, (
        "expected to find exactly one BENIGN entry for the single-digit pattern -- if this rule's "
        "own regex changed, update this test's assumption about which entry to remove"
    )
    benign_without_digit_rule = [pat for pat in C.BENIGN if pat[0] != r"^\d$"]

    orig_benign = C.BENIGN
    try:
        C.BENIGN = benign_without_digit_rule
        out, _n_tok, _n_benign = C.orphans(canon, sections)
    finally:
        C.BENIGN = orig_benign

    orphaned_digits = {tok for tok in out if tok in set("0123456789")}
    assert orphaned_digits == set(), (
        f"{orphaned_digits} became genuinely orphaned once the BENIGN single-digit rule was "
        f"removed -- corpus density no longer silently saturates every digit 0-9. This is GOOD "
        f"NEWS for detection power, but the `^\\d$` entry's own justification in claims.py's "
        f"BENIGN table ('corpus density alone already makes every digit 0-9 indistinguishable "
        f"from an unrelated canonical value', 'does not add or remove any actual detection "
        f"power') is now stale and must be updated to match, not left describing a state that no "
        f"longer holds."
    )


def test_digit_rule_regex_still_matches_only_bare_single_digits():
    """Sanity check on the rule's own scope -- it must not accidentally widen to multi-digit
    numbers, which would silence real results far more aggressively than intended."""
    pattern = next(pat for pat, _why in C.BENIGN if pat == r"^\d$")
    import re
    assert re.match(pattern, "7")
    assert not re.match(pattern, "17")
    assert not re.match(pattern, "70")


# Generalizing the single-digit rule's honesty fix above to the rest of the BENIGN table.
#
# EVERY BENIGN entry is classified into exactly one of the two lists below (plus the one
# self-tested exemption named further down) by the SAME removal test the parametrized cases
# perform: drop the entry from BENIGN, re-run orphans() against the live corpus, and see whether
# UNRESOLVED moves. Every entry turned out to be measurable this way -- there is no residual
# "too broad to measure" category.
#
#   REDUNDANT   -- removing it leaves UNRESOLVED unchanged: the token is already absorbed by an
#                  unrelated canonical value's own rendering, so the rule adds zero marginal
#                  detection power today. The entries are kept (not deleted) because each is
#                  individually defensible and the incidental collision is not guaranteed to
#                  persist.
#   LOAD_BEARING -- removing it RAISES UNRESOLVED: no other canonical value's rendering collides
#                  with that exact token, so the rule is genuinely doing work.
#
# A rule's status can move in either direction when the corpus changes (a regenerated value that
# starts or stops rendering as the same token), which is exactly why both directions are pinned:
# a pattern drifting out of its list fails here rather than silently going stale. This file's
# convention is that EVERY BENIGN entry is classified into exactly one of the two lists -- enforced
# by test_every_benign_pattern_is_classified() below, so a newly-added entry cannot silently escape
# the removal-test methodology. (Previously the convention was only STATED: 23 of the 44 entries
# were listed and 21 -- including `^110$`, `^1\.96$`, `^64$`, `^21$` and `^399$` -- were silently
# uncovered, with nothing asserting completeness.)
#
# The one documented exemption is `^\d$` itself, which has its own dedicated pair of tests at the
# top of this file; it measures REDUNDANT, and listing it again here would duplicate them.
#
# Note for anyone re-measuring by hand: benign()'s own `bare = tok.lstrip("+")` strips a leading
# sign before matching, so e.g. "+0.34" DOES match `^0\.34$` -- match against the normalized form,
# not the raw token, or a rule will look dormant when it is merely redundant.
SELF_TESTED_PATTERNS = [r"^\d$"]

REDUNDANT_PATTERNS = [r"^0\.(05|10|25)$", r"^0\.50$", r"^0\.3$", r"^0\.003$", r"^0\.34$",
                      r"^2,000$", r"^5,000$", r"^48,000$", r"^6\.5$", r"^1\.00$",
                      # measured in the 2026-09 completeness pass, same removal methodology
                      r"^1,500$", r"^118$", r"^60$", r"^21$"]
# The `^(111|312)$` / `^3,314$` / `^98$` / `^(122|388)$` / `^0\.007$` / `^3,417$` / `^3,305$` /
# `^(102|103)$` / `^0\.91$` entries back verification statements in Sections 5 and 7 that are
# sourced from paper/external_facts.json rather than canonical.json.
LOAD_BEARING_PATTERNS = [r"^0\.60$", r"^1,000$", r"^20,000$", r"^25,000$",
                         r"^(111|312)$", r"^3,314$", r"^98$", r"^(122|388)$", r"^0\.007$",
                         r"^3,417$", r"^3,305$", r"^(102|103)$", r"^0\.91$",
                         # measured in the 2026-09 completeness pass, same removal methodology
                         r"^(19|20)\d{2}$", r"^(0001|0009|2447|0336)$", r"^10\.5281$", r"^110$",
                         r"^1\.96$", r"^(100|200|300|400|500|700|1000|2000|5000|800)$", r"^64$",
                         r"^615\.1[12]$", r"^7503218$", r"^0\d{2}$", r"^127$",
                         r"^\d{4}\.\d{5}$", r"^(237|245)$", r"^256$", r"^(16|32)$", r"^399$"]


def test_every_benign_pattern_is_classified():
    """Enforces the convention stated above: no BENIGN entry may sit outside the removal-test
    methodology. Every pattern must appear in exactly one of REDUNDANT_PATTERNS,
    LOAD_BEARING_PATTERNS or SELF_TESTED_PATTERNS, and none of those lists may name a pattern that
    is no longer in BENIGN."""
    live = [pat for pat, _why in C.BENIGN]
    assert len(live) == len(set(live)), f"BENIGN contains duplicate patterns: {live}"

    classified = REDUNDANT_PATTERNS + LOAD_BEARING_PATTERNS + SELF_TESTED_PATTERNS
    dupes = [p for p in set(classified) if classified.count(p) > 1]
    assert dupes == [], f"pattern(s) classified in more than one list: {dupes}"

    unclassified = [p for p in live if p not in classified]
    assert unclassified == [], (
        f"{len(unclassified)} BENIGN pattern(s) are not classified as REDUNDANT or LOAD_BEARING: "
        f"{unclassified} -- run the removal test (drop the entry from BENIGN, re-run orphans() on "
        f"the live corpus, see whether UNRESOLVED moves) and add each to the right list"
    )
    stale = [p for p in classified if p not in live]
    assert stale == [], (
        f"pattern(s) classified here but no longer present in claims.py's BENIGN table: {stale}"
    )


@pytest.mark.parametrize("pattern", REDUNDANT_PATTERNS)
def test_redundant_benign_entries_still_add_zero_marginal_detection_power(pattern):
    canon = C._canon()
    sections = C.index_draft()
    baseline, _n_tok, _n_benign = C.orphans(canon, sections)

    orig_benign = C.BENIGN
    try:
        C.BENIGN = [p for p in C.BENIGN if p[0] != pattern]
        assert len(C.BENIGN) == len(orig_benign) - 1, f"expected exactly one BENIGN entry for {pattern!r}"
        out, _n_tok2, _n_benign2 = C.orphans(canon, sections)
    finally:
        C.BENIGN = orig_benign

    assert len(out) == len(baseline), (
        f"removing {pattern!r} changed UNRESOLVED from {len(baseline)} to {len(out)} -- this "
        f"pattern is no longer redundant with corpus collision; update its comment in claims.py "
        f"(it may now be doing real work, which is good news, but the comment must reflect it)"
    )


@pytest.mark.parametrize("pattern", LOAD_BEARING_PATTERNS)
def test_load_bearing_benign_entries_still_add_real_detection_power(pattern):
    """The mirror image of
    test_redundant_benign_entries_still_add_zero_marginal_detection_power above -- locks in that a
    pattern DOCUMENTED as load-bearing (not redundant) still genuinely is, so this status can't
    silently drift the other way either (e.g. if the corpus changes again and some other canonical
    value starts coincidentally colliding with this token, making the rule redundant again without
    anyone noticing and updating the comment)."""
    canon = C._canon()
    sections = C.index_draft()
    baseline, _n_tok, _n_benign = C.orphans(canon, sections)

    orig_benign = C.BENIGN
    try:
        C.BENIGN = [p for p in C.BENIGN if p[0] != pattern]
        assert len(C.BENIGN) == len(orig_benign) - 1, f"expected exactly one BENIGN entry for {pattern!r}"
        out, _n_tok2, _n_benign2 = C.orphans(canon, sections)
    finally:
        C.BENIGN = orig_benign

    assert len(out) > len(baseline), (
        f"removing {pattern!r} left UNRESOLVED unchanged ({len(baseline)}) -- this pattern is no "
        f"longer load-bearing (some canonical value now coincidentally collides with its token "
        f"again); move it back to REDUNDANT_PATTERNS and update its comment in claims.py"
    )

"""Regression tests for the claims.py logic that drove the ORPHAN backlog to zero and
MISPLACED/unknown-location down to a small, understood residual. Each fix here demonstrated a REAL
false-positive orphan on the live manuscript before it was added; these tests lock in the
mechanism, not just the current manuscript's content.

Covers: the COVID-19 substring strip, the subsection-cross-reference exclusion (bare "N.N" tokens
that are themselves a valid section id, e.g. a prose reference to "Section 3.4"), the aux_max/
n_qualifying extra-field coverage in orphans(), and the new narrow BENIGN entries (arXiv id,
M4 submission id, SHA-256, the legacy neural exhibit's C(7,2) = 21 pairwise loss differentials --
this entry was originally justified as "the k* test count" and was rewritten -- and the two LLM
pool-size labels).
"""
import json

import pytest

import claims as C


def _canon_with(**values):
    """Build a minimal canonical dict: key -> {"value": v, plus any extra kwargs}."""
    canon = {}
    for k, v in values.items():
        canon[k] = {"value": v, "producer": "test", "cites": [], "cross_check": None}
    return canon


# --------------------------- COVID-19 strip ---------------------------

def test_strip_covid19_removes_only_the_disease_name_suffix():
    assert C._strip_covid19("the abrupt COVID-19 shock") == "the abrupt COVID shock"
    # a genuine bare "19" elsewhere in the same body must survive untouched
    assert C._strip_covid19("19 states were affected") == "19 states were affected"


def test_covid19_suffix_is_stripped_before_tokenisation():
    """Mechanism-level reproduction of the real false positive: 'COVID-19' tokenised '19' as a bare
    orphan number (the disease name's suffix, not a data value) before this fix. orphans() applies
    _strip_covid19() to the body internally, so it cannot express the before/after contrast; this
    test asserts on tokenise() either side of the strip instead."""
    body = "the abrupt, single-month COVID-19 claims/labor-market shock."
    toks_before_strip = set(C.tokenise(body))
    assert "19" in toks_before_strip, "the reproduction depends on tokenise() extracting 19 from COVID-19"
    stripped = C._strip_covid19(body)
    toks_after_strip = set(C.tokenise(stripped))
    assert "19" not in toks_after_strip, "COVID-19's suffix must not tokenise as a bare orphan number"


# --------------------------- subsection cross-reference exclusion ---------------------------

def test_subsection_cross_reference_is_not_flagged_as_orphan(tmp_path, monkeypatch):
    """A bare 'N.N' token that matches an INDEXED section id (e.g. prose reading '...the
    complement to 3.3...' where '**3.3 ...**' is a real subsection) is a cross-reference, not an
    orphaned data value -- the real false positive this fix closed on Section 3's own self-references."""
    draft = tmp_path / "draft.md"
    draft.write_text(
        "## 3. Data\n\n"
        "**3.3 The state panel.** Some content here.\n\n"
        "**3.4 National UNRATE.** Used for the complement to 3.3 described above.\n\n"
        "## 4. Results\n\nSome results text with no numbers of interest.\n"
    )
    canon_path = tmp_path / "canonical.json"
    canon_path.write_text(json.dumps({"test.dummy": {"value": 42, "cites": [], "producer": "test"}}))

    monkeypatch.setattr(C, "DRAFT", draft)
    monkeypatch.setattr(C, "CANON", canon_path)
    monkeypatch.setattr(C, "_DRAFT_TEXT_CACHE", None)
    monkeypatch.setattr(C, "_CANON_CACHE", None)

    canon = C._canon()
    sections = C.index_draft()
    assert "S3.3" in sections and "S3.4" in sections
    orph, _n_tok, _n_benign = C.orphans(canon, sections)
    assert "3.3" not in orph, "a cross-reference to an indexed subsection must not orphan"


def test_subsection_cross_reference_exclusion_requires_the_sections_arg():
    """Without a sections index passed in (the old 1-arg call signature, still used by
    tests/pipeline/test_claims_reads_draft_md.py), the cross-reference exclusion must simply not
    apply -- backward compatible, not a crash -- so a bare 'N.N' token can still orphan normally."""
    canon = _canon_with(**{"some.other_value": 1})
    orph, _n_tok, _n_benign = C.orphans(canon)  # sections omitted, defaults to None
    assert "3.3" in orph


def test_subsection_cross_reference_does_not_swallow_a_real_value_at_an_unindexed_number(tmp_path, monkeypatch):
    """Safety check on the exclusion's own scope: a bare 'N.N' token must still orphan normally
    when it does NOT match any indexed location, even if it superficially looks like a subsection
    number (e.g. '9.9' with no Section 9.9 ever drafted)."""
    draft = tmp_path / "draft.md"
    draft.write_text("## 3. Data\n\nSome text.\n\n## 4. Results\n\nAn unrelated figure of 9.9 appears here.\n")
    canon_path = tmp_path / "canonical.json"
    canon_path.write_text(json.dumps({"test.dummy": {"value": 42, "cites": [], "producer": "test"}}))
    monkeypatch.setattr(C, "DRAFT", draft)
    monkeypatch.setattr(C, "CANON", canon_path)
    monkeypatch.setattr(C, "_DRAFT_TEXT_CACHE", None)
    monkeypatch.setattr(C, "_CANON_CACHE", None)

    canon = C._canon()
    sections = C.index_draft()
    assert "S9.9" not in sections
    orph, _n_tok, _n_benign = C.orphans(canon, sections)
    assert "9.9" in orph, "a token resembling a subsection id must still orphan when no such section exists"


# --------------------------- aux_max / n_qualifying extra-field coverage ---------------------------

def test_aux_max_extra_field_covers_its_own_rendering():
    canon = _canon_with(**{"comb.sn_worst_ratio_min": 1.6821289218337883})
    canon["comb.sn_worst_ratio_min"]["aux_max"] = 7.4369670493616225
    orph, _n_tok, _n_benign = C.orphans(canon)
    assert "7.4" not in orph, "aux_max must be covered the same way numerator/denominator/n are"


def test_n_qualifying_extra_field_covers_its_own_rendering():
    canon = _canon_with(**{"mcspower.conditional_power_10pct": 0.8545})
    canon["mcspower.conditional_power_10pct"]["n_qualifying"] = 55
    orph, _n_tok, _n_benign = C.orphans(canon)
    assert "55" not in orph, "n_qualifying must be covered the same way other extra fields are"


def test_sd_extra_field_covers_its_own_rendering():
    """The note's S4 quotes measured seed-to-seed `sd` fields in prose ("the observed value's own
    seed-to-seed sd is 0.011" -- llm.winner_stability.sd = 0.0113). That rendering must be covered
    the same way aux_max/n_qualifying are.

    FALSIFIABILITY NOTE: orphans() keys its result by tokens of the REAL manuscript, so a
    `not in orph` assertion is only meaningful for a token the manuscript actually contains --
    otherwise it can never fail whatever claims.py does. The earlier version of this test asserted
    on "0.008" (sd 0.0083), which occurs ZERO times in either manuscript, so deleting `"sd", ` from
    the extra-field tuple in orphans() left it green. "0.011" IS a live token (it renders
    llm.winner_stability.sd in the note's S4 prose), it is not matched by any BENIGN rule, and it is
    not among renderings(0.672) -- so the ONLY thing that can cover it here is the `sd` rule."""
    canon = _canon_with(**{"llm.winner_stability": 0.672})
    canon["llm.winner_stability"]["sd"] = 0.0113
    orph, _n_tok, _n_benign = C.orphans(canon)
    assert "0.011" not in orph, "sd must be covered the same way other extra fields are"


def test_band_extra_field_covers_its_own_rendering():
    """election.stability_null_p05 really does ship a `band`, and this is its own band -- unlike
    eb.knife_edge_pct, which has no band field at all.

    FALSIFIABILITY NOTE: same trap as the `sd` test above. The earlier version used
    llm.winner_stability_null_p05's band [0.129, 0.16]; neither "0.129" nor "0.16" is a token of
    either manuscript, so the assertion could not fail. "0.307" (this band's upper endpoint) IS a
    live token, is not BENIGN, and is not among renderings(0.298), so only the `band` rule can
    cover it. (The lower endpoint 0.282 is NOT a live token, so it is deliberately not asserted
    on -- asserting it would merely re-add vacuous coverage.)"""
    canon = _canon_with(**{"election.stability_null_p05": 0.298})
    canon["election.stability_null_p05"]["band"] = [0.282, 0.307]
    orph, _n_tok, _n_benign = C.orphans(canon)
    assert "0.307" not in orph, "band must be covered the same way other extra fields are"


# --------------------------- new BENIGN entries ---------------------------

@pytest.mark.parametrize("tok,why_substr", [
    ("2508.11847", "arXiv"),
    ("118", "M4"),
    ("256", "SHA-256"),
    # claims.py's `^21$` justification was rewritten (it is no longer "a test count" but the legacy
    # neural exhibit's C(7,2) = 21 pairwise loss differentials); the expected substring tracks it.
    ("21", "pairwise loss differentials"),
    ("16", "pool size"),
    ("32", "pool size"),
])
def test_new_benign_entries_match_and_have_a_real_justification(tok, why_substr):
    """Each token must not merely match SOME BENIGN rule, but match the one whose justification is
    the intended one -- an unrelated rule quietly absorbing the token would otherwise satisfy a
    bare "matched something" check while the entry this test exists to guard went untested."""
    import re
    bare = tok.lstrip("+")
    hit = None
    for pat, why in C.BENIGN:
        if re.match(pat, bare):
            hit = why
            break
    assert hit is not None, f"{tok!r} must match a BENIGN rule"
    assert len(hit) > 10, "the justification must be a real explanation, not a placeholder"
    assert why_substr in hit, (
        f"{tok!r} matched a BENIGN rule justified as {hit!r}, which is not the intended "
        f"{why_substr!r} entry -- either the intended entry was removed/reworded, or an unrelated "
        f"rule earlier in the table now shadows it"
    )


def test_benign_16_32_do_not_blanket_allow_every_two_digit_number():
    """Narrow-scope check: the (16|32) rule must not accidentally swallow other two-digit numbers
    that happen to share a leading digit pattern -- e.g. 17 or 31 must NOT match."""
    import re
    # TWO BENIGN entries mention "pool size" (the `^64$` largest-pool entry and this `^(16|32)$`
    # one), and `^64$` comes first -- select on the rule that actually matches 16, not on the
    # justification text alone, or this test silently guards the wrong rule.
    candidates = [p for p, why in C.BENIGN if "pool size" in why and re.match(p, "16")]
    assert len(candidates) == 1, (
        f"expected exactly one BENIGN pool-size rule matching '16', found {candidates!r}"
    )
    pat = candidates[0]
    assert re.match(pat, "16") and re.match(pat, "32")
    assert not re.match(pat, "17")
    assert not re.match(pat, "31")
    assert not re.match(pat, "160")   # must not match as a prefix of a longer number

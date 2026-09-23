"""Unit tests for the verification layer (canonical._close(), claims
tokenise/orphans/misplaced/BENIGN), which previously had none:

  * canonical._close()'s tolerance: no test pinned `abs(a-b) <= max(1e-9, |b|*1e-6)`,
    so loosening the relative tolerance to 4e-2 passed the selftest AND the suite. Pinned here with
    a ~4% relative change (11 -> 11.44), a 1e-3 relative change, the 1e-9 absolute floor, NaN==NaN
    and the string fallback.
  * claims.tokenise()/orphans() sign rule on the BARE-number path: a prose "0.62"
    against a canonical -0.62 must be an ORPHAN and "-0.62" must be covered; the U+2212 typographic
    minus BYPASSES this rule today -- recorded as two strict xfails so the day
    tokenise() learns `[+\\-−]?` they flip to XPASS and must be promoted. (Those two are
    defect pins, NOT mutation kills.)
  * MISPLACED accepting a value that appears nowhere (0.534 passes on "rho=0.5"): the accepted
    renderings of 0.534 are pinned as exactly {0.534, 0.53, 0.5} on the bare path, and the
    NEIGHBOURING tokens 0.54 / 0.6 / 5.34 / 1 / 53 are asserted MISPLACED, so any widening of
    renderings() or a leak of the percent set into the bare comparison is caught.
  * BENIGN allowlist scope: every pattern must be fully anchored (`^...$`) and the
    load-bearing entries must not swallow a digit-extended, decimal-extended or SIGN-FLIPPED
    neighbour of their literal; exercised through the real orphans() closure with an EMPTY canon so
    nothing is covered by corpus collision. Complements (does not duplicate)
    tests/gates/test_claims_single_digit_detection_power.py, which measures marginal detection power
    on the live corpus; this file is corpus-independent.

All fixtures are synthetic; claims.py's manuscript paths are redirected to tmp_path (same
_patch_body scaffold as tests/gates/test_claims_orphans_companion_array_allowlist.py). Relies on
tests/conftest.py for sys.path.
"""
import pytest

import canonical as K  # noqa: E402  (path via tests/conftest.py)
import claims as C  # noqa: E402


# --- canonical._close() ---------------------------------------------------------------------------

@pytest.mark.parametrize("a,b", [
    (11, 11.44),                 # +4% relative: a loosened relative tolerance (1e-6 -> 4e-2)
    (11, 12),                    # an integer headline slip: mcs.curated_nonidentified 11 -> 12
    (1.0, 1.001),                # 1e-3 relative (selftest's "+1e-3" mutation never reaches _close)
    (1.0, 1.00001),              # 1e-5 relative, still far outside 1e-6
    (0.534, 0.5),                # a value vs its 1-dp prose rendering (0.534 vs "0.5")
    (0.001, 0.0015),             # the 1e-9 absolute floor must NOT be 1e-3: 50% of a small value
    (0.0, 2e-9),                 # just outside the absolute floor
    (float("nan"), 1.0),         # NaN vs a real number is a difference
    (-0.62, 0.62),               # sign flip
])
def test_close_rejects_material_differences(a, b):
    """kills: loosening `abs(fb) * 1e-6` to `* 4e-2`; raising `max(1e-9, ...)` to
    `max(1e-3, ...)`; and a NaN-blind `return True` shortcut."""
    assert not K._close(a, b), f"_close({a!r}, {b!r}) must be False"
    assert not K._close(b, a), f"_close({b!r}, {a!r}) must be False"


@pytest.mark.parametrize("a,b", [
    (1.0, 1.0 + 5e-7),           # 5e-7 relative: inside 1e-6
    (1e6, 1e6 + 0.5),            # 5e-7 relative on a large value
    (0.0, 5e-10),                # inside the 1e-9 absolute floor
    (float("nan"), float("nan")),
    (11, 11.0),
    ("fragile", "fragile"),      # string fallback
])
def test_close_accepts_only_float_noise_nan_pairs_and_equal_strings(a, b):
    """kills: `<= max(1e-9, |b|*1e-6)` -> `<= 0` (exact equality) or `1e-6` -> `1e-9`; the NaN
    special case removed."""
    assert K._close(a, b) and K._close(b, a)


def test_close_string_fallback_is_exact_not_close():
    assert not K._close("fragile", "not_fragile")
    assert not K._close("RSAFS", "rsafs")


# --- claims: the sign rule on the bare-number path, end to end through orphans() -----------------

def _patch_body(monkeypatch, tmp_path, body: str):
    draft = tmp_path / "draft.md"
    draft.write_text(f"## 1. Section\n\n{body}\n")
    monkeypatch.setattr(C, "DRAFT", draft)
    monkeypatch.setattr(C, "NOTE", tmp_path / "does_not_exist_note.md")
    monkeypatch.setattr(C, "_DRAFT_TEXT_CACHE", None)
    monkeypatch.setattr(C, "_NOTE_TEXT_CACHE", None)


def _entry(value, cites=()):
    return {"value": value, "producer": "test", "cites": list(cites), "cross_check": None}


def test_orphans_keeps_the_ascii_minus_as_part_of_the_claim(monkeypatch, tmp_path):
    """kills: stripping the minus at orphans()'s COVERED-CHECK -- the `bare = tok.lstrip("+")` that
    is immediately followed by `covered = tok in known or bare in known`, NOT the identically-spelled
    line inside the nested `benign()` helper a few lines above it (they are quoted here rather than
    cited by line number precisely because those two lines collide) -- and `[+-]?` -> `[+]?` in BARE
    (a dropped sign would tokenise "-0.62" as "0.62").
    Canonical value -0.62: prose "-0.62" is covered; prose "0.62" (sign lost) is an ORPHAN; a
    leading "+" on a positive value is not part of the claim; a prose "-0.31" against a canonical
    +0.31 is an ORPHAN."""
    canon = {"t.neg": _entry(-0.62), "t.pos": _entry(0.31)}
    _patch_body(monkeypatch, tmp_path, "The correlation is -0.62 here.")
    out, _n, _b = C.orphans(canon)
    assert out == {}, f"the correctly-signed rendering must be covered, got {out}"

    _patch_body(monkeypatch, tmp_path, "The correlation is 0.62 here.")
    out, _n, _b = C.orphans(canon)
    assert "0.62" in out, "a sign-dropped rendering of a negative canonical value must be an ORPHAN"

    _patch_body(monkeypatch, tmp_path, "The correlation is +0.31 here.")
    out, _n, _b = C.orphans(canon)
    assert out == {}, "a leading '+' must not block the match against a positive canonical value"

    # the historical bug recorded by orphans()'s own comment -- "A leading PLUS must not block the
    # match; a leading MINUS must, because the sign is part of the claim. Stripping both made a
    # prose -0.62 resolve against a canonical +0.62." -- in the same terms: a prose -0.31 must NOT
    # resolve against the canonical +0.31.
    _patch_body(monkeypatch, tmp_path, "The correlation is -0.31 here.")
    out, _n, _b = C.orphans(canon)
    assert "-0.31" in out, "a sign-flipped prose value must be an ORPHAN, not resolved sign-blind"


@pytest.mark.xfail(strict=True, reason="claims.tokenise() recognises "
                   "only the ASCII '-', so a U+2212 typographic minus tokenises sign-less and the sign "
                   "rule is bypassed. Defect pin, not a mutation kill: promote to a plain test when "
                   "tokenise() accepts [+\\-−]?.")
def test_orphans_treats_the_typographic_minus_as_a_sign(monkeypatch, tmp_path):
    canon = {"t.pos": _entry(0.62)}
    _patch_body(monkeypatch, tmp_path, "The correlation is −0.62 here.")
    out, _n, _b = C.orphans(canon)
    assert out, "a U+2212-signed prose value must NOT be covered by the unsigned canonical value"


@pytest.mark.xfail(strict=True, reason="election.houst_own_ar1_median "
                   "= -0.009 is MISPLACED against the prose '(−0.01)' only because the U+2212 "
                   "minus is not tokenised as a sign. Defect pin, not a mutation kill.")
def test_misplaced_finds_a_negative_value_written_with_the_typographic_minus():
    canon = {"t.neg": _entry(-0.009, ["S5.4 body: AR(1)-adjusted"])}
    bad, unknown, checked, uncited = C.misplaced(canon, {"S5.4": "the own-series AR(1) median (−0.01) is small."})
    assert bad == []


def test_misplaced_finds_a_negative_value_written_with_the_ascii_minus():
    """Control for the xfail above: the same sentence with an ASCII minus is found (2-dp rendering
    of -0.009 is -0.01)."""
    canon = {"t.neg": _entry(-0.009, ["S5.4 body: AR(1)-adjusted"])}
    bad, _u, checked, _uc = C.misplaced(canon, {"S5.4": "the own-series AR(1) median (-0.01) is small."})
    assert bad == [] and checked == 1
    bad, _u, _c, _uc = C.misplaced(canon, {"S5.4": "the own-series AR(1) median (0.01) is small."})
    assert len(bad) == 1, "the sign-dropped rendering must be MISPLACED"


# --- claims: MISPLACED's rounding leniency is bounded ---------------------------------------------

def test_renderings_of_0_534_on_the_bare_path_are_exactly_its_documented_roundings():
    """kills: renderings() gaining an nd=0 rounding for |v| < 10 (0.534 -> "1"), or any other
    widening; and the percent set leaking into the bare set. The bare renderings of 0.534 are
    exactly {0.534, 0.53, 0.5} -- the "0.5" member is the 1-dp leniency ("rho=0.5" satisfies a
    cite of 0.534) and is pinned here as CURRENT behaviour so a change in either direction is a
    deliberate one."""
    bare = C.renderings(0.534) - C.percent_renderings(0.534)
    assert bare == {"0.534", "0.53", "0.5"}
    assert C.percent_renderings(0.534) == {"53", "53.4", "53.40"}


@pytest.mark.parametrize("prose", [
    "the AR(1)-adjusted null median is 0.54 across series",     # 2-dp neighbour (0.534 -> 0.53)
    "the AR(1)-adjusted null median is 0.6 across series",      # 1-dp neighbour
    "the AR(1)-adjusted null median is 5.34 across series",     # decimal shift
    "the AR(1)-adjusted null median is 1 across series",        # nd=0 rounding is NOT a rendering
    "the AR(1)-adjusted null median is 53 across series",       # its percent form, without a % sign
    "the AR(1)-adjusted null median is 0.5340 across series",   # over-precise, not a rendering
])
def test_misplaced_rejects_neighbours_of_0_534(prose):
    """kills: `want_bare = want - want_pct` -> `want_bare = want` in misplaced() (the "53" case);
    renderings() nd=0 widening (the "1" case); a `found = True` short-circuit."""
    canon = {"election.stability_null_median_ar1": _entry(0.534, ["S5.4 body: AR(1)-adjusted null"])}
    bad, unknown, checked, uncited = C.misplaced(canon, {"S5.4": prose})
    assert len(bad) == 1 and bad[0].startswith("election.stability_null_median_ar1 = 0.534 is not in S5.4"), (
        f"{prose!r} must NOT satisfy a cite of 0.534, got bad={bad}")
    assert unknown == [] and checked == 1 and uncited == 0


@pytest.mark.parametrize("prose", [
    "the AR(1)-adjusted null median is 0.534 across series",
    "the AR(1)-adjusted null median is 0.53 across series",
    "the AR(1)-adjusted null median is 53% across series",      # percent rendering WITH the % sign
    "the AR(1)-adjusted null median is 0.5 across series",      # the 1-dp leniency, pinned as current
])
def test_misplaced_accepts_the_documented_renderings_of_0_534(prose):
    canon = {"election.stability_null_median_ar1": _entry(0.534, ["S5.4 body: AR(1)-adjusted null"])}
    bad, _u, checked, _uc = C.misplaced(canon, {"S5.4": prose})
    assert bad == [] and checked == 1


# --- claims: BENIGN allowlist scope -----------------------------------------------------------------

def test_every_benign_pattern_is_fully_anchored():
    """kills: any BENIGN entry losing its `$` (re.match anchors the start only, so `^98` would
    swallow 98.5, 980, 9800...)."""
    for pat, _why in C.BENIGN:
        assert pat.startswith("^") and pat.endswith("$"), f"BENIGN pattern {pat!r} is not fully anchored"


# (load-bearing literal, tokens that must NOT be benign)
_NEIGHBOURS = [
    ("98",    ["980", "98.5", "9.8", "-98"]),
    ("122",   ["1220", "122.5", "12.2", "-122"]),
    ("388",   ["3880", "388.0", "38.8", "-388"]),
    ("3,314", ["33,314", "3314.5"]),
    ("0.007", ["0.0070", "0.07", "0.008", "-0.007"]),
    ("111",   ["1111", "11.1", "111.5", "-111"]),
    ("312",   ["3120", "31.2", "312.5", "-312"]),
    ("110",   ["1100", "11.0", "110.5", "-110"]),
    ("64",    ["640", "6.4", "64.5", "-64"]),
    ("0.60",  ["0.600", "0.61", "-0.60"]),
    ("6.5",   ["6.55", "65", "-6.5"]),
]


@pytest.mark.parametrize("literal,neighbours", _NEIGHBOURS)
def test_benign_swallows_its_literal_but_not_digit_decimal_or_sign_neighbours(literal, neighbours, monkeypatch, tmp_path):
    """kills: `^98$` -> `^98` (anchor dropped); benign()'s `bare = tok.lstrip("+")` ->
    `lstrip("+-")` (a sign-flipped -98 silently benign). Run through the REAL orphans() closure
    with an EMPTY canon, so the only way a token disappears from `out` is the BENIGN rule."""
    body = " and ".join([f"value {literal}"] + [f"value {n}" for n in neighbours]) + "."
    _patch_body(monkeypatch, tmp_path, body)
    out, n_tok, n_benign = C.orphans({})
    toks = set(C.tokenise(C.load_body()))
    assert literal in toks and all(n in toks for n in neighbours), (
        f"fixture error: not every probe token survives tokenise(): {toks}")
    assert literal not in out, f"{literal!r} is a documented BENIGN literal and must be swallowed"
    leaked = [n for n in neighbours if n not in out]
    assert leaked == [], f"BENIGN swallowed neighbour(s) {leaked} of {literal!r} -- scope has widened"
    # the disclosed benign count must account for exactly the literal
    assert n_benign == 1

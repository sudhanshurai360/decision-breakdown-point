#!/usr/bin/env python3
"""
claims.py — the two-way map between the manuscript's numbers and the canonical object.
"""
import decimal
import hashlib
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent   # <repo>/code/gates/ -> <repo>
DRAFT = ROOT / "paper" / "draft_ssrn.md"
NOTE = ROOT / "paper" / "note_ssrn.md"
CANON = ROOT / "paper" / "canonical.json"
TABLE = ROOT / "paper" / "figures" / "table1_identification.md"
TABLE_P = ROOT / "paper" / "figures" / "tableP_permutation_null.md"
TABLE_Q = ROOT / "paper" / "figures" / "tableQ_ar1null.md"
TABLE_R = ROOT / "paper" / "figures" / "tableR_accuracy.md"


N_MISPLACED = 15   # accepted MISPLACED+unknown entries, in 4 classes: booleans the prose states in
                   # words; superseded arms (arch_gap_moving_block); companions quoted at a
                   # different rounding; and values the draft writes with a UNICODE MINUS (U+2212),
                   # which tokenise()'s ASCII [+-] character class does not read as a sign -- so a
                   # correctly quoted negative tokenises unsigned and never meets its own rendering
                   # (election.houst_own_ar1_median = -0.009, which S5.4 writes as U+2212 then
                   # "0.01": renderings() offers "-0.01", tokenise() sees "0.01")
                   # -- reviewed entry by entry, none a stale number

MISPLACED_SET_HASH = "ecb7c4424f15cc26f9ef06d46d3dc36a7b12da5d72cc3ea3729c818b6962ccfc"

_CANON_CACHE: dict | None = None


def _canon() -> dict:
    global _CANON_CACHE
    if _CANON_CACHE is None:
        _CANON_CACHE = {k: v for k, v in json.loads(CANON.read_text(encoding="utf-8")).items()
                         if not k.startswith("_")}
    return _CANON_CACHE


def _draft_exists() -> bool:
    return DRAFT.exists()


_DRAFT_TEXT_CACHE: str | None = None


def _draft_text() -> str:
    """The manuscript's full text."""
    global _DRAFT_TEXT_CACHE
    if _DRAFT_TEXT_CACHE is None:
        _DRAFT_TEXT_CACHE = _read_manuscript_text(DRAFT)
    return _DRAFT_TEXT_CACHE


_NOTE_TEXT_CACHE: str | None = None


def _note_text() -> str:
    """note_ssrn.md's full text, same literal-prose convention as _draft_text()."""
    global _NOTE_TEXT_CACHE
    if _NOTE_TEXT_CACHE is None:
        _NOTE_TEXT_CACHE = _read_manuscript_text(NOTE) if NOTE.exists() else ""
    return _NOTE_TEXT_CACHE


def _read_manuscript_text(path: Path) -> str:
    """Shared body for _draft_text()/_note_text(): read the manuscript as literal prose; warn if
    '{{' appears, since this file does not resolve templating and an unresolved placeholder would
    hide the number from ORPHAN scanning."""
    text = path.read_text(encoding="utf-8")
    if "{{" in text:
        print(f"  WARNING: {path.name} contains '{{{{' -- looks like {{key}} templating was "
              f"reintroduced, but claims.py reads this file as literal rendered prose and will "
              f"not resolve placeholders. Re-check _read_manuscript_text().", file=sys.stderr)
    return text

NOT_A_CLAIM = [
    (r"\b(19|20)\d{2}[-–]\d{2}(-\d{2})?\b",  "an ISO date or a year range, not two numbers"),
    (r"\bSections?\s+\d+(\.\d+)?([-–]\d+(\.\d+)?)?", "a cross-reference to a section, or a range of them"),
    (r"§\s*\d+(\.\d+)?([-–]\d+(\.\d+)?)?", "a §-prefixed cross-reference to a section"),
    (r"\(\s*\d\.\d\s*\)",                    "a bare parenthetical subsection reference"),
    (r"^\*\*\d+\.\d+\s+[A-Z]",               "a numbered subsection label"),
    (r"^#{1,6}\s+\d+(\.\d+)?\.?\s",          "a numbered heading"),
    (r"\bT\s*=\s*\d+\b",                     "a stated panel length in a formula"),
    (r"\bb\s*=\s*\d+\b",                     "a stated block length in a formula"),
]

BENIGN = [
    (r"^(19|20)\d{2}$",            "calendar year"),
    (r"^(0001|0009|2447|0336)$",   "a 4-digit group of the author's own ORCID "
                                    "(0009-0001-2447-0336) in the byline"),
    (r"^10\.5281$",                "DOI prefix (10.5281, Zenodo) in a byline/citation link, "
                                    "not a quantity"),
    (r"^\d$",                      "single digit — corpus density alone already makes every digit "
                                    "0-9 indistinguishable from an unrelated canonical value; this "
                                    "rule does not add or remove any actual detection power, it "
                                    "only makes the pass-through an explicit, disclosed no-op "
                                    "instead of an implicit one"),
    (r"^0\.(05|10|25)$",           "significance level we choose, a stated method parameter"),
    (r"^6\.5$",                    "statutory EB TUR trigger floor / the FOMC's 2012-14 "
                                    "forward-guidance UNRATE threshold"),
    (r"^0\.50$",                   "statutory Sahm-rule threshold"),
    (r"^110$",                     "statutory EB 110%-of-prior-year leg"),
    (r"^1\.96$",                   "standard-normal 5% critical value"),
    (r"^0\.3$",                    "stated definitional threshold, not a measurement"),
    (r"^0\.60$",                   "stated definitional threshold (FRAGILE_STABILITY_THRESH), not "
                                    "a measurement -- currently load-bearing, not redundant"),
    (r"^1\.00$",                   "definitional endpoint of winner_stability, not a measurement "
                                    "(it is a live token -- Table 1's PI row reads 1.00 -- but a "
                                    "canonical value already covers it, so this rule is redundant)"),
    (r"^(100|200|300|400|500|700|1000|2000|5000|800)$", "method parameter (draws, resamples, panels)"),
    (r"^1,000$",                   "N_PERM/n_boot-style method parameter (permutation/bootstrap draws)"),
    (r"^1,500$",                   "Table 1 winner_stability resamples per seed (social_choice.py n_boot=1500, 20 seeds), a stated method parameter"),
    (r"^2,000$",                   "B (MCS bootstrap replications), a stated method parameter"),
    (r"^5,000$",                   "flagship winner_stability n_boot, a stated method parameter"),
    (r"^20,000$",                  "method parameter (draws/resamples), not a measurement"),
    (r"^25,000$",                  "method parameter (draws/resamples), not a measurement"),
    (r"^48,000$",                  "M4 Monthly competition series count, a corpus-size fact quoted from the M4 dataset, not a canonical key"),
    (r"^64$",                      "largest pool size in the crowding-curve grid k in {2,4,8,16,32,64} (arena_crowding.py / ml_witness.py), a method parameter; the per-k results are the llm.crowd_*_k<k> keys"),
    (r"^615\.1[12]$",              "citation to 20 CFR 615.11/615.12, the governing EB regulation"),
    (r"^(111|312)$",               "P.L. 111-312, the public-law citation for the temporary 2010-13 third-year EB "
                                   "look-back -- external_facts.json public_law_111_312; a citation, not a quantity"),
    (r"^3,314$",                   "DOL trigger-notice state-weeks compared in the Section 7 verification statement -- "
                                   "external_facts.json dol_trigger_notice_comparison_2026_09_20 (a verification figure, "
                                   "not a pipeline output)"),
    (r"^0\.91$",                  "Pearson correlation of the abrupt and vol proxies in the Section 4.3 population regression -- "
                                   "external_facts.json fredmd_law_transform_sensitivity_2026_09_21 (a read-only check, not a pipeline output)"),
    (r"^7503218$",                 "the paper's SSRN Abstract ID, quoted in the note's Section 6 link -- an identifier, not a quantity"),
    (r"^3,417$",                   "DOL trigger-notice state-week rows parsed (67 notices x 51 jurisdictions), the base of the "
                                   "3,314 subset stated in Section 7 -- external_facts.json dol_trigger_notice_comparison_2026_09_20"),
    (r"^3,305$",                   "DOL trigger-notice rows whose ON/OFF call the implemented rule matches (3,305 of 3,314) -- "
                                   "external_facts.json dol_trigger_notice_comparison_2026_09_20"),
    (r"^(102|103)$",               "rows excluded from the DOL trigger-notice comparison (103 = 102 misprinted-notice rows + 1 DC row), "
                                   "stated in Section 7 -- external_facts.json dol_trigger_notice_comparison_2026_09_20"),
    (r"^98$",                      "share of DOL-notice rows whose 3-month TUR the implemented rule matches within 0.1pp "
                                   "-- external_facts.json dol_trigger_notice_comparison_2026_09_20"),
    (r"^(122|388)$",               "series-years with fewer than 12 months, of the panel's series-years -- the equal-vote "
                                   "sensitivity disclosed in Section 5; external_facts.json partial_year_sensitivity_2026_09_20"),
    (r"^0\.007$",                  "the ar1 model path's margin against the Sahm threshold at the 2001-05 origin, quoted in "
                                   "Section 7's knife-edge qualification -- external_facts.json sahm_knife_edge_margins_2026_09_20"),
    (r"^0\d{2}$",                  "zero-padded M4 submission id (e.g. submission-072), not a quantity"),
    (r"^127$",                     "nominal size of the PUBLISHED McCracken-Ng FRED-MD panel — an "
                                   "external fact about someone else's dataset, not a result of ours; "
                                   "the paper explicitly contrasts it with what we actually pull"),
    (r"^\d{4}\.\d{5}$",            "arXiv identifier (e.g. 2508.11847) — an external citation, not a "
                                   "computed value; the 4-digit/5-digit shape is specific enough to "
                                   "arXiv IDs that it will not collide with a real result"),
    (r"^118$",                     "M4 competition submission ID (the winning entry's accession "
                                   "number in the M4 archive) — an external fact about a specific "
                                   "competition entry, not a value this paper computed"),
    (r"^(237|245)$",               "M4 competition submission ID (a runner-up entry's accession "
                                   "number, cited in the margin table) — an external fact about a "
                                   "specific competition entry, not a value this paper computed"),
    (r"^60$",                      "count of M4 rivals the champion is compared against (61 "
                                   "submissions minus the champion itself) — arithmetic on "
                                   "m4sub.n_submissions, not an independent measurement"),
    (r"^256$",                     "SHA-256, the hash algorithm name quoted in Section 3.6's "
                                   "provenance disclosure — not a data value"),
    (r"^21$",                      "the legacy neural exhibit's 21 pairwise loss differentials — "
                                   "C(7,2) on its seven models, arithmetic rather than an "
                                   "independent measurement, and already carried as the n "
                                   "companion on flagship.ar1_coefficient_empirical (so this rule "
                                   "is currently inert: that companion resolves the token before "
                                   "the BENIGN list is consulted; kept only as a guard should the "
                                   "companion stop being exported)"),
    (r"^(16|32)$",                 "Open LLM Leaderboard crowding-null pool size (k=2,4,8,16,32,64 "
                                   "is the axis tested, not a result) — the other four pool sizes in "
                                   "this same axis (2,4,8,64) already resolve incidentally against "
                                   "unrelated canonical values, so this is narrowly scoped to the two "
                                   "that do not, rather than allowlisting the whole axis"),
    (r"^0\.003$",                  "Huang, Shen, Wei & Broderick (2025) arXiv:2508.11847's "
                                   "adversarially-chosen drop fraction that flips a Bradley-Terry "
                                   "leaderboard's top rank — external_facts.json "
                                   "leaderboardk_huang2025.drop_fraction_flips_top_rank "
                                   "(currently redundant with corpus collision)"),
    (r"^399$",                     "Walsh et al. (2014)'s trial count (the 399 RCTs whose median "
                                   "Fragility Index the paper reports) — external_facts.json "
                                   "walsh_2014_fragility_index.median_fi"),
    (r"^0\.34$",                   "Chodorow-Reich & Karabarbounis (2016)'s own reported SD of the "
                                   "difference between the real-time and the revised state total "
                                   "unemployment rate — external_facts.json "
                                   "chodorow_reich_karabarbounis_2016.tur_realtime_error_sd_0_34pp, "
                                   "whose quote states no averaging or smoothing basis, so it is "
                                   "not established to be on this paper's own monthly basis "
                                   "(currently redundant with corpus collision)"),
]

TITLE_ALIASES = ("Abstract", "Introduction", "Conclusion")


def _strip_code(t: str) -> str:
    return re.sub(r"`[^`]*`", "", t)


def _strip_html_comments(t: str) -> str:
    """Drop every HTML comment (e.g. the <!-- ASSERT ... --> convention used elsewhere in the
    manuscript)."""
    return re.sub(r"<!--.*?-->", " ", t, flags=re.S)


def _unwrap(t: str) -> str:
    return re.sub(r"(?<!\n)[ \t]*\n(?!\n)[ \t]*", " ", t)


def _strip_covid19(t: str) -> str:
    """Rewrite 'COVID-19' to 'COVID' before tokenising. Left in place, the trailing '-19'
    tokenises as a bare orphan number (the disease name, not a data value) -- the same class of
    false positive as the ASSERT-comment self-references _strip_html_comments handles, and handled
    the same way: strip the literal substring rather than allowlisting '19' globally, which would
    also hide a genuine future orphan integer 19."""
    return re.sub(r"COVID-19", "COVID", t)


def index_draft() -> dict[str, str]:
    """{location id: its text}, merged across the paper ('S'-prefixed, e.g. 'S4.1') and the note
    ('N'-prefixed)."""
    out = _index_headings(_draft_text(), "S")
    note_text = _note_text()
    if note_text:
        for key, text in _index_headings(note_text, "N").items():
            if key in out:
                raise RuntimeError(f"location collision: '{key}' indexed by both the paper and the note")
            out[key] = text
    return out


def _index_headings(text: str, prefix: str) -> dict[str, str]:
    """One manuscript's text -> {prefix+location id: its text}."""
    lines = text.split("## References")[0].splitlines()

    # Blocks that are their own location: a bold caption line through to the next blank line.
    blocks: dict[str, set[int]] = {}
    for i, ln in enumerate(lines):
        m = re.match(r"\*\*(Figure (\d+)|Table (\d+))\.", ln)
        if not m:
            continue
        key = f"{prefix}Fig{m.group(2)} caption" if m.group(2) else f"{prefix}Table {m.group(3)} note"
        j = i
        while j < len(lines) and (j == i or lines[j].strip()):
            j += 1
        blocks[key] = set(range(i, j))
    captioned = set().union(*blocks.values()) if blocks else set()

    heads: list[tuple[int, list[str], bool]] = []
    cur_major: str | None = None  # the innermost numbered ## section we're currently inside
    for i, ln in enumerate(lines):
        m = re.match(r"^#{2,3}\s+(?:(\d+(?:\.\d+)?)\.?\s+)?(.*)$", ln)
        if m:
            num, title = m.group(1), m.group(2).strip()
            keys = ([f"{prefix}{num}"] if num else [])
            if prefix == "S":
                keys += [t for t in TITLE_ALIASES if title == t]
            if keys:
                heads.append((i, keys, False))
            if num:
                cur_major = num.split(".")[0]
            continue
        m = re.match(r"^\*\*(\d{1,2}\.\d{1,2})\s+[A-Z]", ln)
        if m and cur_major is not None and m.group(1).split(".")[0] == cur_major:
            heads.append((i, [f"{prefix}{m.group(1)}"], True))

    out: dict[str, str] = {}
    for n, (i, keys, inline) in enumerate(heads):
        end = heads[n + 1][0] if n + 1 < len(heads) else len(lines)
        span = set(range(i if inline else i + 1, end)) - captioned
        text = _unwrap("\n".join(lines[k] for k in sorted(span)))
        for key in keys:
            if key in out:
                raise RuntimeError(f"_index_headings: duplicate location key '{key}' within one "
                                   f"manuscript (prefix={prefix!r}) -- a heading and a bold "
                                   f"subsection label (or two headings) produced the same location "
                                   f"id, silently shrinking the earlier one's checked span")
            out[key] = text
    for key, span in blocks.items():
        if key in out:
            raise RuntimeError(f"_index_headings: duplicate location key '{key}' within one "
                               f"manuscript (prefix={prefix!r})")
        out[key] = _unwrap("\n".join(lines[k] for k in sorted(span)))
    return out


def _strip_not_a_claim(t: str) -> str:
    """Remove text whose numbers are addresses rather than assertions -- dates, section references,
    numbered headings, stated formula parameters."""
    out = []
    for ln in t.splitlines():
        for pat, _ in NOT_A_CLAIM:
            ln = re.sub(pat, " ", ln)
        out.append(ln)
    return "\n".join(out)


def load_body() -> str:
    """The whole prose surface, for the ORPHAN direction -- draft_ssrn.md AND note_ssrn.md (if it
    exists) concatenated, since a number in either manuscript must trace to a canonical value."""
    t = _draft_text().split("## References")[0]
    for ln in _table_rows():
        t = t.replace(ln, "")
    note_t = _note_text().split("## References")[0]
    return _unwrap(_strip_code(_strip_not_a_claim(_strip_html_comments(t + "\n" + note_t))))


def _table_rows() -> list[str]:
    """The GENERATED Table 1 data rows, verbatim."""
    if not TABLE.exists():
        return []
    rows = [ln.rstrip() for ln in TABLE.read_text(encoding="utf-8").splitlines()
            if ln.startswith("| ") and not ln.startswith("|:")]
    if TABLE_P.exists():
        rows += [ln.rstrip() for ln in TABLE_P.read_text(encoding="utf-8").splitlines()
                 if ln.startswith("| ") and "---" not in ln]
    if TABLE_Q.exists():
        # Same header-inclusion reasoning as TABLE_P above.
        rows += [ln.rstrip() for ln in TABLE_Q.read_text(encoding="utf-8").splitlines()
                 if ln.startswith("| ") and "---" not in ln]
    if TABLE_R.exists():
        # Same header-inclusion reasoning as TABLE_P above.
        rows += [ln.rstrip() for ln in TABLE_R.read_text(encoding="utf-8").splitlines()
                 if ln.startswith("| ") and "---" not in ln]
    return rows


def table_drift() -> list[str]:
    """Does the draft's copy of Table 1 match the generated one?"""
    rows = _table_rows()
    if not rows:
        return ["Table 1 has not been generated — run make_figures.py"]
    draft = _draft_text()
    out = [f"Table 1 row missing from the draft (regenerate the manuscript's copy): {r[:70]}"
           for r in rows if r not in draft]
    generated = set(rows)
    for ln in draft.splitlines():
        ln = ln.rstrip()
        if (ln.startswith("| ") and not ln.startswith("|:")
                and "---" not in ln and ln not in generated and ln.count("|") >= 5):
            out.append(f"draft has a Table-1-shaped row that the generator did not produce: {ln[:70]}")
    return out


def _half_up(x: float, nd: int) -> str | None:
    """x rounded to `nd` decimals with ROUND_HALF_UP (what a human rounding 0.2975 to 3dp by hand
    gets: "0.298"), as a string -- or None if x is not finite (nan/inf have no rounded form)."""
    if not math.isfinite(x):
        return None
    try:
        q = decimal.Decimal(str(x)).quantize(decimal.Decimal(1).scaleb(-nd),
                                              rounding=decimal.ROUND_HALF_UP)
    except (decimal.InvalidOperation, OverflowError):
        return None
    return f"{q:g}" if nd == 0 else f"{q}"


def renderings(v) -> set[str]:
    """Every defensible way a canonical value can appear in prose."""
    o: set[str] = set()
    if v is None:
        return o
    if isinstance(v, bool):
        return o
    if isinstance(v, str):
        return {v}
    if isinstance(v, dict):
        for elem in v.values():
            o |= renderings(elem)
        return o
    if isinstance(v, list):
        if len(v) == 2:                       # a range renders with either dash, and as endpoints
            o |= {f"{v[0]}-{v[1]}", f"{v[0]}–{v[1]}"}
        for elem in v:
            o |= renderings(elem)
        return o
    if isinstance(v, int):
        return {str(v), f"{v:,}"}
    o.add(f"{v:g}")
    for nd in ((0, 1, 2, 3) if abs(v) >= 10 else (1, 2, 3)):
        # int() AND round(): banker's rounding turns .5 down to even, so both forms are legitimate
        o |= {f"{int(v) if nd == 0 else round(v, nd):g}", f"{v:.{nd}f}"}
        h = _half_up(v, nd)
        if h is not None:
            o.add(h)
    if 1e-4 < abs(v) <= 1:                    # a share may be written as a percentage
        for nd in (0, 1, 2):
            o |= {f"{round(v * 100, nd):g}", f"{v * 100:.{nd}f}"}
            h = _half_up(v * 100, nd)
            if h is not None:
                o.add(h)
    if 0 < abs(v) < 1e-3:
        # A p-value is written "1.6 × 10⁻⁹", never 1.6112145732095232e-09. The exponent is not a
        # token the tokeniser produces, so match on the mantissa: it is the part a reader checks.
        mant = v / 10 ** math.floor(math.log10(abs(v)))
        for nd in (1, 2, 3):
            o |= {f"{round(mant, nd):g}", f"{mant:.{nd}f}"}
    return {x for x in o if x}


def percent_renderings(v) -> set[str]:
    """Every way a canonical value could legitimately appear in the prose as a PERCENT token,
    recursing into lists and dicts so a companion field is covered too."""
    if isinstance(v, list):
        o: set[str] = set()
        for elem in v:
            o |= percent_renderings(elem)
        return o
    if isinstance(v, dict):
        o: set[str] = set()
        for elem in v.values():
            o |= percent_renderings(elem)
        return o
    if v is None or isinstance(v, (bool, str, int)):
        return set()
    o: set[str] = set()
    if 1e-4 < abs(v) <= 1:
        for nd in (0, 1, 2):
            o |= {f"{round(v * 100, nd):g}", f"{v * 100:.{nd}f}"}
            h = _half_up(v * 100, nd)
            if h is not None:
                o.add(h)
    return {x for x in o if x}


THOUS = r"(?<![\w.])\d{1,3}(?:,\d{3})+(?![\w.]|,\d{3})"
THOUS_PP = r"(?<![\w.])(\d{1,3}(?:,\d{3})+)pp(?!\w)"
BARE = (r"(?<![\w.])([+-]?\d+(?:\.\d+)?)(?![\w%])(?!\.\d)"
        r"|(?<![\w.])([+-]?\d+(?:\.\d+)?)%"
        r"|(?<![\w.])([+-]?\d+(?:\.\d+)?)pp(?!\w)")


def tokenise(text: str) -> list[str]:
    toks = re.findall(THOUS, text)
    toks += re.findall(THOUS_PP, text)  # THOUS_PP must be matched before falling back to BARE,
                                         # otherwise a comma-grouped pp figure like "3,314pp" would
                                         # tokenise into separate "3" and "314pp" fragments
    without_thous = re.sub(THOUS_PP, " ", re.sub(THOUS, " ", text))
    for a, b in re.findall(r"(?<![\w.])(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)(?![\w.])", without_thous):
        toks += [a, b]
    toks += [m[0] or m[1] or m[2] for m in re.findall(BARE, without_thous)]
    return toks


def tokenise_percent(text: str) -> set[str]:
    """Every number written with a trailing '%' in `text`, thousands separators removed first, as
    bare numeric strings -- the percent counterpart to tokenise()."""
    return set(re.findall(r"(?<![\w.])([+-]?\d+(?:\.\d+)?)%", re.sub(THOUS, " ", text)))


def cite_location(cite: str) -> str:
    """'S4.1 body: monthly-resolution check' -> 'S4.1'. The descriptor after the colon is for a human."""
    head = cite.split(":", 1)[0].strip()
    return re.sub(r"\s+body$", "", head).strip()


def orphans(canon: dict, sections: dict[str, str] | None = None) -> tuple[dict, int, int]:
    """Prose numbers that no canonical value produces."""
    known: set[str] = set()
    known_percent: set[str] = set()

    def _add(renders, extra_v):
        all_r = renders(extra_v)
        pct_r = percent_renderings(extra_v)
        known.update(all_r - pct_r)
        known_percent.update(pct_r)

    for v in canon.values():
        _add(renderings, v["value"])
        if v.get("of") is not None:
            _add(renderings, v["of"])
        for extra in ("numerator", "denominator", "states", "n", "n_cells", "n_series", "p_value",
                      "aux_max", "n_qualifying", "sd", "band", "p05_min", "p05_max", "p05_sd_max",
                      "p25_min", "p25_max", "permutation_p", "permutation_draws",
                      "bonferroni_p_value", "bonferroni_permutation_p", "bonferroni_m",
                      "counts", "thresholds_pp", "distinct_counts", "distinct_pi", "alphas",
                      "band_count", "seed0", "n_seeds",
                      "null_sd", "n_perm", "first_permutation",
                      "n_state_months",
                      "expected",
                      "n_rep",
                      # companions the prose quotes directly
                      "rate_arch", "rate_ours", "n_panels", "n_levels", "levels_median",
                      "iqr_lo", "iqr_hi",
                      "adjacent", "ratio_to_model_episodes_lo", "ratio_to_model_episodes_hi",
                      "movement_from_levels_subset", "nine_leg", "model_episodes_revised", "model_episodes_initial"):
            if v.get(extra) is not None:
                _add(renderings, v[extra])

    def benign(tok: str):
        bare = tok.lstrip("+")
        for pat, why in BENIGN:
            if re.match(pat, bare):
                return why
        return None

    body = _strip_covid19(load_body())
    toks = set(tokenise(body))
    pct_toks = tokenise_percent(body)
    out, n_benign = {}, 0
    for tok in toks:
        # A leading PLUS must not block the match; a leading MINUS must, because the sign is part of
        # the claim. Stripping both made a prose -0.62 resolve against a canonical +0.62.
        bare = tok.lstrip("+")
        covered = tok in known or bare in known
        if not covered and (tok in pct_toks or bare in pct_toks):
            # This specific token genuinely appeared with a % sign somewhere in the prose, so a
            # percent-derived rendering is a defensible match for it too.
            covered = tok in known_percent or bare in known_percent
        if not covered and sections is not None and re.match(r"^\d\.\d$", tok) and f"S{tok}" in sections:
            covered = True
        if covered:
            continue
        if benign(tok):
            n_benign += 1
            continue
        m = re.search(rf"(?<![\w.]){re.escape(tok)}(?![\w])(?!\.\d)", body)
        out[tok] = re.sub(r"\s+", " ",
                          body[max(0, m.start() - 60):m.start() + len(tok) + 28]).strip() if m else ""
    return out, len(toks), n_benign


def misplaced(canon: dict, sections: dict[str, str]) -> tuple[list[str], list[str], int, int]:
    """Canonical values that are not where their `cites` field says they are."""
    bad, unknown_loc, checked, uncited = [], [], 0, 0
    for k, v in sorted(canon.items()):
        cites = v.get("cites") or []
        if not cites:
            uncited += 1
            continue
        want = renderings(v["value"])
        want_pct = percent_renderings(v["value"])
        want_bare = want - want_pct
        for cite in cites:
            loc = cite_location(cite)
            if loc not in sections:
                unknown_loc.append(f"{k}: cites {cite!r}, but {loc!r} is not a location in the draft")
                continue
            checked += 1
            text = sections[loc]
            if isinstance(v["value"], str):
                if v["value"] == "":
                    found = False
                    parts = []   # keeps the len(parts) > 1 guard below well-defined and False
                else:
                    candidates = [v["value"]]
                    sep = ";" if ";" in v["value"] else "|"
                    parts = [p.strip() for p in v["value"].split(sep) if p.strip()]
                    if len(parts) > 1:
                        candidates.append(", ".join(parts[:-1]) + " and " + parts[-1])
                        candidates.append(", ".join(parts[:-1]) + ", and " + parts[-1])   # serial comma
                    found = any(re.search(rf"(?<!\w){re.escape(c)}(?!\w)", text) for c in candidates)
                if not found and len(parts) > 1 and not all(re.fullmatch(r"[\d.\-]+", p) for p in parts):
                    def _flex(p: str) -> str:
                        return re.escape(p).replace("_", "[-_ ]")
                    found = all(re.search(rf"(?<!\w){_flex(p)}(?!\w)", text, re.IGNORECASE) for p in parts)
            elif (isinstance(v["value"], list) and len(v["value"]) == 2
                  and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v["value"])):
                stripped = _strip_code(text)
                toks_bare = {t.lstrip("+") for t in tokenise(stripped)}
                toks_pct = {t.lstrip("+") for t in tokenise_percent(stripped)}
                dash_forms = {f"{v['value'][0]}-{v['value'][1]}", f"{v['value'][0]}–{v['value'][1]}"}
                if dash_forms & toks_bare:
                    found = True
                else:
                    lo_pct, hi_pct = percent_renderings(v["value"][0]), percent_renderings(v["value"][1])
                    lo_bare = renderings(v["value"][0]) - lo_pct
                    hi_bare = renderings(v["value"][1]) - hi_pct
                    lo_found = bool(lo_bare & toks_bare) or bool(lo_pct & toks_pct)
                    hi_found = bool(hi_bare & toks_bare) or bool(hi_pct & toks_pct)
                    found = lo_found and hi_found
            else:
                stripped = _strip_code(text)
                bare_hit = bool(want_bare & {t.lstrip("+") for t in tokenise(stripped)})
                pct_hit = bool(want_pct & {t.lstrip("+") for t in tokenise_percent(stripped)})
                found = bare_hit or pct_hit
            if not found:
                bad.append(f"{k} = {v['value']!r} is not in {loc} — declared by {cite!r} "
                           f"(producer: {v['producer']})")
    return bad, unknown_loc, checked, uncited


N_UNDETECTABLE_CITED = 132           # of 362 cited numeric values; recount with --rehash-coverage
UNDETECTABLE_SET_HASH = "f2366db39a5beb417c31c382bc4d0251da124095eb5a040728faca3fc154b494"


def _perturb_cited(v):
    """The deterministic perturbation the coverage measurement applies: ints +1, floats x1.05,
    numeric lists element-wise; None for anything that is not numeric (skipped)."""
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v + 1
    if isinstance(v, float):
        return v * 1.05
    if isinstance(v, list) and v and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v):
        return [_perturb_cited(x) for x in v]
    return None


def coverage(canon: dict, sections: dict[str, str]) -> tuple[int, list[str]]:
    """(n_cited_numeric, undetectable_keys): for every cited canonical value with a numeric
    `value`, perturb it in memory, leave the manuscript untouched, and ask whether orphans() or
    misplaced() notices -- a NEW orphan token or a NEW misplaced entry counts as detection."""
    base_orph, _, _ = orphans(canon, sections)
    base_bad, base_unknown, _, _ = misplaced(canon, sections)
    base_orph_keys, base_mis = set(base_orph), set(base_bad + base_unknown)
    n = 0
    undetectable = []
    for k in sorted(canon):
        v = canon[k]
        if not v.get("cites"):
            continue
        pv = _perturb_cited(v.get("value"))
        if pv is None:
            continue
        n += 1
        mutated = dict(canon)
        mutated[k] = {**v, "value": pv}
        orph, _, _ = orphans(mutated, sections)
        bad, unknown, _, _ = misplaced(mutated, sections)
        new_orphan = bool(set(orph) - base_orph_keys)
        new_misplaced = bool(set(bad + unknown) - base_mis)
        if not (new_orphan or new_misplaced):
            undetectable.append(k)
    return n, undetectable


def main() -> int:
    recognized = {"--verbose", "--coverage", "--rehash-coverage"}
    unknown = [a for a in sys.argv[1:] if a not in recognized]
    if unknown:
        print(f"  UNKNOWN ARGUMENT(S): {unknown} -- refusing to silently ignore a possible typo. "
              f"Recognized: {', '.join(sorted(recognized))}.", file=sys.stderr)
        return 2
    if "--coverage" in sys.argv or "--rehash-coverage" in sys.argv:
        if not _draft_exists():
            print(f"  {DRAFT.relative_to(ROOT)} does not exist -- nothing to measure coverage against.")
            return 2
        n, undetectable = coverage(_canon(), index_draft())
        h = hashlib.sha256(",".join(sorted(undetectable)).encode()).hexdigest()
        if "--rehash-coverage" in sys.argv:
            print(f"N_UNDETECTABLE_CITED = {len(undetectable)}   # of {n} cited numeric values")
            print(f'UNDETECTABLE_SET_HASH = "{h}"')
            return 0
        print("=== DETECTION COVERAGE — cited numeric values whose silent change this gate cannot see ===")
        print(f"  cited numeric values perturbed: {n}   undetectable: {len(undetectable)} "
              f"({100 * len(undetectable) / max(n, 1):.0f}%)")
        for k in undetectable:
            print(f"    {k}")
        ok = (N_UNDETECTABLE_CITED == len(undetectable)) and (UNDETECTABLE_SET_HASH == h)
        if not ok:
            print(f"  RATCHET: N_UNDETECTABLE_CITED says {N_UNDETECTABLE_CITED} / UNDETECTABLE_SET_HASH "
                  f"'{UNDETECTABLE_SET_HASH}'; this run gives {len(undetectable)} / '{h}'. If prose or "
                  f"values changed legitimately, re-run with --rehash-coverage and paste the literals; "
                  f"if the undetectable set GREW, a new cited number is now invisible to this gate.")
        print(f"\n  {'PASS' if ok else 'FAIL'}  detection-coverage ratchet")
        return 0 if ok else 1
    if not _draft_exists():
        print("=== CLAIMS ===")
        print(f"  {DRAFT.relative_to(ROOT)} does not exist -- nothing to check the manuscript's "
              f"numbers against. Not a failure -- this script reports, it never fails the run.")
        return 0
    note_suffix = f" + {NOTE.relative_to(ROOT)}" if NOTE.exists() else ""
    print(f"  checking {DRAFT.relative_to(ROOT)}{note_suffix}")
    canon = _canon()
    sections = index_draft()

    orph, n_tok, n_benign = orphans(canon, sections)
    print("=== ORPHANS — prose numbers with no canonical value ===")
    print(f"  canonical values: {len(canon)}   distinct tokens: {n_tok}   benign-by-rule: {n_benign}")
    print(f"  UNRESOLVED: {len(orph)}\n")
    for tok in sorted(orph, key=lambda x: float(x.lstrip('+').replace(',', ''))):
        print(f"    {tok:<9} …{orph[tok]}…")
    if not orph:
        print("    every numeric claim resolves to a canonical value or a stated benign rule.")

    drift = table_drift()
    print("\n=== TABLE 1 — generated vs the draft's copy ===")
    print(f"  generated data rows: {len(_table_rows())}   DRIFT: {len(drift)}\n")
    for d in drift:
        print(f"    {d}")
    if not drift:
        print("    the draft's Table 1 is row-for-row the generated one.")

    bad, unknown, n_checked, uncited = misplaced(canon, sections)
    print("\n=== MISPLACED — canonical values not where their `cites` says ===")
    print(f"  locations indexed: {len(sections)}   cite assertions checked: {n_checked}   "
          f"values with no cites: {uncited}")
    print(f"  MISPLACED: {len(bad)}   unknown location: {len(unknown)}\n")
    for b in bad + unknown:
        print(f"    {b}")
    if not bad and not unknown:
        print("    every canonical value appears at every location it declares.")

    misplaced_ratchet_ok = (len(bad) + len(unknown)) == N_MISPLACED
    if not misplaced_ratchet_ok:
        print(f"  RATCHET: expected exactly {N_MISPLACED} MISPLACED+unknown entries (all previously "
              f"reviewed and accepted), got {len(bad) + len(unknown)} -- if this is a legitimate new "
              f"accepted entry, review it against the 4 known limitation classes described in "
              f"N_MISPLACED's own comment above and bump N_MISPLACED in the same edit; otherwise "
              f"this is a real regression.")

    misplaced_set_hash = hashlib.sha256(",".join(sorted(bad + unknown)).encode()).hexdigest()
    misplaced_hash_ok = misplaced_set_hash == MISPLACED_SET_HASH
    if not misplaced_hash_ok:
        print(f"  RATCHET: MISPLACED_SET_HASH says '{MISPLACED_SET_HASH}', the actual "
              f"MISPLACED+unknown set hashes to '{misplaced_set_hash}' -- the COUNT may be "
              f"unchanged while WHICH entries are covered has changed; review before bumping.")

    if "--verbose" in sys.argv:
        print("\n  locations indexed: " + ", ".join(sorted(sections)))

    good = not orph and not drift and misplaced_ratchet_ok and misplaced_hash_ok
    print(f"\n  {'PASS' if good else 'FAIL'}  ORPHAN: {len(orph)}  DRIFT: {len(drift)}  "
          f"MISPLACED+unknown: {len(bad) + len(unknown)} (ratchet {'ok' if misplaced_ratchet_ok else 'FAILED'})")
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())

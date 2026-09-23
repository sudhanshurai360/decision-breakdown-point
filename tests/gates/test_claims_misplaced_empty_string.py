"""Regression test: claims.py's
`misplaced()` used to silently pass any canonical value whose `value` is the empty string, no
matter where (or whether) it was cited.

Mechanism: `re.escape("")` produces `""`, so the match pattern `(?<!\\w){re.escape(c)}(?!\\w)`
becomes the zero-width assertion `(?<!\\w)(?!\\w)`, which matches at virtually any non-word-
boundary position in real prose -- `found` was always True regardless of the section's content.

Not hypothetical: `_joined()` (code/gates/canonical.py, imported from data_config.py) returns ""
whenever its input iterable is empty -- e.g. `data.curated_dropped` (currently "TTLCONS") would
silently become "" the day a data refresh drops zero candidate series, and `misplaced()` would
then silently rubber-stamp that citation forever, no matter what the cited section actually says.

`tests/gates/test_claims_guard_chains_battery.py` includes "" in its EDGE_CASES list, but that
battery only exercises `renderings()`/`percent_renderings()`, never `misplaced()` itself -- where
this specific defect actually lives -- so the empty string looked "covered" while the real bug
was untested. This file exercises `misplaced()` directly.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "code" / "gates"))

import claims as C  # noqa: E402


def test_empty_string_value_is_reported_misplaced_not_silently_passed():
    """The core repro from the finding: an empty-string value, cited to a section that never
    mentions it at all, must be reported MISPLACED -- not silently pass because re.escape("")
    matches everywhere."""
    canon = {"fake.empty_str": {"value": "", "cites": ["S1 body"], "producer": "test"}}
    sections = {"S1": "This section says absolutely nothing about the fake key at all."}
    bad, unknown, checked, uncited = C.misplaced(canon, sections)
    assert bad == ["fake.empty_str = '' is not in S1 — declared by 'S1 body' (producer: test)"]
    assert unknown == []
    assert checked == 1
    assert uncited == 0


def test_empty_string_value_is_flagged_even_in_a_section_full_of_other_text():
    """Confirms the failure mode precisely: BEFORE the fix, this would also silently pass -- the
    zero-width regex matches regardless of how much unrelated text is in the section."""
    canon = {"fake.empty_str": {"value": "", "cites": ["S2 body"], "producer": "test"}}
    sections = {"S2": "A long section with lots of real numbers like 0.85 and 42 and dates like "
                      "1988, 2009 and 2010, but nothing resembling an empty-string claim."}
    bad, *_ = C.misplaced(canon, sections)
    assert len(bad) == 1


def test_non_empty_string_correctly_found_is_unaffected_by_the_fix():
    """Sanity/no-regression check: a normal non-empty string value that IS genuinely present in
    its cited section must still pass, confirming the fix didn't break the ordinary path."""
    canon = {"fake.ok": {"value": "unicorn", "cites": ["S1 body"], "producer": "test"}}
    sections = {"S1": "This section mentions unicorn explicitly."}
    bad, *_ = C.misplaced(canon, sections)
    assert bad == []


def test_semicolon_joined_non_empty_list_is_unaffected_by_the_fix():
    """Sanity/no-regression check: the semicolon/pipe-joined-list rendering logic (the code path
    immediately after the new empty-string guard) must still work for genuine multi-part values."""
    canon = {"fake.list": {"value": "1988;2009;2010", "cites": ["S1 body"], "producer": "test"}}
    sections = {"S1": "The relevant years were 1988, 2009 and 2010, each independently confirmed."}
    bad, *_ = C.misplaced(canon, sections)
    assert bad == []

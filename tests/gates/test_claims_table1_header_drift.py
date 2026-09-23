"""Regression test: claims.py's table_drift() used to be blind to Table 1's own header row in BOTH
directions -- `_table_rows()` excluded it from the generated-rows set via a `"Series" not in ln`
filter (Table P/Q/R's headers had no such exclusion), and table_drift()'s reverse-direction orphan
scan reused the SAME substring test to skip headers generically, so Table 1's header -- which also
legitimately contains "Series" -- was silently exempted from ever being checked. Live-demonstrated
before the fix: mutating the draft's copy of Table 1's header text (e.g. a column label change)
produced zero reported drift, defeating the exact safety net table_drift() exists to provide.

Fixed by including Table 1's header in the generated-rows set (matching Table P/Q/R's own
convention) and removing the now-redundant "Series" exclusion from the reverse-direction scan.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "code" / "gates"))

import claims as CL  # noqa: E402

pytestmark_reason = "paper/figures/table1_identification.md not available"


def _skip_if_missing():
    import pytest
    if not CL.TABLE.exists():
        pytest.skip(pytestmark_reason)


def test_table1_header_is_now_part_of_the_generated_rows_set():
    _skip_if_missing()
    rows = CL._table_rows()
    headers = [r for r in rows if r.startswith("| Series |") and "MCS" in r]
    assert headers, "Table 1's header row must now be included in _table_rows()'s output"


def test_a_mutated_draft_header_is_caught_as_drift(monkeypatch):
    """THE CORE REGRESSION: reproduces the exact live demonstration -- mutate the
    draft's copy of Table 1's header text and confirm table_drift() now reports it, instead of
    silently passing as it did before this fix."""
    _skip_if_missing()
    rows = CL._table_rows()
    real_header = next((r for r in rows if r.startswith("| Series |") and "MCS" in r), None)
    assert real_header is not None, "could not locate Table 1's real header in the generated rows"

    mutated_header = real_header.replace("Concentration", "Concentration-CHANGED")
    real_draft_text = CL._draft_text()
    assert real_header in real_draft_text, "sanity check: the real header must appear in the real draft"

    monkeypatch.setattr(CL, "_draft_text", lambda: real_draft_text.replace(real_header, mutated_header))
    drift = CL.table_drift()
    assert drift, "a mutated Table 1 header must be reported as drift, not silently pass"
    assert any("Concentration-CHANGED" in d or real_header[:70] in d for d in drift)


def test_a_genuinely_unmutated_draft_still_reports_zero_drift():
    """Sanity check the fix isn't over-eager: the real, current, unmutated draft_ssrn.md must still
    report zero Table 1 drift."""
    _skip_if_missing()
    drift = CL.table_drift()
    table1_drift = [d for d in drift if "Table 1" in d or "Table-1-shaped" in d]
    assert table1_drift == [], f"unexpected Table 1 drift on the real unmutated draft: {table1_drift}"

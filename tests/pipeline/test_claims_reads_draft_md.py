"""claims.py's real-manuscript reading.

claims.py is "the two-way map between the manuscript's numbers and the canonical object" (its own
module docstring), so it is worthless if it reads the wrong file: it once pointed at
paper/manuscript/sections/*.md, a directory structure archived when the project reverted to a
single paper/draft.md. main() then printed "no drafted sections exist... nothing to check" and
exited 0 on every run -- a gate reporting zero problems because it is looking at nothing,
indistinguishable in its own output from reporting zero because there genuinely are none.

The resolution: _draft_text() points at paper/draft.md directly (since renamed to
paper/draft_ssrn.md), and the render.py {{key}}-template rendering step is gone entirely -- the
draft's numbers are typed as literal prose, not template placeholders.

These tests verify: (1) claims.py now reads real prose and finds real signal (not "no drafted
sections"), on both the real manuscript (now paper/draft_ssrn.md) and a small synthetic fixture
where the expected ORPHAN/MISPLACED result is known in advance; (2) the stray-'{{' warning fires
if templating ever reappears.
"""
import json

import claims as C


def test_draft_exists_and_is_the_single_file_not_a_sections_directory():
    assert C.DRAFT.name == "draft_ssrn.md"
    assert C._draft_exists()


def test_draft_text_reads_real_prose_not_a_template():
    text = C._draft_text()
    assert len(text) > 1000
    assert "{{" not in text, "the manuscript should contain literal numbers, not {{key}} placeholders"


def test_main_reports_real_checking_not_no_drafted_sections(capsys, monkeypatch):
    """Regression lock for the exact bug: main() must not print the old 'no drafted sections
    exist' message when real prose exists."""
    # claims.py's main() now validates sys.argv against an allowlist --
    # without this, it would see pytest's own argv and reject it as unrecognized.
    monkeypatch.setattr("sys.argv", ["claims.py"])
    rc = C.main()
    out = capsys.readouterr().out
    assert rc == 0
    assert "no drafted sections exist" not in out
    assert "does not exist" not in out
    assert "checking paper/draft_ssrn.md" in out
    assert "=== ORPHANS" in out
    assert "=== MISPLACED" in out


def test_orphans_and_misplaced_find_real_signal_on_the_real_draft():
    """Locks in that the mechanism produces non-trivial output on the real manuscript: real
    numeric tokens indexed, real sections found, real cite assertions checked -- not that it
    trivially returns nothing because it is looking at an empty or malformed document."""
    canon = C._canon()
    orph, n_tok, n_benign = C.orphans(canon)
    assert n_tok > 50   # real prose has real numeric tokens, not a handful
    sections = C.index_draft()
    assert len(sections) > 5   # real numbered sections were actually indexed
    bad, unknown, n_checked, uncited = C.misplaced(canon, sections)
    assert n_checked > 20   # real cite assertions were actually checked against real section text


def test_synthetic_fixture_orphan_and_misplaced_directions(tmp_path, monkeypatch):
    """Small, fully-controlled fixture -- independent of the real manuscript's current (changing)
    content -- verifying the ORPHAN and MISPLACED directions each catch the specific defect they
    are supposed to."""
    draft = tmp_path / "draft.md"
    draft.write_text(
        "## 1. Introduction\n\n"
        "We find k*=2 in this panel, and a spurious number 999 that no canonical value produces.\n\n"
        "## 2. Results\n\n"
        "The concentration is not mentioned here at all.\n"
    )
    canon_path = tmp_path / "canonical.json"
    canon_path.write_text(json.dumps({
        "test.k_star": {"value": 2, "cites": ["S1 body: k*"], "producer": "test"},
        "test.concentration": {"value": 4.2, "cites": ["S1 body: concentration"], "producer": "test"},
    }))

    monkeypatch.setattr(C, "DRAFT", draft)
    monkeypatch.setattr(C, "CANON", canon_path)
    monkeypatch.setattr(C, "_DRAFT_TEXT_CACHE", None)
    monkeypatch.setattr(C, "_CANON_CACHE", None)
    # Without this, index_draft()/load_body() would ALSO read the real paper/note_ssrn.md (via the
    # real C.NOTE path, untouched by the DRAFT patch above), breaking this fixture's whole point --
    # independence from any real manuscript content. Point NOTE at a nonexistent tmp path so
    # _note_text() takes its empty-string branch (claims.py:62, `if NOTE.exists()`).
    monkeypatch.setattr(C, "NOTE", tmp_path / "note.md")
    monkeypatch.setattr(C, "_NOTE_TEXT_CACHE", None)

    canon = C._canon()
    orph, _n_tok, _n_benign = C.orphans(canon)
    assert "999" in orph, "the ORPHAN direction must catch a prose number with no canonical source"

    sections = C.index_draft()
    bad, unknown, n_checked, uncited = C.misplaced(canon, sections)
    assert any("test.concentration" in b for b in bad), (
        "the MISPLACED direction must catch a canonical value that is NOT where its own cites field "
        "says it is (test.concentration cites S1, but its value 4.2 appears nowhere in the draft)"
    )

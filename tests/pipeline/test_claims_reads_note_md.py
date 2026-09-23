"""claims.py's note support.

paper/note_ssrn.md carries the Open LLM Leaderboard result. It needs the identical ORPHAN (a prose
number with no canonical source) and MISPLACED (a canonical value not where its own `cites` field
says) discipline as the paper -- not a permanent unchecked exception, which is exactly the "gate
reporting zero problems because it is looking at nothing" failure mode claims.py exists to prevent.

These tests use a fully synthetic draft + note pair, independent of the real manuscript's
(changing) content -- same isolation discipline as test_claims_reads_draft_md.py's own synthetic
fixture test.
"""
import json

import claims as C


def _patch(monkeypatch, tmp_path, draft_text, note_text, canon):
    draft = tmp_path / "draft.md"
    draft.write_text(draft_text)
    note = tmp_path / "note.md"
    note.write_text(note_text)
    canon_path = tmp_path / "canonical.json"
    canon_path.write_text(json.dumps(canon))

    monkeypatch.setattr(C, "DRAFT", draft)
    monkeypatch.setattr(C, "NOTE", note)
    monkeypatch.setattr(C, "CANON", canon_path)
    monkeypatch.setattr(C, "_DRAFT_TEXT_CACHE", None)
    monkeypatch.setattr(C, "_NOTE_TEXT_CACHE", None)
    monkeypatch.setattr(C, "_CANON_CACHE", None)


DRAFT_TEXT = (
    "## 1. Introduction\n\n"
    "We find k*=2 in this panel.\n\n"
    "## 2. Results\n\n"
    "The concentration is 4.2.\n"
)
NOTE_TEXT = (
    "## 1. The instrument\n\n"
    "This note applies the same instrument elsewhere.\n\n"
    "## 2. The finding\n\n"
    "On this leaderboard, k*=1 of 6 tasks, and a spurious number 777 that no canonical value produces.\n"
)
CANON = {
    "test.k_star": {"value": 2, "cites": ["S1 body: k*"], "producer": "test"},
    "test.concentration": {"value": 4.2, "cites": ["S2 body: concentration"], "producer": "test"},
    "test.llm_kstar": {"value": 1, "cites": ["N2 body: the finding"], "producer": "test"},
}


def test_index_draft_merges_note_sections_under_an_n_prefix(monkeypatch, tmp_path):
    _patch(monkeypatch, tmp_path, DRAFT_TEXT, NOTE_TEXT, CANON)
    sections = C.index_draft()
    assert "S1" in sections and "S2" in sections, "draft.md sections must still index as before"
    assert "N1" in sections and "N2" in sections, "note.md sections must index under the N prefix"
    assert "leaderboard" in sections["N2"]
    assert "leaderboard" not in sections.get("S1", "") and "leaderboard" not in sections.get("S2", "")


def test_note_md_orphan_and_misplaced(monkeypatch, tmp_path):
    _patch(monkeypatch, tmp_path, DRAFT_TEXT, NOTE_TEXT, CANON)
    canon = C._canon()
    sections = C.index_draft()

    orph, n_tok, n_benign = C.orphans(canon, sections)
    assert "777" in orph, "a prose number in note.md with no canonical source must be an ORPHAN"

    bad, unknown, n_checked, uncited = C.misplaced(canon, sections)
    assert not any("test.llm_kstar" in b for b in bad), (
        "test.llm_kstar cites N2, and its value (1) genuinely appears in note.md's N2 span -- must "
        "NOT be reported MISPLACED"
    )


def test_note_md_misplaced_direction_catches_a_value_not_at_its_cited_note_location(monkeypatch, tmp_path):
    bad_canon = dict(CANON)
    bad_canon["test.llm_kstar"] = {"value": 999, "cites": ["N2 body: the finding"], "producer": "test"}
    _patch(monkeypatch, tmp_path, DRAFT_TEXT, NOTE_TEXT, bad_canon)
    canon = C._canon()
    sections = C.index_draft()
    bad, unknown, n_checked, uncited = C.misplaced(canon, sections)
    assert any("test.llm_kstar" in b for b in bad), (
        "test.llm_kstar cites N2 but its value (999) does not appear there -- must be MISPLACED"
    )


def test_no_note_md_behaves_exactly_as_before(monkeypatch, tmp_path):
    """If note.md does not exist, index_draft()/load_body() must behave exactly as they did before
    this feature existed -- no N-prefixed keys, no crash, no change to draft-only checking."""
    draft = tmp_path / "draft.md"
    draft.write_text(DRAFT_TEXT)
    canon_path = tmp_path / "canonical.json"
    canon_path.write_text(json.dumps(CANON))
    monkeypatch.setattr(C, "DRAFT", draft)
    monkeypatch.setattr(C, "NOTE", tmp_path / "note.md")  # does not exist
    monkeypatch.setattr(C, "CANON", canon_path)
    monkeypatch.setattr(C, "_DRAFT_TEXT_CACHE", None)
    monkeypatch.setattr(C, "_NOTE_TEXT_CACHE", None)
    monkeypatch.setattr(C, "_CANON_CACHE", None)

    assert C._note_text() == ""
    sections = C.index_draft()
    assert "S1" in sections and "S2" in sections
    assert not any(k.startswith("N") for k in sections), "no note.md means no N-prefixed locations"

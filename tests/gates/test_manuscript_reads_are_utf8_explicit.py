"""Regression tests: every read of the manuscript files (draft_ssrn.md/note_ssrn.md and the
generated Table 1/P/Q/R markdown fragments) in claims.py used to call `.read_text()` with no
explicit `encoding=`, which falls back to `locale.getpreferredencoding()` -- cp1252 by default on
Windows (Python's UTF-8-by-default mode, PEP 686, only ships in 3.15; REPRODUCE.md claims "any
Python 3.10+ environment... works", a range where this bug was live).

The manuscript is genuinely non-ASCII (em-dashes, Greek letters, math operators). Empirically
confirmed: decoding the real file as cp1252 raises `UnicodeDecodeError: 'charmap' codec can't
decode byte` -- a hard crash of claims.py, the core
reproducibility-verification gate, before any check logic runs.

Fixed by adding an explicit encoding="utf-8" to every read_text()/write_text() call touching a
manuscript file in claims.py. These tests lock that in two ways: (1) a source-level scan confirming
no bare, unencoded read/write of a manuscript-adjacent path remains in the gate file; (2) a direct
behavioral demonstration, using a small synthetic fixture containing the same class of non-ASCII
content, that reading it as cp1252 (what an unpatched bare read_text() would do on Windows) raises,
while the real function (which now hardcodes utf-8) succeeds regardless of the process's locale.
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "code" / "gates"))

import claims as CL  # noqa: E402

GATE_SRC = {
    "claims.py": (ROOT / "code" / "gates" / "claims.py").read_text(encoding="utf-8"),
}


def test_no_bare_unencoded_read_text_or_write_text_remains_in_the_gate_file():
    """Static regression: every `.read_text(`/`.write_text(` call in claims.py
    must carry an explicit `encoding=` argument on the SAME call (not a nearby one) -- a future edit
    that adds a new read of draft_ssrn.md/note_ssrn.md/a generated table without copying the encoding= kwarg
    would silently reintroduce this exact bug class. Only scans actual executable lines -- comment
    lines and docstring bodies are skipped, since this file's own fix comments quote the pre-fix
    bare-call pattern verbatim as an example."""
    for name, src in GATE_SRC.items():
        in_docstring = False
        for line in src.splitlines():
            stripped = line.strip()
            if stripped.count('"""') % 2 == 1:
                in_docstring = not in_docstring
                continue
            if in_docstring or stripped.startswith("#"):
                continue
            for m in re.finditer(r"\.read_text\(([^)]*)\)|\.write_text\(([^)]*)\)", line):
                args = m.group(1) if m.group(1) is not None else m.group(2)
                assert "encoding=" in args, (
                    f"{name}: found a read_text()/write_text() call with no explicit encoding= "
                    f"(line={line!r})"
                )


def test_non_ascii_manuscript_content_is_invalid_under_the_windows_default_locale_encoding():
    """Confirms the vulnerability class is real, not hypothetical: the same kind of non-ASCII
    content actually present in draft_ssrn.md (em-dashes, Greek letters) fails to decode as cp1252,
    the default `locale.getpreferredencoding()` result on a stock Windows install."""
    sample = "Table 1 -- alpha=α, rho=ρ (— em-dash, ≤ le, × times)".encode("utf-8")
    try:
        sample.decode("cp1252")
        raised = False
    except UnicodeDecodeError:
        raised = True
    assert raised, (
        "expected this UTF-8-encoded, manuscript-realistic sample to be invalid under cp1252 -- "
        "if this now passes, the sample no longer represents the real risk and should be updated"
    )


def test_draft_text_reads_correctly_regardless_of_what_a_bare_read_text_would_have_done(tmp_path, monkeypatch):
    """Direct behavioral proof: point claims.py's manuscript reader at a fixture file containing the
    same non-ASCII content, and confirm it reads back correctly (the code now hardcodes utf-8,
    independent of locale) -- while a bare, unencoded read of that same file (what the pre-fix code
    path did) would have raised on a cp1252-default system."""
    fixture = tmp_path / "draft_fixture.md"
    content = "## Abstract\n\nTable 1 -- alpha=α, rho=ρ (— em-dash, ≤ le, × times)\n"
    fixture.write_text(content, encoding="utf-8")

    # What a bare `.read_text()` on Windows' default locale would have done to this exact file:
    raw_bytes = fixture.read_bytes()
    try:
        raw_bytes.decode("cp1252")
        would_have_crashed = False
    except UnicodeDecodeError:
        would_have_crashed = True
    assert would_have_crashed, "fixture no longer reproduces the real risk -- update its content"

    # What the fixed code actually does: hardcoded utf-8, independent of locale.
    monkeypatch.setattr(CL, "_DRAFT_TEXT_CACHE", None)
    monkeypatch.setattr(CL, "DRAFT", fixture)
    got = CL._draft_text()
    assert got == content


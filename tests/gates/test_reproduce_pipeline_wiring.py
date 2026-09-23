"""Regression test: data_config.py (16
canonical data.* values) was never added to reproduce.py's PIPELINE list, so it never regenerated
on a full "one command" reproduction despite its own inputs regenerating. Fixed by adding it.

Neither canonical.py --check nor acceptance.py would
catch data_config.py (or any other producer) being silently DROPPED from PIPELINE again in the
future -- both gates re-derive values straight from files on disk, never by checking whether the
producing stage is actually wired into a full reproduction. This is not hypothetical: this exact
mistake (a producer left out of PIPELINE) has already happened at least twice in this codebase.

This test encodes a general, low-false-positive version of the check: every code/pipeline/ or
code/instrument/ script that calls `emit(` (i.e. genuinely produces canonical.json values -- the
same signal a human would use to decide "does this need to be in PIPELINE") must appear somewhere
in PIPELINE, unless explicitly named in EXEMPT below with a stated reason. Adding a new producer
without wiring it in (or exempting it with a reason) now fails this test instead of only failing
silently, months later, when someone notices canonical.json is stale.
"""
import re
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
REPRODUCE = ROOT / "reproduce.py"

# Files that call emit( but are deliberately not (yet) wired into PIPELINE, with a real reason.
# Adding an entry here is a decision, not a shortcut -- it must name why, matching this project's
# own "a floor is satisfied by deleting a check" discipline applied to exemption lists elsewhere
# (acceptance.py's BLOCKED, differential_test.py's KNOWN_DIVERGENT).
EXEMPT = {
    "code/instrument/canonical_emit.py": "defines emit() itself (the library, not a producer) -- "
        "matched by the emit( grep as a false positive, not a real candidate stage",
}

pytestmark = pytest.mark.skipif(not REPRODUCE.exists(), reason="reproduce.py not available")


def _pipeline_script_paths():
    text = REPRODUCE.read_text()
    m = re.search(r"^PIPELINE\s*=\s*\[(.*?)\n\]", text, re.S | re.M)
    assert m, "could not locate PIPELINE = [...] in reproduce.py -- has it been renamed?"
    return set(re.findall(r'"(code/(?:pipeline|instrument)/[a-zA-Z0-9_]+\.py)"', m.group(1)))


def _emit_producing_scripts():
    """A literal substring check, not an AST/call-graph analysis -- sound against every current
    file (no false positive or false negative), but it is a
    SCOPED guarantee: a future producer that calls emit() indirectly (`from canonical_emit import
    emit as e; e(...)`, a wrapper helper, getattr-based dispatch) would not contain the literal
    text "emit(" and would silently escape this check. Also non-recursive (.glob, not
    .rglob) -- code/pipeline/ and code/instrument/ are flat today, but a future subdirectory would
    escape scanning too. Neither pattern exists in this repo."""
    out = set()
    for sub in ("pipeline", "instrument"):
        for f in (ROOT / "code" / sub).glob("*.py"):
            if "emit(" in f.read_text():
                out.add(f"code/{sub}/{f.name}")
    return out


def test_every_emit_producer_is_wired_or_explicitly_exempt():
    wired = _pipeline_script_paths()
    producers = _emit_producing_scripts()
    unwired = producers - wired - set(EXEMPT)
    assert not unwired, (
        f"these producer scripts call emit( but are neither in reproduce.py's PIPELINE nor in "
        f"this test's EXEMPT allowlist: {sorted(unwired)} -- either wire them in (mind dependency "
        f"ordering) or add a reasoned EXEMPT entry"
    )


def test_exempt_entries_still_call_emit():
    """Guards EXEMPT itself against drift: an entry naming a file that no longer calls emit(
    (or was deleted) should be removed, not silently kept around."""
    producers = _emit_producing_scripts()
    for path in EXEMPT:
        assert path in producers, (
            f"EXEMPT names {path!r}, but it no longer calls emit( (or doesn't exist) -- remove "
            f"this stale entry"
        )


def test_data_config_specifically_is_wired():
    """The exact regression described above -- kept as its own explicit assertion so a
    failure here names the specific file, not just 'some producer is unwired'."""
    assert "code/pipeline/data_config.py" in _pipeline_script_paths()

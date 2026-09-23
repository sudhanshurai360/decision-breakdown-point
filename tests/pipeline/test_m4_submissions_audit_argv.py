"""m4_submissions_audit.py::main()'s argv validation. `limit = int(...) if "--limit" in sys.argv
else 0`, where `limit=0` means "audit ALL submissions", makes a typo'd flag (e.g. "--limt 8", meant
as "--limit 8") dangerous: without a check it falls through to limit=0, silently escalating an
intended 8-submission proof run into a full, unattended audit of every submission on the archive.
The same defect class applies to reproduce.py's own --list/--fast.

The guard is an explicit argv allowlist check placed BEFORE any network call
(list_submissions()), so these tests can safely subprocess-invoke the real script without ever
hitting GitHub.
"""
import subprocess
import sys
import textwrap
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SCRIPT = ROOT / "code" / "pipeline" / "m4_submissions_audit.py"


def test_typo_flag_exits_loudly_instead_of_silently_auditing_everything():
    r = subprocess.run([sys.executable, str(SCRIPT), "--limt", "8"], capture_output=True, text=True,
                        cwd=str(ROOT), timeout=30)
    assert r.returncode == 2, f"expected exit 2 on a typo'd flag, got {r.returncode}: {r.stdout}{r.stderr}"
    assert "UNKNOWN ARGUMENT" in (r.stdout + r.stderr)
    # sanity: the run must die before list_submissions(), whose result main() announces with this
    # exact line (m4_submissions_audit.py:132) -- its absence proves no network call was made.
    assert "submissions on the repo" not in r.stdout


def test_completely_unrelated_flag_also_rejected():
    r = subprocess.run([sys.executable, str(SCRIPT), "--force"], capture_output=True, text=True,
                        cwd=str(ROOT), timeout=30)
    assert r.returncode == 2
    assert "UNKNOWN ARGUMENT" in (r.stdout + r.stderr)


def test_limit_with_no_value_exits_loudly_instead_of_crashing_ugly():
    """Regression test: an earlier argv
    validation consumed "--limit"'s value with `i += 2` but never checked a value actually
    follows -- "--limit" as the last/only arg passed validation cleanly and then crashed with an
    unhandled IndexError at the int() conversion, instead of this file's own clean rejection
    message."""
    r = subprocess.run([sys.executable, str(SCRIPT), "--limit"], capture_output=True, text=True,
                        cwd=str(ROOT), timeout=30)
    assert r.returncode == 2, f"expected exit 2, got {r.returncode}: {r.stdout}{r.stderr}"
    assert "UNKNOWN ARGUMENT" in (r.stdout + r.stderr)
    assert "IndexError" not in (r.stdout + r.stderr), "must not crash with a raw traceback"


def test_limit_with_a_non_numeric_value_exits_loudly_instead_of_crashing_ugly():
    r = subprocess.run([sys.executable, str(SCRIPT), "--limit", "abc"], capture_output=True,
                        text=True, cwd=str(ROOT), timeout=30)
    assert r.returncode == 2, f"expected exit 2, got {r.returncode}: {r.stdout}{r.stderr}"
    assert "UNKNOWN ARGUMENT" in (r.stdout + r.stderr)
    assert "ValueError" not in (r.stdout + r.stderr), "must not crash with a raw traceback"


def test_limit_with_a_negative_value_is_rejected_not_silently_mis_sliced():
    """Regression test: an earlier fix used `.lstrip("-").isdigit()`, which strips a leading minus sign BEFORE checking
    digit-ness -- "--limit -5" passed as "valid" (unknown=[]), reaching `limit = int(...) = -5`,
    and Python's negative-slice semantics (`subs[:-5]`) then silently audited "all but the last 5
    submissions" instead of erroring or auditing "the first N" as documented. A negative limit is
    never semantically valid here."""
    r = subprocess.run([sys.executable, str(SCRIPT), "--limit", "-5"], capture_output=True,
                        text=True, cwd=str(ROOT), timeout=30)
    assert r.returncode == 2, f"expected exit 2, got {r.returncode}: {r.stdout}{r.stderr}"
    assert "UNKNOWN ARGUMENT" in (r.stdout + r.stderr)


def _unknown(argv):
    """main()'s OWN argv-scanning loop, lifted verbatim out of m4_submissions_audit.py (from the
    `known = {...}` allowlist down to the `if unknown:` rejection) and run against a synthetic argv.
    It sits directly above main()'s first network call, so it cannot be reached by calling main();
    executing its real source is what keeps this check from drifting into a private re-implementation
    that would pass no matter what the script does."""
    lines = SCRIPT.read_text(encoding="utf-8").splitlines()
    starts = [i for i, ln in enumerate(lines) if ln.strip() == 'known = {"--inspect", "--limit"}']
    ends = [i for i, ln in enumerate(lines) if ln.strip() == "if unknown:"]
    assert len(starts) == 1 and len(ends) == 1 and starts[0] < ends[0], (
        f"m4_submissions_audit.py's argv block no longer has the expected shape "
        f"(starts={starts}, ends={ends})"
    )
    ns = {"sys": types.SimpleNamespace(argv=list(argv))}
    exec(textwrap.dedent("\n".join(lines[starts[0]:ends[0]])), ns)
    return ns["unknown"]


def test_argv_validation_logic_accepts_every_documented_usage():
    """Unit-level check of the validation logic itself (isolated from main()'s network calls) --
    every combination the script's own USAGE docstring documents must produce zero unknown args,
    and each malformed form must be reported exactly once. In particular --limit's value is checked
    for presence and numeric-ness, not merely consumed positionally, and the check is a plain
    `.isdigit()` rather than `.lstrip("-").isdigit()`, so a negative value is rejected."""
    assert _unknown(["x"]) == []
    assert _unknown(["x", "--inspect"]) == []
    assert _unknown(["x", "--limit", "8"]) == []
    assert _unknown(["x", "--limit", "8", "--inspect"]) == []
    assert _unknown(["x", "--limt", "8"]) == ["--limt", "8"]
    assert _unknown(["x", "--limit"]) == ["--limit"]
    assert _unknown(["x", "--limit", "abc"]) == ["--limit abc"]
    assert _unknown(["x", "--limit", "-5"]) == ["--limit -5"]

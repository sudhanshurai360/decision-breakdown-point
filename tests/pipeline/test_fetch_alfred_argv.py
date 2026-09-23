"""Regression test: fetch_alfred.py::main() used to compute
`wanted = [a for a in sys.argv[1:] if a in SERIES] or list(SERIES)` -- indistinguishable between
"no series args given" (legitimate, means "all series") and "series args given but NONE
recognized" (a typo, e.g. "UNRATEE" instead of "UNRATE"), both producing an empty list and
silently falling back to `list(SERIES)`. Combined with a correctly-spelled --force, a request to
re-fetch ONE series could silently re-fetch and overwrite EVERY series -- these parquets are named
in reproduce.py's own FROZEN_ATOM list specifically because FRED/ALFRED revises data over time.

Fixed by distinguishing the two cases explicitly, and refusing loudly (exit 2) rather than
broadening scope when a positional argument doesn't match any known series. This check happens
before the Fred API client is constructed or any network call is made, so these tests can safely
subprocess-invoke the real script.
"""
import io
import subprocess
import sys
import textwrap
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SCRIPT = ROOT / "code" / "pipeline" / "fetch_alfred.py"


def test_typo_series_name_exits_loudly_instead_of_silently_refetching_everything():
    r = subprocess.run([sys.executable, str(SCRIPT), "UNRATEE", "--force"], capture_output=True,
                        text=True, cwd=str(ROOT), timeout=30)
    assert r.returncode == 2, f"expected exit 2 on a typo'd series name, got {r.returncode}: {r.stdout}{r.stderr}"
    assert "UNKNOWN SERIES" in (r.stdout + r.stderr)
    assert "UNRATEE" in (r.stdout + r.stderr)


def test_argv_scope_resolution_logic_directly():
    """Unit-level check of the wanted-series resolution logic in isolation (no network, no Fred
    client construction). main() builds its Fred client a few lines below this block and cannot be
    called as a whole, so the block's OWN source is lifted out of fetch_alfred.py and executed
    against a synthetic argv -- the resolution rule under test is the shipped one, not a copy."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("fa", SCRIPT)
    FA = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(FA)

    lines = SCRIPT.read_text(encoding="utf-8").splitlines()
    starts = [i for i, ln in enumerate(lines)
              if ln.strip().startswith("positional = [a for a in sys.argv")]
    ends = [i for i, ln in enumerate(lines) if ln.strip() == "wanted = list(SERIES)"]
    assert len(starts) == 1 and len(ends) == 1 and starts[0] < ends[0], (
        "fetch_alfred.py's wanted-series resolution block no longer has the expected shape "
        f"(starts={starts}, ends={ends})"
    )
    block = textwrap.dedent("\n".join(lines[starts[0]:ends[0] + 1]))

    def _resolve(argv):
        ns = {
            "sys": types.SimpleNamespace(argv=["fetch_alfred.py", *argv], exit=sys.exit,
                                         stderr=io.StringIO()),
            "SERIES": FA.SERIES,
        }
        try:
            exec(block, ns)
        except SystemExit as e:
            assert e.code == 2, f"unexpected exit code {e.code} for argv {argv}"
            return None  # rejected loudly
        return ns["wanted"]

    assert _resolve([]) == list(FA.SERIES), "bare invocation must mean ALL series, unchanged"
    assert _resolve(["PAYEMS"]) == ["PAYEMS"], "a real series name must still select just that one"
    assert _resolve(["PAYEMS", "--force"]) == ["PAYEMS"], "--force must not be treated as a series name"
    assert _resolve(["UNRATEE"]) is None, "an unrecognized series name must be rejected, not fall back to ALL"

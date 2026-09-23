"""Regression test: claims.py was the last script in the
CLI-argv-silent-ignore family left open after fixes to fetch_alfred.py and
m4_submissions_audit.py, and further fixes to fetch_state_ur_vintages.py, breadth_fetch.py,
sahm_evans_realtime.py, and run_multiseries.py.
`"--verbose" in sys.argv` used to silently ignore a typo'd flag (e.g. "--verbse") and just run
without verbose output -- fails toward the safe default (unlike the HIGH cases, which could
silently escalate scope), but still gave no signal the flag was ignored. Fixed identically to every
sibling script: reject any unrecognized argument loudly (exit 2) instead of silently proceeding.
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SCRIPT = ROOT / "code" / "gates" / "claims.py"


def _run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True,
                           cwd=str(ROOT), timeout=60)


def test_rejects_a_typo_d_verbose_flag():
    r = _run("--verbse")
    assert r.returncode == 2, r.stdout + r.stderr
    assert "UNKNOWN ARGUMENT" in (r.stdout + r.stderr)


def test_still_accepts_no_flags():
    r = _run()
    assert r.returncode in (0, 1), r.stdout + r.stderr  # never 2 -- argv check must not fire
    assert "UNKNOWN ARGUMENT" not in (r.stdout + r.stderr)


def test_still_accepts_the_real_verbose_flag():
    r = _run("--verbose")
    assert r.returncode in (0, 1), r.stdout + r.stderr
    assert "UNKNOWN ARGUMENT" not in (r.stdout + r.stderr)
    assert "locations indexed:" in r.stdout

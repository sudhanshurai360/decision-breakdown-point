"""The CLI-argv silent-ignore defect class, across the four scripts that take bare `--flag`
arguments with no argparse behind them: fetch_state_ur_vintages.py, breadth_fetch.py,
sahm_evans_realtime.py, run_multiseries.py. (fetch_alfred.py and m4_submissions_audit.py have their
own tests in tests/pipeline/test_fetch_alfred_argv.py and
tests/pipeline/test_m4_submissions_audit_argv.py.) A typo'd flag must be rejected loudly (exit 2),
never silently ignored so the script falls through to its default -- which for these four is the
conservative default, but still not what the caller asked for. These tests subprocess-invoke each
real script with a typo'd flag, verifying the check runs before any network or file-write side
effect.
"""
import subprocess
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
PIPELINE = ROOT / "code" / "pipeline"


def _run(script, *args):
    return subprocess.run([sys.executable, str(PIPELINE / script), *args], capture_output=True,
                           text=True, cwd=str(ROOT), timeout=30)


def test_fetch_state_ur_vintages_rejects_unknown_flag():
    r = _run("fetch_state_ur_vintages.py", "--Force")
    assert r.returncode == 2, r.stdout + r.stderr
    assert "UNKNOWN ARGUMENT" in (r.stdout + r.stderr)


def test_fetch_state_ur_vintages_still_accepts_known_flags():
    """Positive control: --as-of= must still parse -- reach past the argv check to the OUT-exists
    guard, not exit 2. Deliberately NOT passing --force here: the real OUT file exists in this
    repo, so a bare/--as-of-only invocation returns 0 immediately (skip-without-overwriting) with
    no network access -- --force would instead trigger a real 51-state rate-limited ALFRED fetch,
    which this test must never do."""
    r = _run("fetch_state_ur_vintages.py", "--as-of=2020-01-01")
    assert r.returncode != 2, r.stdout + r.stderr


def test_breadth_fetch_rejects_unknown_flag():
    r = _run("breadth_fetch.py", "--emplyoment")
    assert r.returncode == 2, r.stdout + r.stderr
    assert "UNKNOWN ARGUMENT" in (r.stdout + r.stderr)


def test_sahm_evans_realtime_rejects_unknown_flag():
    r = _run("sahm_evans_realtime.py", "--frce")
    assert r.returncode == 2, r.stdout + r.stderr
    assert "UNKNOWN ARGUMENT" in (r.stdout + r.stderr)


def test_run_multiseries_rejects_a_dash_prefixed_series_name_typo():
    """THE CORE SCENARIO this fix targets: a user typo'ing a series name AS a flag (following this
    file's own --combine/--force-style sibling convention) must not silently fall back to
    processing ALL series."""
    r = _run("run_multiseries.py", "--UNRATE")
    assert r.returncode == 2, r.stdout + r.stderr
    assert "UNKNOWN FLAG" in (r.stdout + r.stderr)


def _production_unknown_flags(argv):
    """run_multiseries.main()'s OWN unknown-flag predicate, lifted verbatim out of the module and
    run against a synthetic argv. Not a copy of it: widen or narrow the accept-list in
    run_multiseries.py and this helper's answers change with it."""
    src = (PIPELINE / "run_multiseries.py").read_text(encoding="utf-8")
    hits = [ln.strip() for ln in src.splitlines() if ln.strip().startswith("unknown_flags = ")]
    assert len(hits) == 1, (
        f"expected exactly one `unknown_flags = ...` predicate in run_multiseries.py, found {hits}"
    )
    ns = {"sys": types.SimpleNamespace(argv=["run_multiseries.py", *argv])}
    exec(hits[0], ns)
    return ns["unknown_flags"]


def test_run_multiseries_known_flags_are_not_rejected_by_the_validation_logic():
    """Positive control, at the logic level rather than a full subprocess run: actually invoking
    --combine or a real series name would read/write this repo's real tracked result files (parts/
    CSVs, multiseries_fragility.csv), which a test must never do as a side effect. So the real
    predicate is executed in isolation instead."""
    for argv in (["--combine"], ["UNRATE"], ["UNRATE", "--combine"]):
        assert _production_unknown_flags(argv) == [], (
            f"{argv} should not trigger the unknown-flag rejection"
        )
    # and it must still flag the typo case the subprocess test above covers end to end
    assert _production_unknown_flags(["--UNRATE"]) == ["--UNRATE"]

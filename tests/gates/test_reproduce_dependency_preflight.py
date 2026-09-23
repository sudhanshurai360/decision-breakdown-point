"""Regression test for the systemic fix: `reproduce.py` used to propagate `sys.executable` to every
downstream stage with zero verification that it -- or the packages the pipeline actually needs --
is importable. A missing `statsmodels` doesn't crash any stage; several stages instead silently
degrade to naive-persistence forecasts with no print, which corrupted a real cited number in a
live incident (`fredmd.nonidentified` swung from 56/74 to 34/74, caught only by chance). This
test locks in the fix: `_preflight_check()` must exit loudly when a required package is missing, and
must pass silently under the project's real venv.
"""
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
REPRODUCE = ROOT / "reproduce.py"

pytestmark = pytest.mark.skipif(not REPRODUCE.exists(), reason="reproduce.py not available")


def test_preflight_passes_silently_under_the_real_venv():
    """Not a mock -- actually imports reproduce.py under this test's own interpreter (assumed to be
    the real project venv, since that's what runs pytest) and calls _preflight_check() for real."""
    sys.path.insert(0, str(ROOT))
    import reproduce as R
    R._preflight_check()  # must not raise/exit


def test_preflight_fails_loudly_when_a_required_package_is_missing():
    """Simulates the exact incident: run reproduce.py's own preflight check under an
    interpreter that genuinely lacks a required package (the system Python, which has no
    statsmodels/fredapi installed in this project's environment) and confirm it exits nonzero with
    a clear message, rather than silently letting the caller proceed."""
    system_python = "/usr/bin/python3"
    if not Path(system_python).exists():
        pytest.skip("system python3 not available at the expected path on this machine")
    r = subprocess.run(
        [system_python, "-c",
         f"import sys; sys.path.insert(0, {str(ROOT)!r}); import reproduce as R; R._preflight_check()"],
        capture_output=True, text=True,
    )
    assert r.returncode != 0, (
        f"_preflight_check() must exit nonzero under an interpreter missing required packages; "
        f"got returncode=0, stdout={r.stdout!r}"
    )
    assert "FATAL" in r.stdout or "FATAL" in r.stderr
    assert "missing required package" in (r.stdout + r.stderr)


def test_run_calls_preflight_check_before_dispatching_any_stage():
    """Source-structure lock: run() must call _preflight_check() before its PIPELINE dispatch loop,
    not just have the function defined and unused -- a future edit that removed the call would
    silently reopen the exact gap this fix closes."""
    import inspect
    import reproduce as R
    src = inspect.getsource(R.run)
    preflight_pos = src.index("_preflight_check()")
    # the loop is enumerated (per-stage log file names carry the stage number)
    loop_pos = src.index("for i, (label, script, out, tier) in enumerate(PIPELINE, 1):")
    assert preflight_pos < loop_pos, (
        "_preflight_check() must be called before the PIPELINE dispatch loop in run(), not after"
    )


def test_preflight_fails_loudly_when_a_frozen_cache_is_missing(monkeypatch, tmp_path):
    """_preflight_check() used to verify only that required PACKAGES were importable, never that the
    4 frozen fetched-data caches (fredmd_api.parquet, state_ur.parquet, unrate_vintages.parquet,
    open_llm.parquet -- FROZEN_ATOM entry 3) actually exist before any stage runs. Their real
    producers (fred_md_sweep.py, state_eb_decision.py, sahm_evans_realtime.py, ml_witness.py) each
    have a live "if cache missing: fetch + overwrite" path with only a bare print, no warning --
    so a deleted cache would previously have been silently replaced with live (possibly since-
    revised) data rather than caught up front. Simulates a missing-cache repo layout via a
    monkeypatched ROOT pointing at an empty tmp_path, without touching the real caches."""
    import reproduce as R

    monkeypatch.setattr(R, "ROOT", tmp_path)
    with pytest.raises(SystemExit) as exc_info:
        R._preflight_check()
    msg = str(exc_info.value)
    assert "FATAL" in msg
    for name in ("fredmd_api.parquet", "state_ur.parquet", "unrate_vintages.parquet", "open_llm.parquet"):
        assert name in msg, f"{name!r} not named in the FATAL message: {msg}"


def test_the_4_frozen_caches_named_in_the_preflight_check_exist_on_the_real_repo():
    """Sanity check on the fixture itself: the 4 paths _preflight_check() now verifies must be the
    real, current locations of these caches (not stale paths that would make the check meaningless
    by always passing regardless of whether the real files are present)."""
    for name in ("fredmd_api.parquet", "state_ur.parquet", "unrate_vintages.parquet", "open_llm.parquet"):
        matches = [p for p in [
            ROOT / "multiseries" / "data" / "fredmd" / "fredmd_api.parquet",
            ROOT / "multiseries" / "data" / "state_ur.parquet",
            ROOT / "multiseries" / "data" / "unrate_vintages.parquet",
            ROOT / "multiseries" / "data" / "open_llm.parquet",
        ] if p.name == name]
        assert matches and matches[0].exists(), f"{name!r}'s expected path does not exist on disk"


def test_unrecognized_cli_flag_exits_loudly_instead_of_silently_running_the_full_pipeline():
    """reproduce.py's `if __name__ == "__main__":` block used to accept any argv silently
    -- "--fast" in sys.argv is simply False on a typo (--Fast, --fsat), so a typo would silently run
    the FULL slow pipeline (30+ min) instead of erroring or running the intended fast subset, with
    no warning. Confirms a typo'd/unknown flag now exits nonzero with a clear message rather than
    falling through to run(). Uses --help (guaranteed never a valid flag) rather than actually
    invoking the pipeline."""
    r = subprocess.run([sys.executable, str(REPRODUCE), "--help"], capture_output=True, text=True,
                        cwd=str(ROOT), timeout=30)
    assert r.returncode == 2, f"expected exit 2 on an unrecognized flag, got {r.returncode}: {r.stdout}{r.stderr}"
    assert "unrecognized argument" in (r.stdout + r.stderr)


def test_known_flags_still_work_after_the_argv_validation_fix():
    """--list must still work after the argv-validation fix above -- confirms the allowlist wasn't
    accidentally tightened to reject a real flag."""
    r = subprocess.run([sys.executable, str(REPRODUCE), "--list"], capture_output=True, text=True,
                        cwd=str(ROOT), timeout=30)
    assert r.returncode == 0, f"--list must still exit 0: {r.stdout}{r.stderr}"
    assert "FROZEN INPUT ATOM" in r.stdout


def test_required_packages_list_matches_what_the_pipeline_actually_imports():
    """Sanity check on the fixture itself: every package named in REQUIRED_PACKAGES must be a
    genuine top-level import somewhere in code/pipeline/ or code/instrument/ (no stale/invented
    entries), and the known statsmodels-dependent files' package is covered."""
    import reproduce as R
    all_code = ""
    for sub in ("pipeline", "instrument"):
        for f in (ROOT / "code" / sub).glob("*.py"):
            all_code += f.read_text()
    for pkg in R.REQUIRED_PACKAGES:
        assert pkg in all_code, f"REQUIRED_PACKAGES names {pkg!r}, but it appears nowhere in code/pipeline or code/instrument"
    assert "statsmodels" in R.REQUIRED_PACKAGES, (
        "statsmodels is the package whose absence caused the real incident this fix "
        "closes -- it must be in REQUIRED_PACKAGES"
    )

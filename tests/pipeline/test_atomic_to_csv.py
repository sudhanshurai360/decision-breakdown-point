"""Regression test for run_multiseries.py::_atomic_to_csv(), the one shared temp-file + copymode +
replace() helper that every CSV write in this pipeline routes through -- 27 call sites across 19
files at the time of writing, including flagship_fragility.py, score_multiseries.py,
regime_robustness.py, fred_md_sweep.py and density_audit.py.

score_multiseries.py's own call site cannot be verified via a full producer rerun (it crashes
earlier, on an unrelated pre-existing UNEMPLOY calendar-gap issue in walkforward.py, before
reaching its write block), so this file is the persisted unit-level substitute for that one site as
well as the direct test of the shared helper.

`_atomic_to_csv()` takes an optional `**to_csv_kwargs` passthrough (default `index=False`) for
m4_submissions_audit.py's one load-bearing non-default kwarg: `index=True`, since its DataFrame's
index is the submission id every downstream reader consumes via `index_col=0`.
`test_kwargs_passthrough_overrides_default_index` below covers that passthrough.
"""
import os
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "code" / "pipeline"))

import run_multiseries as RM  # noqa: E402


def test_writes_correct_content(tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    out = tmp_path / "out.csv"
    RM._atomic_to_csv(df, out)
    assert out.exists()
    got = pd.read_csv(out)
    pd.testing.assert_frame_equal(got, df)


def test_no_leftover_tmp_file(tmp_path):
    df = pd.DataFrame({"a": [1]})
    out = tmp_path / "out.csv"
    RM._atomic_to_csv(df, out)
    assert not (tmp_path / "out.csv.tmp").exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out.csv"]


def test_overwrite_preserves_existing_mode(tmp_path):
    """Path.replace() does not carry over permission bits on its own -- _atomic_to_csv() must
    explicitly copy the destination's existing mode onto the temp file before the replace()."""
    df1 = pd.DataFrame({"a": [1]})
    out = tmp_path / "out.csv"
    RM._atomic_to_csv(df1, out)
    os.chmod(out, 0o640)
    assert (os.stat(out).st_mode & 0o777) == 0o640

    df2 = pd.DataFrame({"a": [1, 2, 3]})
    RM._atomic_to_csv(df2, out)
    assert (os.stat(out).st_mode & 0o777) == 0o640
    pd.testing.assert_frame_equal(pd.read_csv(out), df2)


def test_first_write_has_no_existing_destination_to_copy_mode_from(tmp_path):
    """out.exists() is False on the very first write -- the shutil.copymode() branch must not
    fire (and must not raise) when there is no destination file yet to copy a mode from."""
    df = pd.DataFrame({"a": [1]})
    out = tmp_path / "brand_new.csv"
    assert not out.exists()
    RM._atomic_to_csv(df, out)
    assert out.exists()
    pd.testing.assert_frame_equal(pd.read_csv(out), df)


def test_kwargs_passthrough_overrides_default_index(tmp_path):
    """m4_submissions_audit.py's call site needs `index=True` (its DataFrame's index is the
    submission id, load-bearing for every downstream `pd.read_csv(..., index_col=0)` reader) --
    an explicit kwarg must override the hardcoded `index=False` default every other call site
    relies on, without those other call sites needing to change."""
    df = pd.DataFrame({"a": [1, 2]}, index=pd.Index(["s1", "s2"], name="submission"))
    out = tmp_path / "out.csv"
    RM._atomic_to_csv(df, out, index=True)
    got = pd.read_csv(out, index_col=0)
    pd.testing.assert_frame_equal(got, df, check_names=False)

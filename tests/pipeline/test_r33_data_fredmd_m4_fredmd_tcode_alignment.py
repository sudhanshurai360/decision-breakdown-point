"""Mutation-verified tests for fred_md_sweep.py's transformation arm: `dates_t = s.index[-len(y_t):]` (main(), the line
that dates a McCracken-Ng-transformed series) had no test; the mutant `s.index[:len(y_t)]` shifts
every year boundary of the tcode arm by 1-2 months, flips 2 of 12 probed real series and would move
`fredmd.nonidentified_tcode` (31 of 63), while 82/82 existing tests pass.

The alignment answer is a hand fact, not a model output: a difference x_t - x_{t-1} is dated t, so
the transformed series takes the LAST len(y_t) dates of the level series (tcode 2/5/7 drop the
first month, tcode 3/6 drop the first two, tcode 1/4 drop none). main() is driven end-to-end on a
six-series synthetic panel with `series_losses` replaced by a recorder that returns a well-formed
loss block, so the assertion is on the exact (y, dates) pair main() hands to the tcode arm.
"""
import contextlib
import io

import numpy as np
import pandas as pd
import pytest

import fred_md_sweep as FS

# lengths for codes 1..7 on an n-point level series: McCracken & Ng (2016) Appendix definitions
_DROP = {1: 0, 2: 1, 3: 2, 4: 0, 5: 1, 6: 2, 7: 2}


def _synthetic_panel(seed=0):
    """Six positive, gap-free monthly series 1959-01..2025-12 named after real FRED-MD mnemonics
    so main() takes each branch: five carry a published code (5, 2, 4, 6, 1) and TCU carries none.
    All five transformed series stay far below TCODE_OUTLIER_RATIO_THRESHOLD (ratios 1-5)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("1959-01-01", "2025-12-01", freq="MS")
    n = len(dates)
    t = np.arange(n)

    def grow(scale, g):
        return scale * np.exp(g * t) * (1 + 0.03 * np.sin(2 * np.pi * t / 12)) * np.exp(rng.normal(0, 0.005, n))

    return pd.DataFrame({
        "PAYEMS": grow(50000, 0.0015),                                              # tcode 5
        "UNRATE": 5 + 1.5 * np.sin(2 * np.pi * t / 96) + rng.normal(0, 0.05, n),    # tcode 2
        "HOUST": grow(1500, 0.0002),                                                # tcode 4
        "M1SL": grow(200, 0.004),                                                   # tcode 6
        "AWHMAN": 40 + 0.5 * np.sin(2 * np.pi * t / 60) + rng.normal(0, 0.05, n),   # tcode 1
        "TCU": 80 + 2 * np.sin(2 * np.pi * t / 84) + rng.normal(0, 0.1, n),        # no code
    }, index=dates)


def _drive_main(monkeypatch, df):
    """Run FS.main() on `df` with every side effect stubbed; return the recorded series_losses calls."""
    calls = []

    def recorder(y, dates):
        y = np.asarray(y, float).copy()
        dates = pd.DatetimeIndex(dates)
        calls.append((y, dates))
        yr = dates.year
        yrs = [Y for Y in range(FS.EVAL_START, int(yr.max()) + 1)
               if (yr < Y).sum() >= 60 and (yr == Y).sum() >= 6]
        r = np.random.default_rng(len(calls))
        return 1 + r.random((len(yrs), 4)), 1 + r.random((len(yrs), 4)), list(FS.MODELS), yrs

    monkeypatch.setattr(FS, "series_losses", recorder)
    monkeypatch.setattr(FS, "load_fredmd", lambda: df)
    monkeypatch.setattr(FS, "emit", lambda *a, **k: None)
    monkeypatch.setattr(FS, "flush", lambda *a, **k: None)
    monkeypatch.setattr(FS.RM, "_atomic_to_csv", lambda *a, **k: None)
    with contextlib.redirect_stdout(io.StringIO()):
        FS.main()
    return calls


def _tcode_calls(calls, levels, code):
    """The recorded call(s) whose y equals apply_tcode(levels, code) -- the tcode arm for that
    series (for tcode 1 the transform is the identity, so the levels arm matches too: 2 hits)."""
    want = FS.apply_tcode(levels, code)
    hits = [(y, d) for (y, d) in calls if len(y) == len(want) and np.allclose(y, want)]
    assert len(hits) == (2 if code == 1 else 1), f"tcode {code}: expected one tcode-arm call, found {len(hits)}"
    return hits


@pytest.mark.parametrize("col", ["PAYEMS", "UNRATE", "HOUST", "M1SL", "AWHMAN"])
def test_tcode_arm_dates_are_the_last_len_yt_dates_of_the_level_series(monkeypatch, col):
    """kills: `dates_t = s.index[:len(y_t)]`. For PAYEMS (tcode 5) the transformed series must run
    1959-02-01..2025-12-01 (the first month has no predecessor to difference against); for M1SL
    (tcode 6) 1959-03-01..2025-12-01; for HOUST/AWHMAN (codes 4/1) the full 1959-01..2025-12."""
    df = _synthetic_panel()
    calls = _drive_main(monkeypatch, df)
    levels = df[col].to_numpy(float)
    code = FS.FREDMD_TCODE[col]
    drop = _DROP[code]
    for y_t, dates_t in _tcode_calls(calls, levels, code):
        assert len(y_t) == len(levels) - drop
        assert len(dates_t) == len(y_t)
        assert dates_t[0] == df.index[drop], (
            f"{col} (tcode {code}): transformed series must start at {df.index[drop].date()}, "
            f"got {dates_t[0].date()} -- the tcode arm is dated from the wrong end")
        assert dates_t[-1] == df.index[-1] == pd.Timestamp("2025-12-01"), (
            f"{col} (tcode {code}): transformed series must end at the level series' last month")
        pd.testing.assert_index_equal(dates_t, df.index[-len(y_t):], check_names=False)


def test_main_runs_both_arms_for_coded_series_and_one_arm_for_uncoded(monkeypatch):
    """Structural companion: 5 coded series x 2 arms + 1 uncoded series x 1 arm = 11 series_losses
    calls, so the alignment test above really did observe the tcode arm of every coded series."""
    df = _synthetic_panel()
    calls = _drive_main(monkeypatch, df)
    assert len(calls) == 11
    lengths = sorted(len(y) for y, _ in calls)
    n = len(df)
    assert lengths == sorted([n] * 6 + [n - 1, n - 1, n, n - 2, n])


# --- the McCracken-Ng formulas themselves, checked against INDEPENDENT closed forms on a random
# positive series (the existing test_fred_md_tcode.py locks 3-4 point golden values; this locks
# the output LENGTH convention the alignment above relies on and the formula on 400 points) -----

def test_tcode_output_lengths_follow_mccracken_ng():
    """kills: any tcode that drops the wrong number of leading observations (e.g. tcode 3 or 6 as a
    single difference, tcode 7 without its outer diff)."""
    y = np.exp(np.random.default_rng(1).normal(0, 0.1, 50).cumsum()) + 1.0
    for code, drop in _DROP.items():
        assert len(FS.apply_tcode(y, code)) == 50 - drop, f"tcode {code}"


def test_tcode_formulas_agree_with_independent_closed_forms_on_a_random_series():
    """kills: formula edits invisible to the 3-4 point golden tests (e.g. tcode 7 as the plain
    growth rate, tcode 6 as a first difference of logs, tcode 5 as log of a difference)."""
    rng = np.random.default_rng(2)
    y = 100.0 * np.exp(rng.normal(0.002, 0.02, 400).cumsum())
    g = y[1:] / y[:-1] - 1.0                                   # growth rate, dated t
    ly = np.log(y)
    expected = {
        1: y,
        2: y[1:] - y[:-1],
        3: (y[2:] - y[1:-1]) - (y[1:-1] - y[:-2]),
        4: ly,
        5: ly[1:] - ly[:-1],
        6: (ly[2:] - ly[1:-1]) - (ly[1:-1] - ly[:-2]),
        7: g[1:] - g[:-1],
    }
    for code, want in expected.items():
        got = FS.apply_tcode(y, code)
        assert got.shape == want.shape, f"tcode {code}: shape {got.shape} vs {want.shape}"
        np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-14, err_msg=f"tcode {code}")

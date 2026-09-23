"""Mutation-verified tests for m4_submissions_audit.py's FREQ table and load_data() scale.
FREQ = {prefix: (name, m, h)} sets the
seasonal period m of every MASE/RMSSE denominator on the 61 x 100,000 M4 re-scoring; the existing
score() test passes precomputed scales in, so mutating e.g. Monthly 12 -> 1 or Weekly 1 -> 52 moved
m4sub.gap_mase_pct / gap_rmsse_pct / k_star with the suite green. Two layers:

1. A synthetic six-frequency M4 layout with LINEAR training series, so the lag-m seasonal
   difference is exactly m: scale_abs must come back as {Y:1, Q:4, M:12, W:1, D:1, H:24} and
   scale_sq as their squares. Hand answer, no real data, < 1 s.
2. GOLDEN PIN on the frozen local M4 inputs (skipped with a reason if absent): submission-118's
   re-scored sMAPE/MASE through the real load_data + parse_submission + score path must equal the
   M4 organisers' published results for the winner (Makridakis, Spiliotis & Assimakopoulos 2020,
   Table 3; Smyl 2020): total 11.374 / 1.536 and per frequency Y 13.176/2.980, Q 9.679/1.118,
   M 12.126/0.884, W 7.817/2.356, D 3.170/3.446, H 9.328/0.893. Protects m4sub.*, and makes S3.5's
   "The re-scored sMAPE and MASE of the winning submission match the organizers' published values,
   a check made by hand rather than enforced by the pipeline" a check the test suite does enforce.
   ~20 s once per module.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import m4_submissions_audit as MSA

ROOT = Path(__file__).resolve().parent.parent.parent
M4_DATA = ROOT / "multiseries" / "data" / "m4_sub"

OFFICIAL_FREQ = {"Y": ("Yearly", 1, 6), "Q": ("Quarterly", 4, 8), "M": ("Monthly", 12, 18),
                 "W": ("Weekly", 1, 13), "D": ("Daily", 1, 14), "H": ("Hourly", 24, 48)}


def test_freq_table_is_the_m4_organisers_seasonal_periods_and_horizons():
    """kills: any edit to FREQ's (m, h). The M4 competition's own definitions (Makridakis et al.
    2020, Table 1): Yearly m=1 h=6, Quarterly m=4 h=8, Monthly m=12 h=18, Weekly m=1 h=13,
    Daily m=1 h=14, Hourly m=24 h=48."""
    assert MSA.FREQ == OFFICIAL_FREQ


# --- layer 1: synthetic M4 layout, hand-computed scales -----------------------------------------

def _write_synthetic_m4(tmp_path):
    """One linear training series per frequency (values 1..n, so every lag-m difference is m),
    one test series of its horizon length, in the organisers' wide right-padded CSV layout."""
    n_train = {"Y": 30, "Q": 40, "M": 60, "W": 80, "D": 50, "H": 100}
    for pfx, (name, m, h) in OFFICIAL_FREQ.items():
        n = n_train[pfx]
        tr = pd.DataFrame([[f"{pfx}1"] + list(np.arange(1, n + 1, dtype=float))],
                          columns=["V1"] + [f"V{i}" for i in range(2, n + 2)])
        te = pd.DataFrame([[f"{pfx}1"] + list(np.arange(n + 1, n + h + 1, dtype=float))],
                          columns=["V1"] + [f"V{i}" for i in range(2, h + 2)])
        tr.to_csv(tmp_path / f"{name}-train.csv", index=False)
        te.to_csv(tmp_path / f"{name}-test.csv", index=False)


def test_load_data_scales_are_the_lag_m_differences_of_each_frequency(monkeypatch, tmp_path):
    """kills: Monthly m 12 -> 1 (scale 12 -> 1), Weekly m 1 -> 52 (scale 1 -> 52), Hourly 24 -> 1,
    or any other m edit in FREQ, and any change to load_data's lag-m seasonal-difference scale."""
    _write_synthetic_m4(tmp_path)
    monkeypatch.setattr(MSA, "DATA", tmp_path)
    monkeypatch.setattr(MSA, "_get", lambda url, dest: dest)        # never touch the network
    truth = MSA.load_data()
    assert set(truth) == {f"{p}1" for p in OFFICIAL_FREQ}
    for pfx, (name, m, h) in OFFICIAL_FREQ.items():
        yt, m_got, sa, ssq = truth[f"{pfx}1"]
        assert m_got == m, (pfx, m_got)
        assert len(yt) == h, (pfx, len(yt))
        assert sa == pytest.approx(float(m), abs=1e-12), f"{name}: scale_abs {sa} != m={m}"
        assert ssq == pytest.approx(float(m * m), abs=1e-9), f"{name}: scale_sq {ssq} != m^2={m*m}"


# --- layer 2: golden pin on the frozen local M4 inputs -------------------------------------------

OFFICIAL_118 = {  # (sMAPE x100, MASE) as published by the organisers for the winning entry
    "total": (11.374, 1.536),
    "Y": (13.176, 2.980), "Q": (9.679, 1.118), "M": (12.126, 0.884),
    "W": (7.817, 2.356), "D": (3.170, 3.446), "H": (9.328, 0.893),
}


@pytest.fixture(scope="module")
def sub118_scores():
    """Per-series (sMAPE, MASE) of submission-118 on all 100,000 series through the real code path."""
    sub_dir = M4_DATA / "submission-118"
    needed = [M4_DATA / f"{name}-{split}.csv" for (name, _, _) in OFFICIAL_FREQ.values()
              for split in ("train", "test")]
    if not sub_dir.exists() or not all(p.exists() for p in needed):
        pytest.skip("local M4 inputs absent (multiseries/data/m4_sub/{*-train,*-test}.csv, submission-118/)")
    truth = MSA.load_data()                                            # reads the local CSVs only
    if len(truth) != 100_000:
        pytest.skip(f"local M4 test set has {len(truth)} series, not the full 100,000")
    fc = MSA.parse_submission(list(sub_dir.rglob("*.csv")) + list(sub_dir.rglob("*.txt")))
    rows = {}
    for sid, yp in fc.items():
        if sid in truth:
            yt, m, sa, ssq = truth[sid]
            sc = MSA.score(yp, yt, sa, ssq)
            if sc:
                rows[sid] = (sc["sMAPE"], sc["MASE"])
    df = pd.DataFrame.from_dict(rows, orient="index", columns=["sMAPE", "MASE"])
    df["freq"] = [s[0] for s in df.index]
    return df


def test_submission_118_total_reproduces_the_published_m4_winner_scores(sub118_scores):
    """GOLDEN PIN: 11.374 / 1.536 (organisers' published totals for the winner). kills: any FREQ
    m/h edit, any scale or score formula drift, a changed submission set."""
    df = sub118_scores
    assert len(df) == 100_000
    assert round(100 * df["sMAPE"].mean(), 3) == OFFICIAL_118["total"][0]
    assert round(df["MASE"].mean(), 3) == OFFICIAL_118["total"][1]


@pytest.mark.parametrize("pfx", ["Y", "Q", "M", "W", "D", "H"])
def test_submission_118_per_frequency_reproduces_the_published_scores(sub118_scores, pfx):
    """GOLDEN PIN per frequency: isolates which FREQ entry (m) a regression touched."""
    g = sub118_scores[sub118_scores["freq"] == pfx]
    assert len(g) > 0
    smape, mase = OFFICIAL_118[pfx]
    assert round(100 * g["sMAPE"].mean(), 3) == smape, (pfx, 100 * g["sMAPE"].mean())
    assert round(g["MASE"].mean(), 3) == mase, (pfx, g["MASE"].mean())

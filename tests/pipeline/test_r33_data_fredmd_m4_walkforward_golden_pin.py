"""GOLDEN PIN: the frozen loss matrix
`multiseries/results/breadth_distinct/breadth_distinct_predictions.csv` must be what the CURRENT
walkforward.py + models_freq.py produce from the CURRENT ALFRED vintage parquets.

The paper asserts this in prose: the file is "kept fixed so that every analysis pass built on top
of it scores the same evaluation lineage" (S3.6). Verified by hand it is (all 12 series x
{seasonal_naive, ar1}, plus ets/theta on RSAFS and BOPGEXP, bit-identical), but a hand check leaves
nothing in the suite machine-checking it -- which is why every
number-changing mutation of walkforward.py / models_freq.py survived: the suite verifies only the
DOWNSTREAM of the frozen file. These pins re-run `WF.run(sid, MF.build(models))` on one or two
real series and assert equality with the frozen rows, so a regression in either module now fails.

Canonical keys protected (all derived from the frozen matrix): modelset.*, data.curated_*,
every Table-1 per-series fragility cell, mcs_test / realised_t / regime_robustness inputs.
Runtime: ~1 s (naive/ar1), ~10 s (ets/theta), ~11 s (dhr); sarima is deliberately not pinned
(minutes per series). Skipped with a reason if the inputs are absent.
kills: every walkforward.py / models_freq.py mutant that touches the
published rows (M1, M2, M6, M7, N1, N2, N5, N7, N8).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "code" / "pipeline"))

import walkforward as WF  # noqa: E402
import models_freq as MF  # noqa: E402

FROZEN = ROOT / "multiseries" / "results" / "breadth_distinct" / "breadth_distinct_predictions.csv"
RAW = ROOT / "multiseries" / "data" / "raw"
COLS = ["y_true", "y_pred", "scale_abs", "scale_sq", "sae", "msse"]


def _require(sid):
    if not FROZEN.exists():
        pytest.skip(f"frozen matrix absent: {FROZEN}")
    if not (RAW / f"{sid}_alfred_vintages.parquet").exists():
        pytest.skip(f"vintage parquet absent for {sid}")


def _frozen(sid, models):
    f = pd.read_csv(FROZEN)
    f["target"] = pd.to_datetime(f["target"])
    return f[(f["series"] == sid) & (f["model"].isin(models))]


def _assert_regenerates(sid, models):
    _require(sid)
    frozen = _frozen(sid, models)
    assert len(frozen) > 0, f"no frozen rows for {sid} x {models}"
    new = WF.run(sid, MF.build(models, m=12), eval_start=1990, eval_end=2025, m=12)
    mrg = frozen.merge(new, on=["series", "model", "target"], how="outer",
                       suffixes=("_f", "_n"), indicator=True)
    missing = mrg[mrg["_merge"] != "both"]
    assert missing.empty, (
        f"{sid} x {models}: row set differs from the frozen matrix "
        f"({(mrg['_merge'] == 'left_only').sum()} only-frozen, "
        f"{(mrg['_merge'] == 'right_only').sum()} only-regenerated)")
    for c in COLS:
        a, b = mrg[f"{c}_f"].to_numpy(float), mrg[f"{c}_n"].to_numpy(float)
        bad = ~np.isclose(a, b, rtol=1e-9, atol=0.0)
        assert not bad.any(), (
            f"{sid} x {models}: column {c} differs from the frozen matrix on {bad.sum()} of "
            f"{len(a)} rows; first: target={mrg['target'][bad].iloc[0].date()} "
            f"model={mrg['model'][bad].iloc[0]} frozen={a[bad][0]!r} regenerated={b[bad][0]!r}")


@pytest.mark.parametrize("sid", ["RSAFS", "BOPGEXP"])
def test_frozen_matrix_regenerates_seasonal_naive_and_ar1(sid):
    """kills: M1, M2, M6, M7 (walkforward.py) -- the scale, COVID mask and sae/msse formulas all
    enter every frozen row. ~0.5 s per series."""
    _assert_regenerates(sid, ["seasonal_naive", "ar1"])


def test_frozen_matrix_regenerates_ets_and_theta_on_rsafs():
    """kills: N1, N7 (ETS spec), N2, N8 (Theta spec). ~10 s."""
    _assert_regenerates("RSAFS", ["ets", "theta"])


def test_frozen_matrix_regenerates_dhr_on_rsafs():
    """kills: N5 (DHR K=1). ~11 s."""
    _assert_regenerates("RSAFS", ["dhr"])

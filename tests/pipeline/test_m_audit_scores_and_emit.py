"""m_audit.py carried two of the codebase's most consequential fixes
(the Monthly 3,000->48,000 sample-size correction, and the entire new --emit mode producing
maudit.flip_mase_rmsse_{weekly,monthly}/maudit.any_flip_{weekly,monthly}) but had zero unit-test
coverage -- nothing hand-verified scores()'s MASE/RMSSE formulas, the winner-argmin logic, or that
--emit actually emits correct weekly/monthly pairs without one clobbering the other via
canonical_emit.flush()'s known "overwrites the whole fragment" behavior.

CAUTION: any test that calls run_one()/main() here MUST redirect DATA/RESULTS to a tmp_path and
never touch the real multiseries/data/m_audit or multiseries/results paths -- a sibling test file
(test_mcs_size_level.py) initially forgot this and silently overwrote
committed result CSVs with garbage low-replication values, caught only by canonical.py --check
failing afterward. Every test below wires DATA/RESULTS/RES to tmp_path explicitly.
"""
import numpy as np
import pandas as pd
import pytest

import m_audit as MA


def test_scores_matches_hand_computed_metrics():
    """y_train (in-sample, m=1 lag-1 diff scale) = [10,11,9,12]; y_true=[1,3,4]; y_pred=[2,3,5].
    Hand-computed outside the module:
      e = y_true - y_pred = [-1, 0, -1]; ae = [1, 0, 1]
      scale (mean abs lag-1 diff of y_train): |11-10|,|9-11|,|12-9| = 1,2,3 -> mean 2.0
      scale_sq (mean sq lag-1 diff): 1,4,9 -> mean 14/3
      MASE = mean(ae)/scale = (2/3)/2.0 = 0.333333...
      RMSSE = sqrt(mean(e^2)/scale_sq) = sqrt((2/3)/(14/3)) = sqrt(1/7) = 0.377964...
      sMAPE = mean(2*|e|/(|y_true|+|y_pred|)) = mean([2/3, 0, 2/9]) = 0.296296...
      RMSE = sqrt(mean(e^2)) = sqrt(2/3) = 0.816497...
      MAE = mean(ae) = 2/3 = 0.666667...
    """
    y_train = np.array([10.0, 11.0, 9.0, 12.0])
    y_true = np.array([1.0, 3.0, 4.0])
    y_pred = np.array([2.0, 3.0, 5.0])
    out = MA.scores(y_train, y_true, y_pred, m=1)
    assert out["MASE"] == pytest.approx(1 / 3)
    assert out["RMSSE"] == pytest.approx(np.sqrt(1 / 7))
    assert out["sMAPE"] == pytest.approx((2 / 3 + 0 + 2 / 9) / 3)  # mean of the 3 per-point terms
    assert out["RMSE"] == pytest.approx(np.sqrt(2 / 3))
    assert out["MAE"] == pytest.approx(2 / 3)


def test_scores_uses_seasonal_lag_m_not_first_order_diff_when_m_greater_than_one():
    """m>1 must use a LAG-m diff (y[m:] - y[:-m]), not a first-order diff -- the exact distinction
    m_audit.py's inline comment at m_audit.py:138 calls out ("LAG-m seasonal diff, not m-th-ORDER
    diff"). A monthly (m=12) series with strong period-12 seasonality and near-zero month-to-month
    first differences should have scale dominated by the seasonal (not adjacent-month) comparison."""
    # 24 months, period-12 seasonal pattern, identical every cycle -> lag-12 diffs are all 0.
    y_train = np.tile(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]), 2)
    y_true = np.array([1.0])
    y_pred = np.array([1.0])
    out = MA.scores(y_train, y_true, y_pred, m=12)
    # lag-12 diffs are exactly 0 everywhere -> scale falls back to the |y_train|.mean()+eps branch
    assert out["MASE"] == pytest.approx(0.0)  # e=0 regardless of scale, so this alone doesn't
    # distinguish lag-m from first-order -- check the fallback scale directly via a nonzero-error case.
    y_pred2 = np.array([2.0])
    out2 = MA.scores(y_train, y_true, y_pred2, m=12)
    expected_scale = np.abs(y_train).mean() + 1e-9  # lag-12 diffs are 0 -> falls back to this branch
    assert out2["MASE"] == pytest.approx(1.0 / expected_scale)


def _write_fake_m4_csv(path, ids, n_train, n_test, seed):
    rng = np.random.default_rng(seed)
    train_rows, test_rows = [], []
    for sid in ids:
        train_rows.append([sid] + list(100 + rng.normal(0, 5, n_train)))
        test_rows.append([sid] + list(100 + rng.normal(0, 5, n_test)))
    cols_tr = ["V1"] + [f"V{i+2}" for i in range(n_train)]
    cols_te = ["V1"] + [f"V{i+2}" for i in range(n_test)]
    pd.DataFrame(train_rows, columns=cols_tr).to_csv(path[0], index=False)
    pd.DataFrame(test_rows, columns=cols_te).to_csv(path[1], index=False)


def test_emit_produces_distinct_weekly_and_monthly_values_not_clobbered(monkeypatch, tmp_path):
    """canonical_emit.flush() OVERWRITES a stage's whole fragment on every call (by design) -- the
    docstring FIXED-note on run_one() says the old single-frequency CLI couldn't safely emit both
    frequencies for exactly this reason, and --emit was added to run both in one process before a
    single flush(). This test locks in that both maudit.flip_mase_rmsse_weekly and
    maudit.flip_mase_rmsse_monthly actually land in the SAME emitted fragment, distinctly, rather
    than the second call silently overwriting the first."""
    data_dir = tmp_path / "data"
    results_dir = tmp_path / "results"
    data_dir.mkdir()
    results_dir.mkdir()
    monkeypatch.setattr(MA, "DATA", data_dir)
    monkeypatch.setattr(MA, "RESULTS", results_dir)

    weekly_files = (data_dir / "Weekly-train.csv", data_dir / "Weekly-test.csv")
    monthly_files = (data_dir / "Monthly-train.csv", data_dir / "Monthly-test.csv")
    _write_fake_m4_csv(weekly_files, [f"W{i}" for i in range(6)], n_train=30, n_test=4, seed=1)
    _write_fake_m4_csv(monthly_files, [f"M{i}" for i in range(6)], n_train=30, n_test=4, seed=2)

    monkeypatch.setattr(MA, "fetch", lambda freq: (data_dir / f"{freq}-train.csv", data_dir / f"{freq}-test.csv"))

    import json
    import canonical_emit as CE
    monkeypatch.setattr(CE, "_PENDING", {})
    frag_dir = tmp_path / "canonical_out"
    monkeypatch.setattr(CE, "_FRAGDIR", frag_dir)
    monkeypatch.setattr(CE, "_ROOT", tmp_path)  # flush()'s own print does out.relative_to(_ROOT)

    monkeypatch.setattr("sys.argv", ["m_audit.py", "--emit"])
    MA.main()

    doc = json.loads((frag_dir / "m_audit.json").read_text())
    assert "maudit.flip_mase_rmsse_weekly" in doc
    assert "maudit.flip_mase_rmsse_monthly" in doc
    assert "maudit.any_flip_weekly" in doc
    assert "maudit.any_flip_monthly" in doc
    # both frequencies' n counts must be independently correct (not one overwriting the other's n)
    assert doc["maudit.flip_mase_rmsse_weekly"]["n"] == 6
    assert doc["maudit.flip_mase_rmsse_monthly"]["n"] == 6


def test_run_one_writes_csv_and_returns_summary_matching_it(monkeypatch, tmp_path):
    """run_one()'s returned summary dict (n, flip_mase_rmsse, any_flip) must match what's actually
    written to its own per-frequency CSV -- the object canonical.py's cross_check points at."""
    data_dir = tmp_path / "data"
    results_dir = tmp_path / "results"
    data_dir.mkdir()
    results_dir.mkdir()
    monkeypatch.setattr(MA, "DATA", data_dir)
    monkeypatch.setattr(MA, "RESULTS", results_dir)
    files = (data_dir / "Weekly-train.csv", data_dir / "Weekly-test.csv")
    _write_fake_m4_csv(files, [f"S{i}" for i in range(10)], n_train=40, n_test=5, seed=7)
    monkeypatch.setattr(MA, "fetch", lambda freq: files)

    summary = MA.run_one("Weekly")
    csv = pd.read_csv(results_dir / "m_audit_Weekly.csv")
    assert summary["n"] == len(csv)
    recomputed_flip = (csv["win_MASE"] != csv["win_RMSSE"]).mean()
    assert summary["flip_mase_rmsse"] == pytest.approx(recomputed_flip)


def test_run_one_raises_a_clear_error_when_every_series_is_filtered_out(monkeypatch, tmp_path):
    """The empty-series guard, unreachable on M4's real population sizes but not structurally
    impossible. Every series here has only 3 train observations (< the len(y)<6 filter), so `rows`
    ends up empty and `d = pd.DataFrame(rows)` is a columnless empty DataFrame -- run_one() must
    fail with a clear, named diagnostic instead of an opaque pandas KeyError on `d["win_MASE"]`."""
    data_dir = tmp_path / "data"
    results_dir = tmp_path / "results"
    data_dir.mkdir()
    results_dir.mkdir()
    monkeypatch.setattr(MA, "DATA", data_dir)
    monkeypatch.setattr(MA, "RESULTS", results_dir)
    files = (data_dir / "Weekly-train.csv", data_dir / "Weekly-test.csv")
    _write_fake_m4_csv(files, [f"S{i}" for i in range(5)], n_train=3, n_test=5, seed=7)
    monkeypatch.setattr(MA, "fetch", lambda freq: files)

    with pytest.raises(ValueError, match=r"0 of 5 series had"):
        MA.run_one("Weekly")
    assert not (results_dir / "m_audit_Weekly.csv").exists()

"""Mutation-verified test for m_audit.run_one()'s per-series winner tie-break: `min(sorted(sc),
key=...)` breaks exact score ties ALPHABETICALLY by model name, which was found inert on the real
Weekly panel (0 exact ties) and untested either way. The mutant `min(sc, key=...)` breaks ties by
MODELS insertion order instead. A planted tie -- two forecasters returning identical forecasts,
registered with the alphabetically-later name FIRST -- separates the two rules.
"""
import numpy as np
import pandas as pd

import m_audit as MA


def _write_m4_layout(data_dir, ids, n_train=30, n_test=5, seed=0):
    rng = np.random.default_rng(seed)
    tr = pd.DataFrame([[i] + list(100 + rng.normal(0, 1, n_train).cumsum()) for i in ids],
                      columns=["V1"] + [f"V{j}" for j in range(2, n_train + 2)])
    te = pd.DataFrame([[i] + list(100 + rng.normal(0, 1, n_test)) for i in ids],
                      columns=["V1"] + [f"V{j}" for j in range(2, n_test + 2)])
    tr.to_csv(data_dir / "Weekly-train.csv", index=False)
    te.to_csv(data_dir / "Weekly-test.csv", index=False)
    return data_dir / "Weekly-train.csv", data_dir / "Weekly-test.csv"


def test_exact_score_ties_are_broken_alphabetically_not_by_registration_order(monkeypatch, tmp_path):
    """kills: `min(sorted(sc), key=lambda mo: sc[mo][metric])` -> `min(sc, key=...)`. 'zeta' and
    'alpha' are the SAME forecaster (last-value persistence) so every metric ties exactly; 'zeta'
    is registered first. The documented rule picks 'alpha' on every series and every metric."""
    data_dir = tmp_path / "data"; data_dir.mkdir()
    results_dir = tmp_path / "results"; results_dir.mkdir()
    monkeypatch.setattr(MA, "DATA", data_dir)
    monkeypatch.setattr(MA, "RESULTS", results_dir)
    files = _write_m4_layout(data_dir, [f"W{i}" for i in range(8)])
    monkeypatch.setattr(MA, "fetch", lambda freq: files)

    def persist(y, m, h):
        return np.repeat(y[-1], h)

    monkeypatch.setattr(MA, "MODELS", {"zeta": persist, "alpha": persist})   # deliberate order
    summary = MA.run_one("Weekly")
    out = pd.read_csv(results_dir / "m_audit_Weekly.csv")
    assert summary["n"] == 8
    for metric in ["MASE", "RMSSE", "sMAPE", "RMSE", "MAE"]:
        assert (out[f"win_{metric}"] == "alpha").all(), (
            f"{metric}: tie between identical forecasters must resolve to the alphabetically-first "
            f"name, got {out[f'win_{metric}'].unique().tolist()}")
    assert summary["any_flip"] == 0.0 and summary["mean_distinct"] == 1.0

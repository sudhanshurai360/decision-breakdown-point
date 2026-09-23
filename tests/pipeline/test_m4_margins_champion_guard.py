"""m4_margins.py: main()'s `champion` is reused as every metric's row label, which is only valid
under Pareto-dominance (the same submission is the argmin on every metric). A comment asserting
that is not a check, so main() asserts it in code before trusting the reuse. These tests exercise
both the happy path (dominance holds, matching the real leaderboard's
current shape) and the guard actually firing when it does not, on a small synthetic leaderboard so
neither depends on `multiseries/results/m4_submissions_leaderboard.csv`.
"""
import pandas as pd
import pytest

import m4_margins as MM


def _write_leaderboard(tmp_path, rows):
    df = pd.DataFrame(rows).set_index("submission")
    (tmp_path / "m4_submissions_leaderboard.csv").parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(tmp_path / "m4_submissions_leaderboard.csv")


def test_main_labels_the_pareto_dominant_submission_as_champion_on_every_metric(monkeypatch, tmp_path):
    _write_leaderboard(tmp_path, [
        {"submission": "champ", "sMAPE": 0.10, "MAPE": 0.10, "MASE": 1.00, "RMSSE": 1.00},
        {"submission": "rival", "sMAPE": 0.12, "MAPE": 0.11, "MASE": 1.05, "RMSSE": 1.02},
    ])
    monkeypatch.setattr(MM, "RES", tmp_path)
    emitted = {}
    monkeypatch.setattr(MM, "emit", lambda key, value, **kw: emitted.__setitem__(key, (value, kw)))
    monkeypatch.setattr(MM, "flush", lambda: None)

    MM.main()

    out = pd.read_csv(tmp_path / "m4_margins.csv")
    assert (out["champion"] == "champ").all()
    assert (out["runner_up"] == "rival").all()
    assert emitted["m4sub.gap_mase_pct"][1]["champion"] == "champ"


def test_main_raises_loudly_when_pareto_dominance_breaks(monkeypatch, tmp_path):
    """'champ' still wins MASE (idxmin, the label source) but LOSES sMAPE to 'other' -- the exact
    silent-mislabel shape the finding describes: without the guard, main() would print champion='champ'
    next to champ_val/runner_up drawn from sMAPE's own argmin ('other'), a value that does not belong
    to 'champ' at all."""
    _write_leaderboard(tmp_path, [
        {"submission": "champ", "sMAPE": 0.15, "MAPE": 0.10, "MASE": 1.00, "RMSSE": 1.00},
        {"submission": "other", "sMAPE": 0.05, "MAPE": 0.20, "MASE": 1.10, "RMSSE": 1.10},
    ])
    monkeypatch.setattr(MM, "RES", tmp_path)
    monkeypatch.setattr(MM, "emit", lambda *a, **k: None)
    monkeypatch.setattr(MM, "flush", lambda: None)

    with pytest.raises(AssertionError, match="Pareto-dominance broke"):
        MM.main()

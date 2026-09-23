"""realised_t.py::main()'s FLAGSHIP call and its completeness guard.

FLAGSHIP must not call analyse() directly, bypassing the `M.notna().all().all() and len(M) >= 20`
completeness guard the per-series loop above it applies before its own analyse() call. Every
fragility.py function analyse() calls except pooled_winner() has no NaN/inf validation of its own
(e.g. F.breakdown_number() silently returns k*=T -- "maximally fragile" -- on a NaN input instead of
erroring), so a gap here could have produced a confident-looking wrong FLAGSHIP row with no trace.
Currently inert on real data (extension_scoreboard_1988-2025.csv, T=38, has zero NaNs and >=20 rows)
-- these tests exercise the guard directly via a monkeypatched FLAG path, since real data alone
cannot discriminate whether the guard is actually wired in.
"""
import numpy as np
import pandas as pd
import pytest

import realised_t as RT


@pytest.fixture(autouse=True)
def _no_side_effects(monkeypatch, tmp_path):
    """Redirect every write/emit main() performs so this test cannot touch real result files.

    PRED carries one minimal, always-valid synthetic series (25 complete years, 2 models) so
    `out` is never left totally empty regardless of what happens to FLAGSHIP -- main()'s own
    final aggregation (spearmanr, emit(prop.*) etc.) requires at least one scored row and is not
    itself part of what these tests exercise."""
    rng = np.random.default_rng(1)
    years = list(range(1990, 2015))
    rows = []
    for y in years:
        base = rng.uniform(0.5, 1.0)
        rows.append({"series": "SYN", "model": "win", "year": y, "sae": base})
        rows.append({"series": "SYN", "model": "rival", "year": y, "sae": base + 0.5})
    monkeypatch.setattr(RT, "RES", tmp_path)
    monkeypatch.setattr(RT, "PRED", tmp_path / "predictions.csv")
    pd.DataFrame(rows).to_csv(RT.PRED, index=False)
    monkeypatch.setattr(RT, "emit", lambda *a, **k: None)
    monkeypatch.setattr(RT, "flush", lambda *a, **k: None)


def test_flagship_with_a_nan_is_dropped_not_silently_scored(monkeypatch, tmp_path, capsys):
    """A FLAGSHIP panel with a NaN cell must be dropped with a printed reason, exactly like the
    per-series loop's own incomplete-panel branch -- not passed through to analyse() (which has no
    NaN guard of its own and would silently mis-score it)."""
    bad = pd.DataFrame({"m1": [0.5] * 25, "m2": [0.6] * 25})
    bad.iloc[3, 1] = np.nan
    flag_path = tmp_path / "flagship_with_nan.csv"
    bad.to_csv(flag_path)
    monkeypatch.setattr(RT, "FLAG", flag_path)

    RT.main()
    out = capsys.readouterr().out
    assert "[drop] FLAGSHIP: incomplete or <20-year panel" in out
    assert "has gaps" in out


def test_flagship_too_short_is_dropped_not_silently_scored(monkeypatch, tmp_path, capsys):
    """A FLAGSHIP panel with fewer than 20 years must also be dropped with a printed reason."""
    short = pd.DataFrame({"m1": [0.5] * 10, "m2": [0.6] * 10})
    flag_path = tmp_path / "flagship_short.csv"
    short.to_csv(flag_path)
    monkeypatch.setattr(RT, "FLAG", flag_path)

    RT.main()
    out = capsys.readouterr().out
    assert "[drop] FLAGSHIP: incomplete or <20-year panel" in out
    assert "complete" in out and "has gaps" not in out.split("[drop] FLAGSHIP")[1].split("\n")[0]


def test_flagship_complete_panel_still_scores_normally(monkeypatch, tmp_path, capsys):
    """Negative control: a complete, >=20-year panel with a real winner must still reach analyse()
    and score normally -- the guard must not become overly strict and reject valid data."""
    rng = np.random.default_rng(0)
    T = 25
    win = rng.uniform(0.5, 1.0, T)
    good = pd.DataFrame({"win": win, "rival": win + 0.5})
    flag_path = tmp_path / "flagship_good.csv"
    good.to_csv(flag_path)
    monkeypatch.setattr(RT, "FLAG", flag_path)

    RT.main()
    out = capsys.readouterr().out
    assert "[drop] FLAGSHIP" not in out


def test_real_flagship_data_currently_satisfies_the_guard():
    """The guard is currently a no-op on real data -- confirms the fix's own claimed inertness
    directly, rather than just asserting it in a comment."""
    df = pd.read_csv(RT.FLAG, index_col=0)
    assert df.notna().all().all() and len(df) >= 20

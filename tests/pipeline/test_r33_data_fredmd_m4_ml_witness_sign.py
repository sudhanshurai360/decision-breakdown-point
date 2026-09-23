"""Mutation-verified test for ml_witness.py's sign convention: `L = 100.0 - S` (main(), line 95) is the only place the Open-LLM leaderboard's
higher-is-better scores become the instrument's lower-is-better losses. The mutant `L = S` changes
llm.top_model, llm.kstar, llm.reversal, llm.fragile, llm.winner_stability and every llm.crowd_real_*
value, and passes the whole suite plus every canonical.py cross-check (the only recomputed llm.*
checks -- n_models and the cross-task correlations -- are invariant to the sign).

main() is driven on a synthetic 40-model board with fetch() replaced, the slow null-budget
calls stubbed, and emit() captured; the planted answer is which model has the HIGHEST mean score.
"""
import contextlib
import io

import numpy as np
import pandas as pd
import pytest

import ml_witness as MW


def _board(seed=0, n=40):
    """40 models x 6 tasks of scores in (0, 100); model 'best' is lifted to ~95 on every task and
    model 'worst' pushed to ~5, so the top model by mean SCORE is 'best' and the top model by
    mean raw score-as-loss (the mutant) is 'worst'."""
    rng = np.random.default_rng(seed)
    S = rng.uniform(30, 70, size=(n, len(MW.TASKS)))
    names = [f"m{i:02d}" for i in range(n)]
    S[3] = rng.uniform(93, 97, len(MW.TASKS)); names[3] = "best"
    S[17] = rng.uniform(3, 7, len(MW.TASKS)); names[17] = "worst"
    df = pd.DataFrame(S, columns=MW.TASKS)
    df.insert(0, "model", names)
    return df


@pytest.fixture
def emitted(monkeypatch):
    """Run MW.main() on the synthetic board with every side effect stubbed; return {key: (value, kwargs)}."""
    df = _board()
    captured = {}

    def emit(key, value, **kw):
        captured[key] = (value, kw)

    monkeypatch.setattr(MW, "fetch", lambda *a, **k: df)
    monkeypatch.setattr(MW, "emit", emit)
    monkeypatch.setattr(MW, "flush", lambda *a, **k: None)
    # the two null-budget calls are minutes of bootstrap and orthogonal to the sign convention
    monkeypatch.setattr(MW.NB, "stable_stability",
                        lambda L, w, **k: {"value": 1.0, "sd": 0.0, "lo": 1.0, "hi": 1.0, "n_seeds": 1})
    monkeypatch.setattr(MW.NB, "stable_benchmark",
                        lambda K, T, **k: {"median": 0.5, "p05": 0.3, "p95": 0.8, "p05_lo": 0.3, "p05_hi": 0.3,
                                           "p05_sd": 0.0, "p95_lo": 0.8, "p95_hi": 0.8, "p95_sd": 0.0,
                                           "n_eff": 1, "n_seeds": 1})
    monkeypatch.setattr(MW, "crowd", lambda mat, rng: {k: 0.0 for k in (2, 4, 8, 16, 32)})
    monkeypatch.setattr(MW.AC, "average_null",
                        lambda L, rng, ks, first, **k: ({k: 0.0 for k in ks}, {k: 0.0 for k in ks},
                                                       {k: 0.0 for k in ks}))
    with contextlib.redirect_stdout(io.StringIO()):
        MW.main()
    return captured


def test_top_model_is_the_highest_mean_score_model(emitted):
    """kills: `L = 100.0 - S` -> `L = S` (the mutant crowns 'worst'). The leaderboard #1 by mean
    score must be the pooled winner the instrument reports."""
    assert emitted["llm.top_model"][0] == "best", emitted["llm.top_model"]


def test_kstar_rival_is_drawn_from_the_top_of_the_leaderboard_not_the_bottom(emitted):
    """kills: `L = S`. The candidate pool is `order[:30]` by mean LOSS, i.e. the 30 HIGHEST-scoring
    models -- so the binding rival named in llm.kstar must be one of them. Under the sign flip the
    same slice is the 30 LOWEST-scoring models, and the rival it names (m00 on this board, mean
    score 47.9, 31st or worse) is not in the top 30 by score at all."""
    S = _board()[MW.TASKS].to_numpy(float)
    names = list(_board()["model"])
    mean_score = S.mean(1)
    top30_by_score = {names[i] for i in np.argsort(-mean_score)[:30]}
    assert "best" in top30_by_score and "worst" not in top30_by_score   # fixture sanity

    rival = emitted["llm.kstar"][1]["rival"]
    assert rival in top30_by_score, (
        f"binding rival {rival!r} is outside the 30 highest-scoring models -- the candidate pool "
        f"was built from the wrong end of the leaderboard"
    )


def test_cross_task_corr_is_sign_invariant_and_counts_the_board(emitted):
    """The llm.* values canonical.py recomputes (the cross-task correlations) ARE sign-invariant
    -- documents why the sign convention needed its own test. Pearson r of (100 - S) columns
    equals that of S columns, and the board count is 40 either way."""
    val, kw = emitted["llm.mean_cross_task_corr"]
    S = _board()[MW.TASKS].to_numpy(float)
    expected = np.corrcoef(S.T)[np.triu_indices(len(MW.TASKS), 1)].mean()
    assert val == pytest.approx(expected)
    assert kw["n_models"] == 40

"""m4_perseries_kstar.py's main() orchestration: the NaN guard, the block=1 choice, and the
CSV/emit wiring, exercised on a small synthetic matrix rather than the real 61 x 100,000 M4 panel
-- that real run takes several minutes and is a reproduce.py "slow" stage, not something the fast
test suite should re-run. `build_matrix()` and `RES` are monkeypatched so these tests touch no real
data files.
"""
import numpy as np
import pandas as pd
import pytest

import m4_perseries_kstar as M4K


def _synthetic_matrix():
    """3 submissions x 12 series, no NaN. k* is small when a FEW periods carry a champion's entire
    net advantage while it's actually behind in most others -- not when an opponent is merely
    "close on average" with a small, uniform per-period gap (a uniform gap needs every period
    removed to flip, since every period's contribution has the same sign; verified by hand before
    writing this fixture). Losses are MASE-like, so LOWER is better: 'champ' beats 'rival' badly in
    3 periods (rival's loss is raised by ~1.0 there) and 'rival' beats champ narrowly in the other
    9 (rival's loss is lowered by ~0.2 there) -- champ still wins pooled (+3.0 against -1.8), but
    removing champ's few big-win periods flips it (k*=2, hand-verified: removing the 2
    largest-margin periods already flips the sign). 'also_ran' is
    uniformly worse than champ everywhere and must not be the binding opponent despite being
    'a' < 'r' alphabetically (decision_breakdown's own documented tie-break would otherwise mask a
    wiring bug that always picked the alphabetically-first model)."""
    rng = np.random.default_rng(0)
    champ = 1.0 + 0.02 * rng.normal(size=12)
    rival = champ.copy()
    rival[0:3] += 1.0 + 0.02 * rng.normal(size=3)     # champ wins big in 3 periods
    rival[3:] -= 0.2 + 0.02 * rng.normal(size=9)      # rival actually beats champ in the other 9
    also_ran = champ + 0.30 + 0.02 * rng.normal(size=12)   # uniformly worse than champ everywhere
    return pd.DataFrame({"champ": champ, "rival": rival, "also_ran": also_ran}).T


def test_main_computes_kstar_and_winner_stability_on_a_synthetic_panel(monkeypatch, tmp_path):
    monkeypatch.setattr(M4K, "build_matrix", _synthetic_matrix)
    monkeypatch.setattr(M4K, "RES", tmp_path)
    emitted = {}
    monkeypatch.setattr(M4K, "emit", lambda key, value, **kw: emitted.__setitem__(key, (value, kw)))
    monkeypatch.setattr(M4K, "flush", lambda: None)

    M4K.main()

    out = pd.read_csv(tmp_path / "m4_perseries_kstar.csv")
    assert len(out) == 1
    row = out.iloc[0]
    assert row["champion"] == "champ", "the lowest-mean-loss submission must be the pooled champion"
    assert row["binding_opponent"] == "rival", (
        "'rival' has the smaller margin to 'champ' and must be the binding opponent, not 'also_ran'"
    )
    assert 1 <= row["k_star"] <= 5, (
        f"champ's entire net advantage over rival is concentrated in 3 periods (of 12) -- k* should "
        f"be small (hand-verified at 2 on the noise-free construction), got {row['k_star']}. A "
        f"value of 12 would mean main() is treating the margin as uniformly diffuse rather than "
        f"concentrated, e.g. from a wrong sign or an unintended aggregation"
    )
    assert 0.0 <= row["winner_stability"] <= 1.0

    assert "m4sub.k_star" in emitted and "m4sub.winner_stability" in emitted
    k_val, k_kw = emitted["m4sub.k_star"]
    assert k_val == row["k_star"]
    assert k_kw["binding_opponent"] == "rival"
    ws_val, ws_kw = emitted["m4sub.winner_stability"]
    assert ws_val == pytest.approx(row["winner_stability"])
    assert ws_kw["block"] == 1, "block=1 (i.i.d. over series) is the documented, deliberate choice for this cross-sectional panel"


def test_main_raises_loudly_on_missing_values(monkeypatch, tmp_path):
    """A real gap in coverage must fail loudly, not silently drop to a different period set per
    model -- the reason main()'s own RuntimeError guard exists (m4_perseries_kstar.py:53-59)."""
    def _matrix_with_nan():
        M = _synthetic_matrix()
        M.iloc[0, 0] = np.nan
        return M

    monkeypatch.setattr(M4K, "build_matrix", _matrix_with_nan)
    monkeypatch.setattr(M4K, "RES", tmp_path)
    monkeypatch.setattr(M4K, "emit", lambda *a, **k: None)
    monkeypatch.setattr(M4K, "flush", lambda: None)

    with pytest.raises(RuntimeError, match="missing"):
        M4K.main()


def test_block1_is_iid_not_a_serial_dependence_assumption():
    """Documents and locks in the methodological choice the m4sub.winner_stability emit's own spec=
    string states ("i.i.d. resampling over series: the M4 series are a cross-section with no
    temporal adjacency, so a block bootstrap has nothing to preserve"): block=1
    makes _block_resample_idx (fragility.py) draw a plain i.i.d. resample regardless of the
    circular flag -- the correct scheme for a cross-sectional panel of unordered series, not a
    time axis. Guards against a future edit silently reverting to winner_stability's own
    time-series default (round(T ** (1/3))).
    """
    import inspect
    src = inspect.getsource(M4K.main)
    assert "block=1" in src, (
        "main() must call winner_stability with block=1 explicitly -- passing block=None here "
        "would silently apply a time-series default block length to M4's unordered series axis, "
        "exactly the methodological error that spec= string rules out."
    )

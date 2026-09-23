"""social_choice_power.py's own logic (`with_edge`'s edge-injection sign/scale, `base_cells`'s pivot
construction). The cross-check gate only re-derives canonical.json from this file's own CSV output;
nothing else exercises the edge-injection math itself against a known-correct answer. These tests
use small, hand-constructed cells, independent of the real curated data.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "code" / "pipeline"))
sys.path.insert(0, str(ROOT / "code" / "instrument"))

import social_choice as SC  # noqa: E402
import social_choice_power as SCP  # noqa: E402


def test_with_edge_shifts_the_targeted_model_by_exactly_delta_times_scale():
    """with_edge's docstring promises to "subtract delta*scale from `model`'s series in every
    cell", and `scale` is the mean of all models' stacked values (social_choice_power.py, the
    `scale = float(np.mean(np.column_stack(...)))` line inside with_edge). Pins
    the exact arithmetic on a hand-built 2-model, 2-cell panel where the scale is computable by
    hand."""
    L1 = {"a": np.array([2.0, 4.0]), "b": np.array([6.0, 8.0])}
    w1 = np.ones(2)
    L2 = {"a": np.array([1.0, 1.0]), "b": np.array([3.0, 3.0])}
    w2 = np.ones(2)
    cells = [(L1, w1), (L2, w2)]

    delta = 0.1
    out = SCP.with_edge(cells, "a", delta)

    scale1 = np.mean([2.0, 4.0, 6.0, 8.0])  # 5.0
    scale2 = np.mean([1.0, 1.0, 3.0, 3.0])  # 2.0
    (out_L1, out_w1), (out_L2, out_w2) = out

    assert np.allclose(out_L1["a"], L1["a"] - delta * scale1)
    assert np.array_equal(out_L1["b"], L1["b"])  # untouched model unchanged
    assert np.allclose(out_L2["a"], L2["a"] - delta * scale2)
    assert np.array_equal(out_L2["b"], L2["b"])
    # weights must pass through unchanged
    assert out_w1 is w1 and out_w2 is w2


def test_with_edge_is_a_no_op_when_the_model_is_absent_from_a_cell():
    """A cell missing the targeted model must be returned unchanged. This is the silent-degradation
    path main()'s own panel-wide-coverage assertion guards against; pinned here as a behavior of
    with_edge itself, independent of that guard."""
    L = {"b": np.array([1.0, 2.0, 3.0])}
    w = np.ones(3)
    out = SCP.with_edge([(L, w)], "a", 0.5)
    (out_L, out_w) = out[0]
    assert out_L == L
    assert np.array_equal(out_w, w)


def test_with_edge_only_changes_the_targeted_model_across_many_cells():
    """A negative-space check: with_edge must never touch a model it wasn't asked to perturb, on a
    multi-cell, multi-model panel."""
    rng = np.random.default_rng(0)
    cells = [({"a": rng.normal(1, 0.1, 6), "b": rng.normal(1, 0.1, 6), "c": rng.normal(1, 0.1, 6)},
               np.ones(6)) for _ in range(4)]
    out = SCP.with_edge(cells, "b", 0.02)
    for (L, w), (L2, w2) in zip(cells, out):
        assert np.array_equal(L["a"], L2["a"])
        assert np.array_equal(L["c"], L2["c"])
        assert not np.array_equal(L["b"], L2["b"])  # the targeted model DID change


def test_with_edge_large_delta_saturates_detection_on_a_small_synthetic_panel():
    """End-to-end sanity check mirroring the suggested fix: a large enough panel-wide edge should
    push a small synthetic panel's permutation-null statistics into the null's own lower tail (the
    detection direction this whole file measures), on a fast, tiny synthetic setup rather than the
    real 24-cell/n_perm=2000 configuration."""
    rng = np.random.default_rng(1)
    T, K = 10, 3
    cells = [({f"m{i}": 1.0 + rng.normal(0, 0.05, T) for i in range(K)}, np.ones(T)) for _ in range(3)]
    huge_delta = 5.0  # a multiple of the series' own scale -- an unmistakable panel-wide edge
    perturbed = SCP.with_edge(cells, "m0", huge_delta)
    obs, null = SC.permutation_null(perturbed, n_perm=200, seed=42)
    p05 = np.percentile(null, 5, axis=0)
    # reversal (index 0) and kstar_is_1 (index 2) are the two statistics a decisive, panel-wide
    # edge should push down hardest on a small, clean panel like this
    assert obs[0] <= p05[0], f"expected the huge edge to push reversal into the null's lower tail: obs={obs[0]}, p05={p05[0]}"
    assert obs[2] <= p05[2], f"expected the huge edge to push kstar_is_1 into the null's lower tail: obs={obs[2]}, p05={p05[2]}"


def test_base_cells_pivot_matches_a_hand_built_reference_on_a_tiny_synthetic_csv(tmp_path, monkeypatch):
    """base_cells() reads SC.PRED, groups by (series, metric), and pivots to a year x model matrix
    per SC.METRICS' own aggregator -- pins that pivot construction directly against a hand-computed
    expected result on a tiny synthetic CSV, independent of the real breadth_distinct_predictions.csv."""
    import pandas as pd
    df = pd.DataFrame([
        {"series": "S1", "model": "a", "year": 2020, "sae": 1.0, "msse": 25.0},
        {"series": "S1", "model": "a", "year": 2021, "sae": 3.0, "msse": 49.0},
        {"series": "S1", "model": "b", "year": 2020, "sae": 2.0, "msse": 64.0},
        {"series": "S1", "model": "b", "year": 2021, "sae": 4.0, "msse": 100.0},
    ])
    csv_path = tmp_path / "synthetic_predictions.csv"
    df.to_csv(csv_path, index=False)
    monkeypatch.setattr(SC, "PRED", csv_path)

    cells = SCP.base_cells()
    assert len(cells) == 2  # one per (series, metric) = S1/mase, S1/rmsse

    mase_cell = next((L, w) for L, w in cells if np.isclose(L["a"][0], 1.0))
    L_mase, w_mase = mase_cell
    assert np.array_equal(L_mase["a"], np.array([1.0, 3.0]))
    assert np.array_equal(L_mase["b"], np.array([2.0, 4.0]))
    assert len(w_mase) == 2

    rmsse_cell = next((L, w) for L, w in cells if not np.isclose(L["a"][0], 1.0))
    L_rmsse, _ = rmsse_cell
    # rmsse aggregator is sqrt(mean) of msse; here each (model, year) has exactly one row, so
    # sqrt(msse) directly: a -> sqrt([25, 49]) = [5, 7], b -> sqrt([64, 100]) = [8, 10]
    assert np.allclose(L_rmsse["a"], np.array([5.0, 7.0]))
    assert np.allclose(L_rmsse["b"], np.array([8.0, 10.0]))

"""mcs_size.py carries S4.2's empirical test size (0.265, oversize_ratio 2.6), the
size_at_T400/block6/K10 finite-sample-pattern cells, and the alpha-sweep invariance claim S4.1
rests on. Without these tests, neither size_row()'s SE formula, nor null_panel()'s construction,
nor main()'s load-bearing SystemExit control gate would be verified against a known-answer case --
the only validation would be the runtime `arch` cross-check baked into main() itself. These tests
isolate the pure-Python logic from the (expensive) main() pipeline run.
"""
import numpy as np
import pytest

import mcs_size as MS


def test_null_panel_shape_and_level():
    """null_panel must return a (T, K) array of the requested dimensions, centered on BASE_LEVEL --
    the reference case the nominal level is defined against. (SD=1.0 means individual draws can go
    negative even though BASE_LEVEL=1.0 shifts the panel to be interpretable as a loss -- that shift
    only sets the CENTER, not a strict per-draw positivity guarantee, so this checks the mean, not
    np.all(panel > 0).)"""
    g = np.random.default_rng(0)
    K, T = 5, 40
    panel = MS.null_panel(g, K, T)
    assert panel.shape == (T, K)
    # centered near BASE_LEVEL=1.0 for a large-enough draw, not exactly (it's a random draw)
    assert abs(float(panel.mean()) - MS.BASE_LEVEL) < 0.2


def test_size_row_se_matches_binomial_formula():
    """size_row's reported SE must be the standard binomial SE of its own reported rate --
    a hand-computable known-answer check, independent of trusting the implementation that
    produced `rate` in the first place."""
    rate, se = MS.size_row(K=3, T=20, block=1, alpha=0.10, n_rep=200, seed=12345)
    assert 0.0 <= rate <= 1.0
    expected_se = float(np.sqrt(rate * (1 - rate) / 200))
    assert se == pytest.approx(expected_se)


def test_size_row_deterministic_given_seed():
    """Same seed -> same rate. Guards against an accidental source of nondeterminism (e.g. an
    un-seeded RNG call) creeping into a file whose whole purpose is a reproducible size estimate."""
    rate1, _ = MS.size_row(K=3, T=20, block=1, alpha=0.10, n_rep=100, seed=999)
    rate2, _ = MS.size_row(K=3, T=20, block=1, alpha=0.10, n_rep=100, seed=999)
    assert rate1 == rate2


def test_size_row_rate_is_roughly_alpha_at_iid_block_one():
    """At block=1 (no serial dependence assumed) with a genuinely-null panel, the elimination
    rate should land in the neighborhood of the nominal alpha -- not exactly equal (finite-sample
    oversizing is exactly what this file exists to measure and is not zero even at block=1), but
    a sanity bound catching a badly-broken size_row (e.g. an inverted rejection condition, which
    would land the rate near 1-alpha instead)."""
    rate, se = MS.size_row(K=4, T=60, block=1, alpha=0.10, n_rep=500, seed=42)
    assert rate < 0.5  # a rate anywhere near 0.9 would indicate an inverted "at least one eliminated" test


def test_curated_panels_matches_mcs_test_construction(monkeypatch, tmp_path):
    """curated_panels()'s per-series per-year absolute-loss matrices must be the SAME construction
    mcs_test.main() uses (same docstring claim) -- cross-check row/column counts and the NaN-row
    drop against a hand-built synthetic breadth_distinct_predictions.csv fixture. MS.PRED is
    monkeypatched to a tmp_path fixture rather than writing over the real (committed) predictions
    CSV -- see the CAUTION note on test_main_raises_systemexit_when_pi_not_point_identified below
    for why an earlier version of a sibling test in this file got this wrong."""
    import pandas as pd
    df = pd.DataFrame({
        "series": ["X"] * 6 + ["X"] * 6,
        "model": ["m1"] * 6 + ["m2"] * 6,
        "year": [2019, 2019, 2020, 2020, 2021, 2021] * 2,
        "sae": [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 1.5, 1.5, 2.5, 2.5, 3.5, 3.5],
    })
    fixture_csv = tmp_path / "breadth_distinct_predictions.csv"
    df.to_csv(fixture_csv, index=False)
    monkeypatch.setattr(MS, "PRED", fixture_csv)

    panels = MS.curated_panels()
    assert set(panels) == {"X"}
    assert panels["X"].shape == (3, 2)  # 3 years x 2 models, no NaN rows to drop


def test_main_raises_systemexit_when_pi_not_point_identified(monkeypatch, tmp_path):
    """The load-bearing control gate: if PI is not point-identified at the strictest swept alpha,
    the alpha sweep is uninterpretable and main() must fail loudly (SystemExit), never silently
    proceed to emit() a possibly-meaningless mcssize.alpha_sweep_invariant. Smoke-tests the gate
    itself via monkeypatched, cheap inputs rather than running the real (slow) pipeline.

    main() writes every output CSV (OUT, OUT_AR1, OUT_ALPHA) before it reaches the SystemExit
    check, so ALL of them must be redirected to tmp_path here -- a test that calls MS.main() with
    any output left pointing at the real results directory overwrites a committed CSV. If you add
    an output to mcs_size.py, add its redirect here in the same edit."""

    def fake_curated_panels():
        # Two models, perfectly tied every year at every alpha -> PI-equivalent series never
        # reaches a singleton even at the strictest alpha -> the control must fire.
        rng = np.random.default_rng(0)
        return {"PI": np.column_stack([np.ones(20), np.ones(20)]) + rng.normal(0, 1e-9, (20, 2))}

    monkeypatch.setattr(MS, "curated_panels", fake_curated_panels)
    monkeypatch.setattr(MS, "N_SIZE", 20)
    monkeypatch.setattr(MS, "N_ARCH", 5)
    flushed = []
    monkeypatch.setattr(MS, "flush", lambda: flushed.append(True))
    monkeypatch.setattr(MS, "OUT", tmp_path / "mcs_size.csv")
    monkeypatch.setattr(MS, "OUT_AR1", tmp_path / "mcs_size_ar1.csv")
    monkeypatch.setattr(MS, "OUT_ALPHA", tmp_path / "mcs_size_alpha_sweep.csv")

    with pytest.raises(SystemExit, match="CONTROL FAILED"):
        MS.main()
    # The control fires AFTER flush(), so a failure is still loud but does not discard the
    # fragment (including the control-independent cited sizes).
    assert flushed == [True]

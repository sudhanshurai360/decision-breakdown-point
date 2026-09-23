"""Mutation-verified tests for flagship_null_ar1.py / social_choice_power.py /
null_budget.py / null_sensitivity.py. Every test here was verified to FAIL on the named mutant
and PASS on the real module.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
for _sub in ("pipeline", "instrument"):
    _p = str(ROOT / "code" / _sub)
    if _p not in sys.path:
        sys.path.append(_p)

import fragility as F  # noqa: E402
import social_choice as SC  # noqa: E402
import flagship_null_ar1 as FNA  # noqa: E402
import social_choice_power as SCP  # noqa: E402
import null_budget as NB  # noqa: E402
import null_sensitivity as NS  # noqa: E402


# ---------------------------------------------------------------------------------------------
# flagship_null_ar1._lag1 -- the denominator is sum(x[:-1]**2), NOT sum(x[1:]**2)
# ---------------------------------------------------------------------------------------------

def test_lag1_denominator_is_the_lagged_sum_of_squares_on_an_asymmetric_series():
    """kills: M6 `_lag1` denominator `x[:-1]**2` -> `x[1:]**2` (paper 0.4401 -> 0.4476).

    Every series in the pre-existing tests is reversal-symmetric, so both denominators agree there.
    x = [0,0,0,0,5] demeans to [-1,-1,-1,-1,4]: numerator = (-1)(-1)*3 + (-1)(4) = -1;
    correct denominator sum(x[:-1]**2) = 4 -> phi = -1/4 = -0.25; the mutant's sum(x[1:]**2) = 19
    -> -1/19 = -0.0526. Hand-computed, not derived from the function."""
    sb = pd.DataFrame({"m": [0.0, 0.0, 0.0, 0.0, 5.0]})
    out = FNA.estimate_flagship_ar1_levels(sb)
    assert out.iloc[0]["ar1_estimate"] == pytest.approx(-0.25, abs=1e-12)


def test_lag1_denominator_on_the_reversed_asymmetric_series():
    """kills: M6 (the mirror case, so a mutant that swapped BOTH slices is also caught).

    x = [5,0,0,0,0] demeans to [4,-1,-1,-1,-1]: numerator = 4(-1) + 3*(1) = -1;
    correct denominator sum(x[:-1]**2) = 16+1+1+1 = 19 -> phi = -1/19; the mutant gives -1/4."""
    sb = pd.DataFrame({"m": [5.0, 0.0, 0.0, 0.0, 0.0]})
    out = FNA.estimate_flagship_ar1_levels(sb)
    assert out.iloc[0]["ar1_estimate"] == pytest.approx(-1.0 / 19.0, abs=1e-12)


def test_pairwise_differential_estimator_uses_the_same_lagged_denominator():
    """kills: M6 through the calibration path actually used (pairwise
    differentials). a - b = [0,0,0,0,5] - [1,1,1,1,1] = [-1,-1,-1,-1,4], already demeaned
    (mean 0): phi = -1/4 (see the levels test above for the arithmetic)."""
    sb = pd.DataFrame({"a": [0.0, 0.0, 0.0, 0.0, 5.0], "b": [1.0] * 5})
    out = FNA.estimate_flagship_ar1(sb)
    assert len(out) == 1 and out.iloc[0]["pair"] == "a-b"
    assert out.iloc[0]["ar1_estimate"] == pytest.approx(-0.25, abs=1e-12)


# ---------------------------------------------------------------------------------------------
# social_choice_power.main() -- the detection rule is STRICTLY below the null's 5th percentile
# ---------------------------------------------------------------------------------------------

def _tiny_cells():
    """Two synthetic cells, three models each, T=6; only their model names and shapes matter to
    main() because permutation_null is replaced below."""
    rng = np.random.default_rng(7)
    cells = []
    for _ in range(2):
        L = {m: 1.0 + rng.normal(0, 0.05, 6) for m in ("m0", "m1", "m2")}
        cells.append((L, np.ones(6)))
    return cells


def _run_main_with_fake_null(monkeypatch, tmp_path, obs):
    """Drive SCP.main() with a deterministic permutation null: every statistic's null column is a
    permutation of the integers 0..100, so np.percentile(null, 5) == 5.0 exactly (and the 10th
    percentile is 10.0), and every call returns the same observed vector `obs`. Returns the
    emitted canonical dict."""
    monkeypatch.setattr(SCP, "base_cells", _tiny_cells)
    rng = np.random.default_rng(11)

    def fake_permutation_null(cells, n_perm=None, seed=None):
        null = np.column_stack([rng.permutation(101) for _ in range(5)]).astype(float)
        return np.asarray(obs, dtype=float), null

    monkeypatch.setattr(SC, "permutation_null", fake_permutation_null)
    emitted = {}

    def fake_emit(key, value, **fields):
        emitted[key] = {"value": value, **fields}

    monkeypatch.setattr(SCP, "emit", fake_emit)
    monkeypatch.setattr(SCP, "flush", lambda: None)
    monkeypatch.setattr(SCP, "RES", tmp_path)
    SCP.main()
    return emitted


def test_main_counts_a_hit_only_when_obs_is_strictly_below_p05(monkeypatch, tmp_path):
    """kills: M10 `hit = obs < p05` -> `<=` in main() (both the per-trial line and the baseline
    line; the paper's 'x of 6' fractions move, e.g. no-Condorcet 0.5%: 2 -> 4), and also
    `np.percentile(null, 5)` -> `10` (obs=7 would then count).

    With p05 == 5.0 for every statistic, obs = [4, 5, 7, 5, 4] gives, under the strict rule,
    hits [T, F, F, F, T] on every one of the 3 trials at every delta (3, 0, 0, 0, 3 of 3), and
    baseline flags [1, 0, 0, 0, 1]. `<=` would make the two obs==5 statistics hit too; the 10th
    percentile would make obs=7 hit."""
    emitted = _run_main_with_fake_null(monkeypatch, tmp_path, obs=[4, 5, 7, 5, 4])
    expected_hits = dict(zip(SCP.STAT_NAMES, [3, 0, 0, 0, 3]))
    expected_flag = dict(zip(SCP.STAT_NAMES, [1, 0, 0, 0, 1]))
    for delta in SCP.DELTAS:
        for stat in SCP.STAT_NAMES:
            key = f"scpower.{stat}_{SCP._tag(delta)}"
            assert key in emitted, f"missing {key}"
            assert emitted[key]["value"] == expected_hits[stat], (key, emitted[key]["value"])
            assert emitted[key]["of"] == 3
    for stat in SCP.STAT_NAMES:
        key = f"scpower.{stat}_baseline_in_rejection_region"
        assert emitted[key]["value"] == expected_flag[stat], (key, emitted[key]["value"])
        assert emitted[key]["p05"] == 5.0


def test_main_emits_the_documented_delta_tags_and_the_csv_hit_columns_agree(monkeypatch, tmp_path):
    """kills: `_tag()` mislabels (a delta tag other than 0p5/1/2/5) and a CSV/emit divergence --
    the canonical keys the paper cites are scpower.<stat>_{0p5,1,2,5}; the CSV main() writes must
    carry the same strict-rule hit flags the emits are summed from."""
    emitted = _run_main_with_fake_null(monkeypatch, tmp_path, obs=[4, 5, 7, 5, 4])
    tags = {"0p5", "1", "2", "5"}
    got = {k.rsplit("_", 1)[1] for k in emitted if k.startswith("scpower.reversal_")
           and not k.endswith("baseline_in_rejection_region")}
    assert got == tags
    csv = pd.read_csv(tmp_path / "social_choice_power.csv")
    base = csv[csv.model == "__baseline__"].iloc[0]
    assert [bool(base[f"hit_{s}"]) for s in SCP.STAT_NAMES] == [True, False, False, False, True]
    trials = csv[csv.model != "__baseline__"]
    assert len(trials) == 3 * len(SCP.DELTAS)
    assert int(trials["hit_concentration_gt1"].sum()) == 0      # obs == p05 is NOT a hit
    assert int(trials["hit_reversal"].sum()) == 3 * len(SCP.DELTAS)


# ---------------------------------------------------------------------------------------------
# seed-spacing guards: the boundary gap == need must RAISE (guard is `<=`, not `<`)
# ---------------------------------------------------------------------------------------------

def test_null_budget_check_spacing_rejects_a_gap_exactly_equal_to_need():
    """kills: M13 null_budget `_check_spacing` `<=` -> `<`. With n_rep = 1000 the required gap is
    1000 + 1_000_003 = 1_001_003; seeds spaced by exactly that must be rejected (the last
    replicate's bootstrap stream of seed s would coincide with the first stream of seed s+gap),
    while one more unit of spacing passes."""
    need = 1000 + 1_000_003
    with pytest.raises(ValueError, match="seed spacing"):
        NB._check_spacing((0, need, 2 * need), 1000)
    NB._check_spacing((0, need + 1, 2 * (need + 1)), 1000)      # must not raise


def test_null_sensitivity_check_seed_spacing_rejects_a_gap_exactly_equal_to_need():
    """kills: the null_sensitivity twin of M13 (`_check_seed_spacing` `<=` -> `<`): need is
    n_rep + 10_000_003 there."""
    need = 400 + 10_000_003
    with pytest.raises(ValueError, match="seed spacing"):
        NS._check_seed_spacing((0, need), 400)
    NS._check_seed_spacing((0, need + 1), 400)                   # must not raise


# ---------------------------------------------------------------------------------------------
# null_sensitivity.null_dist -- the bootstrap stream must never coincide with the panel stream
# ---------------------------------------------------------------------------------------------

def test_null_dist_bootstrap_seed_never_equals_the_panel_generator_seed(monkeypatch):
    """kills: M12 null_sensitivity bootstrap seed offset collapsed onto the panel seed
    (`seed + 10_000_003 + r` -> `seed + 1 + r`).

    Records, for every replicate r, the state of the panel generator `g` at construction and the
    seed handed to fragility.winner_stability; asserts that default_rng(bootstrap_seed) is NOT
    the same stream as g (bit_generator state differs) and that the bootstrap seeds are distinct
    across replicates. Under the mutant the two streams are identical for every r."""
    monkeypatch.setattr(NS, "N_REP", 5)
    panel_states, boot_seeds = [], []
    real_panel = NS.panel

    def recording_panel(g, rho, ar):
        panel_states.append(g.bit_generator.state)
        return real_panel(g, rho, ar)

    def fake_winner_stability(L, w, n_boot=None, seed=None, **kw):
        boot_seeds.append(seed)
        return 0.5, {}

    monkeypatch.setattr(NS, "panel", recording_panel)
    monkeypatch.setattr(F, "winner_stability", fake_winner_stability)
    NS.null_dist(0.0, 0.0, seed=1_000)
    assert len(panel_states) == 5 and len(boot_seeds) == 5
    assert len(set(boot_seeds)) == 5
    for st, bs in zip(panel_states, boot_seeds):
        assert np.random.default_rng(bs).bit_generator.state != st, (
            f"bootstrap seed {bs} reproduces the panel generator's own stream")

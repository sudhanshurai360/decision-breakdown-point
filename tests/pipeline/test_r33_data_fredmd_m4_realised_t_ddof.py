"""Mutation-verified test for realised_t.analyse(): the studentization uses the POPULATION sd (`d.std(ddof=0)`), which is what
Proposition 2.2 is stated for (S2.2). The mutant `ddof=1` moves prop.median_slack 0.285 -> 0.280,
flagship.t_binding 0.490 -> 0.484 and flagship.t_max 4.743 -> 4.68 without failing any other test,
because the existing realised_t tests read the committed CSV/JSON instead of recomputing.

Fixture with a hand answer: T = 20 years, the champion at a constant loss 1.0 and one rival at
1 + d with d alternating 1, 3 (mean 2, population variance exactly 1). Then
    t = sqrt(T) * mean(d) / sd_pop(d) = sqrt(20) * 2 / 1 = 8.94427...
whereas the sample-sd version gives sqrt(20) * 2 / sqrt(20/19) = 8.71780...
"""
import numpy as np
import pandas as pd
import pytest

import realised_t as RT

T = 20
D = np.array([1.0, 3.0] * (T // 2))            # mean 2, population var 1, sample var 20/19


def _panel():
    years = list(range(1990, 1990 + T))
    return pd.DataFrame({"champ": np.ones(T), "rival": 1.0 + D}, index=years)


def test_t_binding_uses_the_population_sd():
    """kills: ddof=0 -> ddof=1 in analyse()."""
    r = RT.analyse("SYN", _panel())
    assert r is not None and r["winner"] == "champ" and r["binding_rival"] == "rival"
    expected_pop = float(np.sqrt(T) * 2.0 / 1.0)
    expected_sample = float(np.sqrt(T) * 2.0 / np.sqrt(T / (T - 1)))
    assert r["t_binding"] == pytest.approx(expected_pop, rel=1e-12), r["t_binding"]
    assert r["t_max"] == pytest.approx(expected_pop, rel=1e-12)
    assert abs(r["t_binding"] - expected_sample) > 0.2, "t_binding is the sample-sd statistic"


def test_slack_is_population_t_over_the_proposition_ceiling():
    """kills: ddof=0 -> ddof=1 (slack = t / ceiling inherits the studentization, so the mutant
    also moves prop.median_slack, the number S2.2 quotes). Fixture: the rival now WINS four of
    the 20 years (d = -0.5), so k* is finite and the ceiling sqrt(T k/(T-k)) is a real number."""
    d = np.array([1.0, 3.0] * 8 + [-0.5] * 4)          # sum 30 > 0: champion still wins pooled
    years = list(range(1990, 1990 + T))
    M = pd.DataFrame({"champ": np.ones(T), "rival": 1.0 + d}, index=years)
    r = RT.analyse("SYN", M)
    k = r["k_star"]
    assert 0 < k < T, r
    ceiling = float(np.sqrt(T * k / (T - k)))
    t_pop = float(np.sqrt(T) * d.mean() / d.std(ddof=0))
    t_sample = float(np.sqrt(T) * d.mean() / d.std(ddof=1))
    assert r["ceiling_binding"] == pytest.approx(ceiling, rel=1e-12)
    assert r["t_binding"] == pytest.approx(t_pop, rel=1e-12)
    assert r["slack"] == pytest.approx(t_pop / ceiling, rel=1e-12)
    assert abs(r["slack"] - t_sample / ceiling) > 1e-3

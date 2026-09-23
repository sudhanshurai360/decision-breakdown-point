"""Mutation-verified behavioural coverage of best_arm.py's main() correlation / leverage logic.
Four number-changing mutations of main() (M5 gap2 = max(t), M6 exclude PCE instead of PI, M7
drop-pct denominator |ex_rho|, M8 H = sum 1/t) pass every other best_arm test, because those tests
cover only studentized_gap() (degenerate tie), the missing-cell guard (T bookkeeping), and the
committed best_arm.csv artefact (which they re-correlate themselves, locking the artefact, not the
code).

This file runs the REAL best_arm.main() on a synthetic 6-series x 3-model x 5-year panel whose
studentized gaps are chosen so every downstream number is an exact hand-derived fraction, with
PRED/MCSF/MULTI redirected to tmp_path and emit() captured in a dict. A closed-form construction
makes the hand derivation trivial:

    rival loss = best loss + a + b * [-2, -1, 0, 1, 2]      (T = 5 years)
    => dd = a + b*[-2..2], mean(dd) = a, sd(dd, ddof=1) = b*sqrt(2.5), se = b*sqrt(0.5)
    => t = a / (b*sqrt(0.5)) = a*sqrt(2)/b,   1/t^2 = b^2 / (2 a^2)

Every series uses b = 0.3; the (a1, a2) pairs and the planted |MCS| column are chosen so that the
six Spearman correlations and the two excl-PI drop percentages are distinct integers/fractions
under the real code and under each of the four mutants (see the per-test "kills:" notes). Every
test here was verified to FAIL on the named mutant and PASS on the real module.

Relies on tests/conftest.py for sys.path; it does NOT insert code/ paths itself.
"""
import math

import pandas as pd
import pytest

import best_arm as BA  # noqa: E402  (path via tests/conftest.py)

SQRT2 = math.sqrt(2.0)
B = 0.3
BASE = [1.0, 1.2, 0.9, 1.1, 1.0]          # the best arm's per-year losses (mean 1.04)
YEARS = [2019, 2020, 2021, 2022, 2023]
PATTERN = [-2, -1, 0, 1, 2]

# series -> (a for rival r1, a for rival r2).  t_i = a_i*sqrt(2)/0.3
DESIGN = {
    "PI":  (3.0, 4.0),
    "PCE": (1.0, 2.0),
    "A":   (0.5, 1.5),
    "B":   (0.3, 2.5),
    "C":   (0.6, 0.9),
    "D":   (0.4, 3.0),
}
# planted |MCS| (PI must be the ONLY identified series -- main() asserts that) and sci
MCS = {"PI": 1, "PCE": 5, "A": 4, "B": 6, "C": 3, "D": 2}
SCI = {"PI": 0.1, "PCE": 0.2, "A": 0.3, "B": 0.4, "C": 0.5, "D": 0.6}


def _t(a):
    return a * SQRT2 / B


def _expected():
    """Hand-derived per-series gap2_t and H, plus the Spearman rhos worked by rank arithmetic in
    the docstrings below (recomputed here from the same closed forms so the assertions are
    self-documenting; the rank arithmetic is spelled out in test_spearman_* docstrings)."""
    gap2 = {s: min(_t(a1), _t(a2)) for s, (a1, a2) in DESIGN.items()}
    H = {s: B**2 / (2 * a1**2) + B**2 / (2 * a2**2) for s, (a1, a2) in DESIGN.items()}
    return gap2, H


def _panel_rows():
    rows = []
    for sid, (a1, a2) in DESIGN.items():
        for j, y in enumerate(YEARS):
            rows.append({"series": sid, "model": "base", "year": y, "sae": BASE[j]})
            rows.append({"series": sid, "model": "r1", "year": y, "sae": BASE[j] + a1 + B * PATTERN[j]})
            rows.append({"series": sid, "model": "r2", "year": y, "sae": BASE[j] + a2 + B * PATTERN[j]})
    return rows


@pytest.fixture
def run_main(monkeypatch, tmp_path, capsys):
    """Run the real best_arm.main() on the synthetic panel; return (best_arm.csv frame, emits)."""
    pred = tmp_path / "pred.csv"
    pd.DataFrame(_panel_rows(), columns=["series", "model", "year", "sae"]).to_csv(pred, index=False)
    mcsf = tmp_path / "mcs_test.csv"
    pd.DataFrame({"series": list(MCS), "MCS_abs_10": list(MCS.values()),
                  "sci": [SCI[s] for s in MCS]}).to_csv(mcsf, index=False)
    (tmp_path / "results").mkdir()
    emits = {}

    def _emit(key, value, **kw):
        emits[key] = {"value": value, **kw}

    monkeypatch.setattr(BA, "PRED", pred)
    monkeypatch.setattr(BA, "MCSF", mcsf)
    monkeypatch.setattr(BA, "MULTI", tmp_path)
    monkeypatch.setattr(BA, "emit", _emit)
    monkeypatch.setattr(BA, "flush", lambda: None)
    BA.main()
    capsys.readouterr()
    r = pd.read_csv(tmp_path / "results" / "best_arm.csv").set_index("series")
    return r, emits


def test_gap2_t_is_the_smallest_rival_t_statistic(run_main):
    """kills: M5 `gap2 = min(tstat.values())` -> `max(...)`.
    Hand values (t = a*sqrt(2)/0.3): PI min(14.142, 18.856)=14.142; PCE min(4.714, 9.428)=4.714;
    A min(2.357, 7.071)=2.357; B min(1.414, 11.785)=1.414; C min(2.828, 4.243)=2.828;
    D min(1.886, 14.142)=1.886.  Under M5 every one of these becomes the larger t."""
    r, _ = run_main
    gap2, _ = _expected()
    for s in DESIGN:
        assert r.loc[s, "gap2_t"] == pytest.approx(gap2[s], rel=1e-9), s
        assert r.loc[s, "T"] == 5
    # explicit spot value: B's closest rival has a = 0.3 = b, so t = sqrt(2) exactly
    assert r.loc["B", "gap2_t"] == pytest.approx(SQRT2, rel=1e-9)


def test_H_is_sum_of_inverse_squared_t(run_main):
    """kills: M8 `H = sum 1/t^2` -> `sum 1/t` (no square).
    1/t^2 = b^2/(2a^2) with b=0.3: PI 0.005+0.0028125=0.0078125; PCE 0.045+0.01125=0.05625;
    A 0.18+0.02=0.2; B 0.5+0.0072=0.5072; C 0.125+0.0555..=0.18055..; D 0.28125+0.005=0.28625.
    Under M8 (sum b/(a*sqrt2)) e.g. PI would be 0.1237 and the A/C ordering flips."""
    r, _ = run_main
    _, H = _expected()
    for s in DESIGN:
        assert r.loc[s, "H"] == pytest.approx(H[s], rel=1e-9), s
    assert r.loc["A", "H"] == pytest.approx(0.2, rel=1e-9)
    assert r.loc["PI", "H"] == pytest.approx(0.0078125, rel=1e-9)


def test_spearman_gap_and_H_vs_mcs_match_hand_rank_arithmetic(run_main):
    """kills: M5 (gap rho -0.4857 -> -0.4286) and M8 (H rho +0.4857 -> +0.4286).
    gap2 ranks (asc): B1 D2 A3 C4 PCE5 PI6.  |MCS| ranks: PI1 D2 C3 A4 PCE5 B6.
      d = PI 5, PCE 0, A -1, B -5, C 1, D 0 -> sum d^2 = 52 -> rho = 1 - 6*52/(6*35) = -17/35.
    H ranks (asc): PI1 PCE2 C3 A4 D5 B6.  d = 0,-3,0,0,3,0 -> sum d^2 = 18 -> rho = 1-108/210 = 17/35.
    sci ranks: PI1 PCE2 A3 B4 C5 D6 vs |MCS| -> d = 0,-3,-1,-2,2,4 -> 34 -> rho = 1-204/210 = 1/35."""
    _, e = run_main
    assert e["bestarm.spearman_gap_mcs"]["value"] == pytest.approx(-17 / 35, abs=1e-12)
    assert e["bestarm.spearman_H_mcs"]["value"] == pytest.approx(17 / 35, abs=1e-12)
    assert e["bestarm.spearman_sci_mcs"]["value"] == pytest.approx(1 / 35, abs=1e-12)
    for k in ("bestarm.spearman_gap_mcs", "bestarm.spearman_H_mcs", "bestarm.spearman_sci_mcs",
              "bestarm.spearman_gap_H"):
        assert e[k]["n"] == 6
    # gap and H are near-inverse by construction here: gap ranks B1 D2 A3 C4 PCE5 PI6 vs
    # H ranks B6 D5 A4 C3 PCE2 PI1 -> d = -5,-3,-1,1,3,5 -> 70 -> rho = 1 - 420/210 = -1
    assert e["bestarm.spearman_gap_H"]["value"] == pytest.approx(-1.0, abs=1e-12)


def test_excl_pi_leverage_drops_exactly_pi_not_another_series(run_main):
    """kills: M6 `excl = r[r.series != "PI"]` -> `!= "PCE"`.
    Excluding PI (5 series PCE A B C D): gap ranks B1 D2 A3 C4 PCE5; |MCS| ranks D1 C2 A3 PCE4 B5
      d = PCE 1, A 0, B -4, C 2, D 1 -> 22 -> rho = 1 - 6*22/(5*24) = -0.1.
    H ranks PCE1 C2 A3 D4 B5 vs |MCS| -> d = -3,0,0,3,0 -> 18 -> rho = 1 - 108/120 = +0.1.
    Under M6 (dropping PCE instead) the gap rho would be -0.7, not -0.1."""
    _, e = run_main
    assert e["bestarm.spearman_gap_mcs_excl_pi"]["value"] == pytest.approx(-0.1, abs=1e-12)
    assert e["bestarm.spearman_H_mcs_excl_pi"]["value"] == pytest.approx(0.1, abs=1e-12)
    assert e["bestarm.spearman_gap_mcs_excl_pi"]["n"] == 5
    assert e["bestarm.spearman_H_mcs_excl_pi"]["n"] == 5


def test_excl_pi_drop_pct_uses_the_full_sample_rho_as_denominator(run_main):
    """kills: M7 `... / abs(gap_rho)` -> `/ abs(ex_gap_rho)` (and the H twin).
    drop = round(100 * (|rho_full| - |rho_excl|) / |rho_full|) = round(100*(17/35 - 1/10)/(17/35))
         = round(100 * 0.385714 / 0.485714) = round(79.41) = 79   (both gap and H by symmetry).
    Under M7 the denominator is 0.1 -> round(385.7) = 386."""
    _, e = run_main
    assert e["bestarm.spearman_gap_mcs_excl_pi_drop_pct"]["value"] == 79
    assert e["bestarm.spearman_H_mcs_excl_pi_drop_pct"]["value"] == 79


def test_identified_set_assert_fires_when_pi_is_not_the_only_identified_series(monkeypatch, tmp_path, capsys):
    """The PI guard (best_arm.py:80-85) must raise, not silently emit, when |MCS|==1 for a series
    other than PI -- the excl-PI 'drop the identified series' semantics would be wrong."""
    pred = tmp_path / "pred.csv"
    pd.DataFrame(_panel_rows(), columns=["series", "model", "year", "sae"]).to_csv(pred, index=False)
    bad = dict(MCS); bad["A"] = 1
    mcsf = tmp_path / "mcs_test.csv"
    pd.DataFrame({"series": list(bad), "MCS_abs_10": list(bad.values())}).to_csv(mcsf, index=False)
    (tmp_path / "results").mkdir()
    monkeypatch.setattr(BA, "PRED", pred)
    monkeypatch.setattr(BA, "MCSF", mcsf)
    monkeypatch.setattr(BA, "MULTI", tmp_path)
    monkeypatch.setattr(BA, "emit", lambda *a, **k: None)
    monkeypatch.setattr(BA, "flush", lambda: None)
    with pytest.raises(AssertionError, match="identified-series set changed"):
        BA.main()
    capsys.readouterr()


# --- GOLDEN PIN on the frozen curated panel ------------------------------------------------------

from pathlib import Path  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]     # tests/pipeline/ -> repo root (not BA.ROOT, so a
                                                 # shadowing module copy still finds the frozen data)
_REAL_PRED = _ROOT / "multiseries" / "results" / "breadth_distinct" / "breadth_distinct_predictions.csv"
_REAL_MCS = _ROOT / "multiseries" / "results" / "mcs_test.csv"


@pytest.mark.skipif(not (_REAL_PRED.exists() and _REAL_MCS.exists()),
                    reason="frozen breadth_distinct_predictions.csv / mcs_test.csv not available")
def test_golden_pin_bestarm_canonical_values_from_frozen_predictions(monkeypatch, tmp_path, capsys):
    """GOLDEN PIN: recomputes the S4.3 bestarm.* canonical
    values through the REAL main() from the frozen breadth_distinct_predictions.csv + mcs_test.csv,
    writing only to tmp_path. Protects: bestarm.spearman_gap_mcs (-0.3635), bestarm.spearman_H_mcs
    (+0.3373), bestarm.spearman_sci_mcs (-0.3260), bestarm.spearman_gap_mcs_excl_pi_drop_pct (57),
    bestarm.spearman_H_mcs_excl_pi_drop_pct (65), bestarm.spearman_gap_H (-0.986). Values were
    re-derived independently from the predictions file.
    kills: M5 (rho -0.36 -> -0.28, drop 57 -> 84), M6 (excl-PI rho -> -0.36, drop 57 -> 0),
    M7 (65 -> 182), M8 (H rho +0.34 -> +0.31, drop 65 -> 73) on the real data."""
    (tmp_path / "results").mkdir()
    emits = {}
    monkeypatch.setattr(BA, "PRED", _REAL_PRED)
    monkeypatch.setattr(BA, "MCSF", _REAL_MCS)
    monkeypatch.setattr(BA, "MULTI", tmp_path)
    monkeypatch.setattr(BA, "emit", lambda k, v, **kw: emits.__setitem__(k, v))
    monkeypatch.setattr(BA, "flush", lambda: None)
    BA.main()
    capsys.readouterr()
    assert emits["bestarm.spearman_gap_mcs"] == pytest.approx(-0.3634871348247943, abs=1e-6)
    assert emits["bestarm.spearman_H_mcs"] == pytest.approx(0.3372561044766132, abs=1e-6)
    assert emits["bestarm.spearman_sci_mcs"] == pytest.approx(-0.3260142343273928, abs=1e-6)
    assert emits["bestarm.spearman_gap_H"] == pytest.approx(-0.9860139860139862, abs=1e-6)
    assert emits["bestarm.spearman_gap_mcs_excl_pi"] == pytest.approx(-0.15449869255431758, abs=1e-6)
    assert emits["bestarm.spearman_H_mcs_excl_pi"] == pytest.approx(0.11961189100979426, abs=1e-6)
    assert emits["bestarm.spearman_gap_mcs_excl_pi_drop_pct"] == 57
    assert emits["bestarm.spearman_H_mcs_excl_pi_drop_pct"] == 65
    r = pd.read_csv(tmp_path / "results" / "best_arm.csv")
    assert len(r) == 12 and set(r.series) >= {"PI", "PCE", "UNEMPLOY"}

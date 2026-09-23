"""Hand-verified coverage of shape_not_magnitude.py's arithmetic.

shape_not_magnitude.py has no separate, importable pure-computation function -- every step (the
per-(series,target) difficulty = MIN over models, the hardest-month selection, the April-2020-vs-
median monthly multiple, the annual-aggregation dilution ratio) lives inline inside main(), which
also does its own file I/O against the real ~26,545-line breadth_distinct_predictions.csv. An
import/smoke check elsewhere in the suite does not hand-verify the arithmetic main() performs
against an independently-computed expected answer -- so a real bug (e.g. `.max()` where
shape_not_magnitude.py:18 takes `.min()` over models, or `.mean()` where `.median()` is required,
or the annual/monthly aggregation windows swapped) could silently flip the headline "12x monthly,
~2x annual" finding while every existing test stayed green.

This file closes that gap by monkeypatching PRED/OUT/HERE (and canonical_emit's internal state, the
same pattern test_m_audit_scores_and_emit.py uses for a main()-only module) to point at a tiny,
hand-computable synthetic prediction panel, then calling shape_not_magnitude.main() for real and
checking BOTH the written CSVs and the emitted canonical values against numbers computed by hand,
independently of the module's own code.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "code" / "pipeline"))
sys.path.insert(0, str(ROOT / "code" / "instrument"))

import shape_not_magnitude as SNM
import canonical_emit as CE


def _rows_for_series(series, month_starts, values, extra_model_offset=1000.0):
    """One (series, target) pair per (month_start, value): TWO model rows each -- 'real' carrying
    the intended `value`, and 'decoy' carrying value+extra_model_offset (obviously wrong and much
    larger). Difficulty is the MIN over models -- "the best any model achieved", the inline comment
    at shape_not_magnitude.py:18, i.e. the surprise NO model escaped. If main() ever used max/mean
    instead of min, the decoy's inflated value would leak
    into every downstream number and every hand-computed expectation below would fail."""
    rows = []
    for m, v in zip(month_starts, values):
        rows.append(dict(series=series, model="real", target=m, sae=float(v)))
        rows.append(dict(series=series, model="decoy", target=m, sae=float(v) + extra_model_offset))
    return rows


def _setup(monkeypatch, tmp_path, rows):
    pred_path = tmp_path / "pred.csv"
    pd.DataFrame(rows, columns=["series", "model", "target", "sae"]).to_csv(pred_path, index=False)

    fake_root = tmp_path / "fakeroot"
    (fake_root / "multiseries" / "results").mkdir(parents=True)
    monkeypatch.setattr(SNM, "PRED", pred_path)
    monkeypatch.setattr(SNM, "OUT", fake_root / "multiseries" / "results" / "shape_not_magnitude.csv")
    monkeypatch.setattr(SNM, "HERE", fake_root / "code" / "pipeline")
    monkeypatch.setattr(SNM, "ROOT_DIR", fake_root)

    frag_dir = fake_root / "canonical_out"
    monkeypatch.setattr(CE, "_PENDING", {})
    monkeypatch.setattr(CE, "_FRAGDIR", frag_dir)
    monkeypatch.setattr(CE, "_ROOT", fake_root)   # flush()'s print does out.relative_to(_ROOT)
    return frag_dir


def _canonical(frag_dir):
    return json.loads((frag_dir / "shape_not_magnitude.json").read_text())


# --- core hand-verified scenario: min-over-models, hardest-month bucket classification -----------

def test_shape_not_magnitude_hand_verified_ratios_and_buckets(monkeypatch, tmp_path, capsys):
    """3 series, deliberately covering all three hardest-month buckets:
      S1: 24 months (2019-01..2020-12), all sae=1.0 except 2020-04=12.0 (hardest -> April 2020).
          median=1.0 (only one outlier among 24) -> monthly ratio = 12.0/1.0 = 12.0.
          annual: 2019 mean=1.0, 2020 mean=(11*1+12)/12=23/12 -> ann ratio = (23/12)/((1+23/12)/2)
                  = 46/35 (exact fraction, hand-derived below).
      S2: 156 months (2008-01..2020-12), all sae=1.0 except 2009-06=5.0 (hardest -> GFC bucket).
          April 2020 sae=1.0 (no spike) -> monthly ratio = 1.0. annual ratio = 1.0 (worked below).
      S3: hardest month is 2010-01 (neither 2020 nor 2008/09) and has NO 2020 data at all, so it
          must be excluded from the ratio/annual-ratio computations via their own .dropna() (n=3
          series total, but len(ratio)==len(ann_ratio)==2) while still counting toward `n` and the
          hardest-bucket denominators.
    Every model row is doubled with a 'decoy' row valued +1000 (see _rows_for_series) -- if main()
    ever used max()/mean() instead of min() over models this test fails immediately and loudly.
    """
    s1_months = pd.date_range("2019-01-01", "2020-12-01", freq="MS")
    s1_vals = [1.0] * len(s1_months)
    s1_vals[list(s1_months).index(pd.Timestamp("2020-04-01"))] = 12.0

    s2_months = pd.date_range("2008-01-01", "2020-12-01", freq="MS")
    s2_vals = [1.0] * len(s2_months)
    s2_vals[list(s2_months).index(pd.Timestamp("2009-06-01"))] = 5.0

    s3_months = pd.date_range("2008-01-01", "2011-12-01", freq="MS")   # no 2020 data at all
    s3_vals = [1.0] * len(s3_months)
    s3_vals[list(s3_months).index(pd.Timestamp("2010-01-01"))] = 3.0

    rows = (_rows_for_series("S1", s1_months, s1_vals)
            + _rows_for_series("S2", s2_months, s2_vals)
            + _rows_for_series("S3", s3_months, s3_vals))
    frag_dir = _setup(monkeypatch, tmp_path, rows)

    SNM.main()
    capsys.readouterr()

    doc = _canonical(frag_dir)
    assert doc["shape.hardest_april2020"]["value"] == 1   # S1 only
    assert doc["shape.hardest_april2020"]["of"] == 3
    assert doc["shape.hardest_2020"]["value"] == 1         # S1 only
    assert doc["shape.hardest_gfc"]["value"] == 1           # S2 only (2009-06)

    # monthly ratios: S1=12.0, S2=1.0; S3 excluded (no April-2020 row at all) -> n_series=2.
    assert doc["shape.monthly_median_x"]["n_series"] == 2
    assert doc["shape.monthly_median_x"]["value"] == pytest.approx((12.0 + 1.0) / 2)   # 6.5
    assert doc["shape.monthly_max_x"]["value"] == pytest.approx(12.0)
    assert doc["shape.monthly_max_x"]["series"] == "S1"

    # annual ratios, hand-derived:
    #   S1: ann_2019=1.0, ann_2020=23/12 -> ratio = (23/12) / ((1 + 23/12)/2) = 46/35
    #   S2: ann_2020=1.0 (12 real months all =1.0); annual median across 13 years (12 years of
    #       exactly 1.0, one year -- 2009 -- of (11*1+5)/12=16/12=4/3) is 1.0 (the 1.0s form the
    #       majority) -> ratio = 1.0/1.0 = 1.0
    s1_ann_ratio = (23 / 12) / ((1.0 + 23 / 12) / 2)
    assert s1_ann_ratio == pytest.approx(46 / 35)
    s2_ann_ratio = 1.0
    expected_annual_median = float(np.median([s1_ann_ratio, s2_ann_ratio]))
    assert doc["shape.annual_median_x"]["n_series"] == 2
    assert doc["shape.annual_median_x"]["value"] == pytest.approx(expected_annual_median)

    # --- the written detail CSV must reflect the same MIN-over-models, hardest-row selection -----
    out = pd.read_csv(SNM.OUT)
    row_s1 = out[out.series == "S1"].iloc[0]
    assert row_s1["ym"] == "2020-04"
    assert row_s1["sae"] == pytest.approx(12.0)     # NOT 12.0+1000 (would mean max/mean leaked in)
    row_s2 = out[out.series == "S2"].iloc[0]
    assert row_s2["ym"] == "2009-06"
    assert row_s2["sae"] == pytest.approx(5.0)
    row_s3 = out[out.series == "S3"].iloc[0]
    assert row_s3["ym"] == "2010-01"
    assert row_s3["sae"] == pytest.approx(3.0)

    # --- the summary CSV's own headline row, independently re-derived from the same numbers ------
    summary = pd.read_csv(fake_root_summary(tmp_path))
    assert summary.loc[0, "n_series"] == 3
    assert summary.loc[0, "hardest_in_2020"] == 1
    assert summary.loc[0, "hardest_in_2008_09"] == 1
    assert summary.loc[0, "monthly_2020_04_median_x"] == pytest.approx(round(6.5, 1))
    assert summary.loc[0, "monthly_2020_04_max_x"] == pytest.approx(round(12.0, 1))
    assert summary.loc[0, "monthly_2020_04_max_series"] == "S1"
    assert summary.loc[0, "annual_2020_median_x"] == pytest.approx(round(expected_annual_median, 1))


def fake_root_summary(tmp_path):
    return tmp_path / "fakeroot" / "multiseries" / "results" / "shape_not_magnitude_summary.csv"


# --- difficulty must be MIN over models, isolated from the ratio machinery -----------------------

def test_shape_not_magnitude_difficulty_is_min_over_models_not_max(monkeypatch, tmp_path, capsys):
    """One series, two months, each with a 'lo' model (the intended difficulty value) and a 'hi'
    decoy valued far above it. The hardest-month row picked by main() must carry the 'lo' value --
    if difficulty were computed via max() instead of min() over models (a plausible one-character
    bug given the module's own 'best any model achieved' framing is easy to get backwards), this
    test catches it directly on the written CSV, independent of any ratio/median machinery."""
    months = [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-04-01")]
    rows = [
        dict(series="X", model="lo", target=months[0], sae=2.0),
        dict(series="X", model="hi", target=months[0], sae=50.0),
        dict(series="X", model="lo", target=months[1], sae=8.0),
        dict(series="X", model="hi", target=months[1], sae=99.0),
    ]
    _setup(monkeypatch, tmp_path, rows)
    SNM.main()
    capsys.readouterr()

    out = pd.read_csv(SNM.OUT)
    row = out[out.series == "X"].iloc[0]
    assert row["ym"] == "2020-04"
    assert row["sae"] == pytest.approx(8.0), (
        f"expected the MIN-over-models value 8.0 (the 'lo' model), got {row['sae']!r} -- "
        f"difficulty must be the best any model achieved, not the worst"
    )


# --- the file's own headline claim: monthly resolution reveals a cliff annual aggregation hides --

def test_shape_not_magnitude_annual_aggregation_dilutes_the_monthly_cliff(monkeypatch, tmp_path, capsys):
    """Single series, 24 months (2019 all normal, 2020 all normal except April=25x). Hand-derived:
      median (23 normal months @1.0, one @25.0) = 1.0 -> monthly ratio = 25.0/1.0 = 25.0
      ann_2019 = 1.0; ann_2020 = (11*1 + 25)/12 = 3.0; annual median of 2 years = (1.0+3.0)/2 = 2.0
      -> annual ratio = 3.0/2.0 = 1.5
    This is the exact MONTHLY->ANNUAL dilution the shape.annual_median_x_matched emit names as the
    exhibit's claim -- locked in here with an independently hand-derived numeric example (25x
    monthly collapses to 1.5x annual), not merely 'ratio > 1' or 'runs without crashing'."""
    months = pd.date_range("2019-01-01", "2020-12-01", freq="MS")
    vals = [1.0] * len(months)
    vals[list(months).index(pd.Timestamp("2020-04-01"))] = 25.0
    rows = _rows_for_series("Y", months, vals)
    frag_dir = _setup(monkeypatch, tmp_path, rows)

    SNM.main()
    capsys.readouterr()
    doc = _canonical(frag_dir)

    assert doc["shape.monthly_median_x"]["value"] == pytest.approx(25.0)
    assert doc["shape.annual_median_x"]["value"] == pytest.approx(1.5)
    monthly_x = doc["shape.monthly_median_x"]["value"]
    annual_x = doc["shape.annual_median_x"]["value"]
    assert annual_x < monthly_x, "annual aggregation must dilute (not amplify) the monthly cliff"

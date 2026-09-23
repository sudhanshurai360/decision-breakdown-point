"""Mutation-verified tests closing a survivor named M9 in this file only:
shape_not_magnitude.py's `ann_ratio_matched = ann_ratio.reindex(ratio.index).dropna()` replaced by
`ann_ratio_matched = ann_ratio` (matched := unmatched). It passes the 3 other tests, whose
fixtures only contain series that are either in BOTH samples or in NEITHER (S3 has no 2020 data at
all). The real panel's BOPGEXP/BOPGIMP case -- 2020 data present but NO 2020-04 evaluation month, so
the series enters the ANNUAL ratio but not the MONTHLY one -- was never modelled. This matters:
those two series sit below the matched median, so the matched/unmatched annual
figures differ (2.5x vs 2.2x) and the paper's dilution sentence depends on which one is quoted.

Same monkeypatch scaffold as tests/pipeline/test_shape_not_magnitude_core.py (PRED/OUT/HERE/ROOT_DIR
and canonical_emit's _PENDING/_FRAGDIR/_ROOT redirected to tmp_path); relies on tests/conftest.py
for sys.path. This test was verified to FAIL on mutant M9 and PASS on the real module.
"""
import json

import pandas as pd
import pytest

import canonical_emit as CE  # noqa: E402
import shape_not_magnitude as SNM  # noqa: E402


def _rows(series, months, values):
    rows = []
    for m, v in zip(months, values):
        rows.append(dict(series=series, model="real", target=m, sae=float(v)))
        rows.append(dict(series=series, model="decoy", target=m, sae=float(v) + 1000.0))
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
    monkeypatch.setattr(CE, "_ROOT", fake_root)
    return frag_dir


def test_annual_median_x_matched_restricts_to_the_monthly_sample(monkeypatch, tmp_path, capsys):
    """kills: M9 `ann_ratio_matched = ann_ratio.reindex(ratio.index).dropna()` -> `= ann_ratio`.
    S1: 2019-01..2020-12, all 1.0 except 2020-04 = 12.0.
        monthly ratio = 12/1 = 12;  annual: 2019 = 1, 2020 = 23/12 -> ratio (23/12)/((1+23/12)/2) = 46/35.
    S4: 2019-01..2020-12 WITHOUT a 2020-04 row (the BOPGEXP/BOPGIMP shape), all 1.0 except 2020-05 = 8.0.
        no monthly ratio (dropna);  annual: 2019 = 1, 2020 = (10*1+8)/11 = 18/11 -> ratio (18/11)/((1+18/11)/2) = 36/29.
    => monthly sample = {S1} (n=1); annual unmatched = {S1, S4} (n=2, median = (46/35+36/29)/2);
       annual MATCHED = {S1} (n=1, value 46/35). Under M9 the matched key silently reports n=2 and
       the unmatched median."""
    months = pd.date_range("2019-01-01", "2020-12-01", freq="MS")
    s1_vals = [1.0] * len(months)
    s1_vals[list(months).index(pd.Timestamp("2020-04-01"))] = 12.0
    s4_months = [m for m in months if m != pd.Timestamp("2020-04-01")]
    s4_vals = [1.0] * len(s4_months)
    s4_vals[s4_months.index(pd.Timestamp("2020-05-01"))] = 8.0
    frag_dir = _setup(monkeypatch, tmp_path, _rows("S1", months, s1_vals) + _rows("S4", s4_months, s4_vals))

    SNM.main()
    capsys.readouterr()
    doc = json.loads((frag_dir / "shape_not_magnitude.json").read_text())

    s1_ann = (23 / 12) / ((1 + 23 / 12) / 2)
    s4_ann = (18 / 11) / ((1 + 18 / 11) / 2)
    assert s1_ann == pytest.approx(46 / 35) and s4_ann == pytest.approx(36 / 29)

    assert doc["shape.monthly_median_x"]["n_series"] == 1
    assert doc["shape.monthly_median_x"]["value"] == pytest.approx(12.0)

    assert doc["shape.annual_median_x"]["n_series"] == 2
    assert doc["shape.annual_median_x"]["value"] == pytest.approx((s1_ann + s4_ann) / 2)

    assert doc["shape.annual_median_x_matched"]["n_series"] == 1, (
        "the matched annual figure must be restricted to the series that have a 2020-04 month")
    assert doc["shape.annual_median_x_matched"]["value"] == pytest.approx(s1_ann)
    assert doc["shape.annual_median_x_matched"]["value"] != pytest.approx(doc["shape.annual_median_x"]["value"])

    # the S4 hardest month is 2020-05 (8.0), counted in hardest_2020 but not hardest_april2020
    assert doc["shape.hardest_april2020"]["value"] == 1 and doc["shape.hardest_april2020"]["of"] == 2
    assert doc["shape.hardest_2020"]["value"] == 2

"""The FRED-MD population-scale
"predictive law" block (F-test, 5,000-draw permutation test, Bonferroni correction) produces 4
canonical keys -- fredmd.law_max_abs_rho, .law_r2_gap_t, .law_r2_mcs_abs, .law_r2_mcs_sq -- but had
no pytest regression test, breaking this project's own established convention
(test_canonical_new_checks.py's pattern for other recent additions). canonical.py's _fredmd_law()
is a hand-copied re-derivation of fred_md_sweep.py's OLS/Spearman logic; this locks in that it
still agrees with the live canonical.json at full precision, so a future edit to either copy
(predictor order, standardization, degrees-of-freedom, the Bonferroni multiplier) can't silently
diverge with nothing but a manual `canonical.py --check` run to catch it.
"""
import json

import pytest

import canonical as C

pytestmark = pytest.mark.skipif(not C.OUT.exists(), reason="paper/canonical.json not available")

FREDMD_LAW_KEYS = [
    "fredmd.law_max_abs_rho",
    "fredmd.law_r2_gap_t",
    "fredmd.law_r2_mcs_abs",
    "fredmd.law_r2_mcs_sq",
]


@pytest.mark.parametrize("key", FREDMD_LAW_KEYS)
def test_fredmd_law_key_registered_and_matches(key):
    assert key in C.CHECKS, f"{key} is not registered in canonical.py's CHECKS"
    doc = json.loads(C.OUT.read_text())
    assert key in doc, f"{key} is not present in the live canonical.json"
    assert C.CHECKS[key]() == doc[key]["value"], (
        f"{key}: canonical.py's independent re-derivation no longer matches the emitted value "
        f"at full precision -- either fred_md_sweep.py's computation or canonical.py's own "
        f"_fredmd_law() re-derivation has drifted."
    )


def test_fredmd_law_r2_values_are_nonnegative_fractions():
    """Sanity bound on the three R^2-family keys (gap_t, mcs_abs, mcs_sq) -- an R^2 must be a
    fraction in [0, 1]; a value outside that range would indicate a formula bug (e.g. using
    correlation instead of R^2, or a sign error) rather than a genuine result."""
    doc = json.loads(C.OUT.read_text())
    for key in ("fredmd.law_r2_gap_t", "fredmd.law_r2_mcs_abs", "fredmd.law_r2_mcs_sq"):
        v = doc[key]["value"]
        assert 0.0 <= v <= 1.0, f"{key}={v} is outside a valid R^2 range [0, 1]"


def test_fredmd_law_max_abs_rho_is_a_valid_correlation():
    doc = json.loads(C.OUT.read_text())
    v = doc["fredmd.law_max_abs_rho"]["value"]
    assert 0.0 <= v <= 1.0, f"fredmd.law_max_abs_rho={v} is outside a valid |correlation| range [0, 1]"

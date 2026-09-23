"""A threshold/parameter-duplication finding, pinned rather than fixed.

walkforward.py's per-origin data-availability gate (`if len(p) < 2 * m + 8: continue`) and
models_freq.py's SARIMA/DHR classes' own internal length gates (SARIMA: `if len(y) < 2 * self.m + 8:
return _last(y)`; DHR: the same test written against a pre-bound `n`, `if n < 2 * self.m + 8:`)
independently hand-type the identical `2*m+8` formula -- "2 full seasonal cycles
plus an 8-period buffer" -- at 3 sites across 2 files, with no shared named constant and no
cross-reference comment linking them. SARIMA/DHR's internal gates exist as their own safety net
(in case a caller bypasses walkforward.py's outer filter, e.g. a rescoring path), so if
walkforward.py's threshold is ever retuned without the same edit landing in both model classes,
SARIMA/DHR would silently start attempting fits on windows walkforward no longer intends to allow,
or keep falling back to naive on windows walkforward now allows.

No shared-import target was hoisted (walkforward.py and models_freq.py do not currently import each
other in either direction, and introducing a new cross-file coupling to remove the duplication was
judged higher-risk than leaving the duplication in place and pinning it) -- pinned instead,
mirroring the same approach in
test_classical_forecaster_formulas_agree.py: these tests assert the 3 sites currently agree, so a
future retune of any one of them fails a test here instead of silently drifting.
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
PIPELINE = ROOT / "code" / "pipeline"
sys.path.insert(0, str(PIPELINE))


def _site_thresholds(m: int) -> dict:
    """Reads each of the 3 production sites' OWN threshold expression out of its own source line and
    evaluates it at `m`. Nothing here re-types the formula: the coefficient and the buffer are the
    integers the production files actually carry, so retuning any one site changes what this returns."""
    wf_src = (PIPELINE / "walkforward.py").read_text(encoding="utf-8")
    mf_src = (PIPELINE / "models_freq.py").read_text(encoding="utf-8")

    wf = re.search(r"len\(p\)\s*<\s*(\d+)\s*\*\s*m\s*\+\s*(\d+)", wf_src)
    assert wf is not None, "walkforward.py's per-origin length gate no longer matches `N * m + N`"
    mf = re.findall(r"<\s*(\d+)\s*\*\s*self\.m\s*\+\s*(\d+)", mf_src)
    assert len(mf) == 2, (
        f"expected exactly 2 sites (SARIMA + DHR) gating on `N * self.m + N` in models_freq.py, "
        f"found {len(mf)}"
    )

    out = {"walkforward": int(wf.group(1)) * m + int(wf.group(2))}
    for i, (coef, buf) in enumerate(mf):
        out[f"models_freq[{i}]"] = int(coef) * m + int(buf)
    return out


def test_walkforward_and_models_freq_source_all_use_the_identical_formula():
    """Static regression: greps each site's actual source line and confirms the literal formula
    text is `2 * m + 8` / `2 * self.m + 8` at all 3 known sites -- a change to the formula's SHAPE
    (not just its m) at any one site fails this test."""
    wf_src = (PIPELINE / "walkforward.py").read_text(encoding="utf-8")
    mf_src = (PIPELINE / "models_freq.py").read_text(encoding="utf-8")

    assert re.search(r"len\(p\)\s*<\s*2\s*\*\s*m\s*\+\s*8", wf_src), (
        "walkforward.py's per-origin length gate formula has changed shape"
    )
    sarima_dhr_hits = re.findall(r"<\s*2\s*\*\s*self\.m\s*\+\s*8", mf_src)
    assert len(sarima_dhr_hits) == 2, (
        f"expected exactly 2 sites (SARIMA + DHR) using `2 * self.m + 8` in models_freq.py, "
        f"found {len(sarima_dhr_hits)} -- update this test if a site was intentionally added/removed"
    )


def test_formula_agrees_numerically_for_monthly_and_weekly_periods():
    """Behavioral regression: the 3 production sites' own threshold expressions, evaluated at the
    two real m values this codebase uses (m=12 monthly, m=52 weekly), must all produce the SAME
    number -- and that number must still be the documented "2 seasonal cycles + 8" (32 and 112). A
    retune of any single site, in coefficient or in buffer, breaks one of these two assertions."""
    for m, expected in ((12, 32), (52, 112)):
        sites = _site_thresholds(m)
        assert len(set(sites.values())) == 1, (
            f"m={m}: the 3 sites no longer agree on the minimum panel length: {sites}"
        )
        assert set(sites.values()) == {expected}, (
            f"m={m}: sites agree on {sorted(set(sites.values()))} but the documented "
            f"'2 seasonal cycles + 8' threshold is {expected} -- update this test and the "
            f"module docstring together if the retune is intentional"
        )


def test_sarima_and_dhr_forecast_actually_use_this_threshold_live():
    """End-to-end: confirms both classes' `forecast()` really does switch from last-value fallback
    to a real fit at EXACTLY this boundary, not just that the source text matches -- the case this
    test exists for is a formula text that still reads `2 * self.m + 8` while the live gate has
    drifted by one (a `<=`/`<` flip, or an `n` that is no longer `len(y)`).

    The boundary must be probed AT the threshold. The previous version of this test probed only
    `threshold - 1` and `threshold + 20`, which straddle the gate so widely that shifting it by a
    month changed nothing it looked at: mutating models_freq.py's DHR `n = len(y)` to
    `n = len(y) - 1` left all three tests in this file green. The three probe lengths below are
    threshold-1 (must fall back), threshold (must NOT) and threshold+1 (must NOT), so a one-step
    shift in either direction fails.

    The fixture is a strong linear trend plus a clean seasonal cycle, so a real fit's one-step
    forecast provably differs from the last observation -- which is what makes "did not take the
    last-value fallback" a decidable assertion rather than a coincidence."""
    import numpy as np
    import warnings
    warnings.filterwarnings("ignore")
    from models_freq import SARIMA, DHR

    m = 12
    threshold = 2 * m + 8

    def _series(n):
        t = np.arange(n)
        return (100.0 + 0.8 * t + 6.0 * np.sin(2 * np.pi * t / m)
                + np.random.default_rng(0).normal(0, 0.3, n))

    for cls in (SARIMA, DHR):
        below = _series(threshold - 1)
        assert cls(m=m).forecast(below, h=1) == float(below[-1]), (
            f"{cls.__name__}: n={threshold - 1} is BELOW the {threshold}-length gate and must fall "
            f"back to last-value exactly"
        )
        for n in (threshold, threshold + 1):
            y = _series(n)
            fc = cls(m=m).forecast(y, h=1)
            assert np.isfinite(fc)
            assert fc != float(y[-1]), (
                f"{cls.__name__}: n={n} is AT or ABOVE the {threshold}-length gate, so forecast() "
                f"must attempt a real fit, but it returned the last observation exactly -- the "
                f"live length gate has drifted off the `2 * m + 8` the source text still claims"
            )

"""Regression test: run_multiseries.py's `MODELS = os.environ.get("MS_MODELS", ...).split(",")` had no `.strip()` per
token -- a naturally-typed `MS_MODELS="sarima, dhr"` (space after comma) yielded
["sarima", " dhr"], and " dhr" then hit models_freq.py's REGISTRY lookup and raised a confusing
`KeyError: ' dhr'` instead of the model name a reader would recognize as correct. An empty
MS_MODELS="" yielded [""], same confusing KeyError.

Fixed by stripping each token and dropping empty ones.
"""
import importlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "code" / "pipeline"))

import run_multiseries as RM  # noqa: E402


def test_ms_models_strips_whitespace_around_commas(monkeypatch):
    monkeypatch.setenv("MS_MODELS", "sarima, dhr")
    importlib.reload(RM)
    try:
        assert RM.MODELS == ["sarima", "dhr"]
    finally:
        monkeypatch.delenv("MS_MODELS", raising=False)
        importlib.reload(RM)


def test_ms_models_empty_string_produces_an_empty_list_not_a_list_with_one_blank_token(monkeypatch):
    monkeypatch.setenv("MS_MODELS", "")
    importlib.reload(RM)
    try:
        assert RM.MODELS == [], "an empty MS_MODELS should yield no models, not ['']"
    finally:
        monkeypatch.delenv("MS_MODELS", raising=False)
        importlib.reload(RM)


def test_ms_models_default_is_unaffected(monkeypatch):
    monkeypatch.delenv("MS_MODELS", raising=False)
    importlib.reload(RM)
    assert RM.MODELS == ["seasonal_naive", "ar1", "ets", "theta", "sarima", "dhr"]

"""
fetch_alfred.py — pull TRUE point-in-time ALFRED vintages for the additional national monthly macro series,
mirroring the UI-claims data lineage (fredapi.get_series_all_releases, all releases = every vintage).
"""
import sys, json, hashlib, datetime as dt
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT  = HERE.parent.parent                 # <repo root>/code/<subdir>/ -> <repo root>
MULTI = ROOT / "multiseries"               # data + results live HERE, not under code/

RAW = MULTI / "data" / "raw"
FRED_ENV_REPO = str(Path(__file__).resolve().parent.parent.parent / ".env")

# series_id -> human label + the dominant shock era (for the write-up; NOT used in any computation)
SERIES = {
    "PAYEMS":  ("Nonfarm payrolls",        "2008 & 2020"),
    "RSAFS":   ("Retail & food services",  "2020"),
    "HOUST":   ("Housing starts",          "2008"),
    "DGORDER": ("Durable goods new orders", "2008"),   # industrial 2008-collapse, natural $ units, vintages from 1999
    "INDPRO":  ("Industrial production",   "2008"),
    "TOTALSA": ("Total vehicle sales",     "2008 & 2020"),
}


def _read_env_key(path: str) -> str | None:
    if not Path(path).exists():
        return None
    for l in open(path):
        if l.startswith("FRED_API_KEY"):
            return l.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def _fred_key() -> str:
    import os
    key = os.environ.get("FRED_API_KEY") or _read_env_key(FRED_ENV_REPO)
    if key is None:
        raise RuntimeError(
            "No FRED API key found. Set the FRED_API_KEY environment variable, or write "
            f"FRED_API_KEY=... to a .env file at the repo root ({FRED_ENV_REPO}). "
            "A free key is available at https://fred.stlouisfed.org/docs/api/api_key.html."
        )
    return key


def fetch_one(series_id: str, fred) -> pd.DataFrame:
    """All-releases vintage frame for one series. fredapi returns columns realtime_start, date, value."""
    raw = fred.get_series_all_releases(series_id)
    df = raw.rename(columns=str.lower)[["realtime_start", "date", "value"]].copy()
    df["realtime_start"] = pd.to_datetime(df["realtime_start"])
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["series_id"] = series_id
    return df[["series_id", "date", "realtime_start", "value"]].sort_values(
        ["date", "realtime_start"]).reset_index(drop=True)


def _sha16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def load_cached_or_fetch(sid, out, fetch_fn, force):
    """Return (df, fetched): the cached parquet at `out` if it exists and `force` is not set,
    else the result of `fetch_fn()`."""
    if out.exists() and not force:
        df = pd.read_parquet(out)
        df["realtime_start"] = pd.to_datetime(df["realtime_start"])
        df["date"] = pd.to_datetime(df["date"])
        return df, False
    return fetch_fn(), True


def main():
    from fredapi import Fred
    import fredapi as _fa
    force = "--force" in sys.argv
    positional = [a for a in sys.argv[1:] if not a.startswith("--")]
    if positional:
        unmatched = [a for a in positional if a not in SERIES]
        if unmatched:
            print(f"  UNKNOWN SERIES ARGUMENT(S): {unmatched} -- refusing to silently fall back to "
                  f"ALL series (a typo here could otherwise re-fetch and overwrite every pinned "
                  f"ALFRED parquet, not just the intended one). Known series: {sorted(SERIES)}.",
                  file=sys.stderr)
            sys.exit(2)
        wanted = positional
    else:
        wanted = list(SERIES)
    fred = Fred(api_key=_fred_key())
    RAW.mkdir(parents=True, exist_ok=True)

    lineage_path = RAW / "lineage.json"
    lineage = json.loads(lineage_path.read_text()) if lineage_path.exists() else {
        "source": "FRED/ALFRED via fredapi.get_series_all_releases",
        "fredapi_version": _fa.__version__, "pandas_version": pd.__version__,
        "per_series": {},
    }
    any_fetched = False  # stamp pulled_at only when a file is actually (re)fetched, not on a no-op run

    for sid in wanted:
        label, shock = SERIES[sid]
        out = RAW / f"{sid}_alfred_vintages.parquet"
        print(f"\n=== {sid} ({label}) ===")
        try:
            df, fetched = load_cached_or_fetch(sid, out, lambda: fetch_one(sid, fred), force)
        except Exception as e:
            print(f"  FETCH FAILED for {sid}: {type(e).__name__}: {e} -- skipping, re-run to retry "
                  f"(already-frozen series are unaffected)")
            continue
        if not fetched:
            print(f"  already frozen at {out.name} -- pass --force to deliberately re-fetch and overwrite")
        else:
            df.to_parquet(out, index=False); any_fetched = True
        # vintage coverage: how many distinct release dates straddle the shocks (sanity that it's REAL real-time)
        rs = df["realtime_start"]
        _d = lambda ts: None if pd.isna(ts) else str(ts.date())
        lineage["per_series"][sid] = {
            "label": label, "dominant_shock": shock, "file": out.name,
            "n_rows": int(len(df)), "n_vintages": int(rs.nunique()),
            "obs_min": _d(df["date"].min()), "obs_max": _d(df["date"].max()),
            "vintage_min": _d(rs.min()), "vintage_max": _d(rs.max()),
            "vintages_pre_2008": int(rs[rs < "2008-01-01"].nunique()),
            "vintages_pre_2020": int(rs[rs < "2020-01-01"].nunique()),
            "n_nan": int(df["value"].isna().sum()),
            "sha256_16": _sha16(out),
        }
        s = lineage["per_series"][sid]
        print(f"  rows={s['n_rows']}  vintages={s['n_vintages']}  obs {s['obs_min']}..{s['obs_max']}")
        print(f"  vintage dates {s['vintage_min']}..{s['vintage_max']}  (pre-2020 vintages: {s['vintages_pre_2020']})")
        print(f"  nan={s['n_nan']}  sha={s['sha256_16']}  -> {out.name}")

    if any_fetched or "pulled_at" not in lineage:
        lineage["pulled_at"] = dt.datetime.now().isoformat(timespec="seconds")
    lineage_path.write_text(json.dumps(lineage, indent=2))
    print(f"\nwrote {lineage_path}")


if __name__ == "__main__":
    main()

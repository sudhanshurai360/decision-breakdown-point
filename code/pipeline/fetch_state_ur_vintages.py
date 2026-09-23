#!/usr/bin/env python3
"""fetch_state_ur_vintages.py — rebuild multiseries/data/raw/state_ur_alfred_vintages.parquet.
"""
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
MULTI = ROOT / "multiseries"
RAW = MULTI / "data" / "raw"
PARTS = RAW / "_state_ur_parts"
OUT = RAW / "state_ur_alfred_vintages.parquet"

AS_OF_DEFAULT = "2026-07-01"
sys.path.insert(0, str(HERE))


def states() -> list[str]:
    """The 51 codes, read from the surviving revised panel so the two panels cannot drift apart."""
    cache = MULTI / "data" / "state_ur.parquet"
    if not cache.exists():
        raise SystemExit(f"missing {cache} -- the revised state panel defines which states to fetch")
    return [str(c) for c in pd.read_parquet(cache).columns]


def main() -> int:
    import fetch_alfred as FA
    from fredapi import Fred

    known_bare = {"--force"}
    unknown = [a for a in sys.argv[1:] if a not in known_bare and not a.startswith("--as-of=")]
    if unknown:
        print(f"  UNKNOWN ARGUMENT(S): {unknown} -- refusing to silently ignore a possible typo. "
              f"Recognized: --force, --as-of=YYYY-MM-DD (or --as-of=none).", file=sys.stderr)
        return 2
    force = "--force" in sys.argv
    if OUT.exists() and not force:
        print(f"\n  {OUT.relative_to(ROOT)} already exists and --force was not passed -- "
              f"NOT overwriting the frozen merged panel. Pass --force to intentionally rebuild it.")
        return 0
    PARTS.mkdir(parents=True, exist_ok=True)
    fred = Fred(api_key=FA._fred_key())
    codes = states()
    print(f"=== STATE UR ALFRED VINTAGES: {len(codes)} states ===")

    frames, failed = [], []
    for i, st in enumerate(codes, 1):
        part = PARTS / f"{st}.parquet"
        if part.exists() and not force:
            frames.append(pd.read_parquet(part))
            print(f"  [{i:>2}/{len(codes)}] {st}  cached", flush=True)
            continue
        sid = f"{st}UR"
        try:
            # Reuse fetch_alfred's own all-releases parser rather than re-deriving it: it already
            # handles fredapi's column casing and the realtime_start/value coercions.
            df = FA.fetch_one(sid, fred).rename(columns={"series_id": "state"})
            df["state"] = st
            df.to_parquet(part, index=False)
            frames.append(df)
            _d = lambda ts: "NaT (empty fetch)" if pd.isna(ts) else str(ts.date())
            print(f"  [{i:>2}/{len(codes)}] {st}  {len(df):>7,} rows  "
                  f"vintages {_d(df.realtime_start.min())}..{_d(df.realtime_start.max())}", flush=True)
        except Exception as e:
            failed.append((st, f"{type(e).__name__}: {e}"))
            print(f"  [{i:>2}/{len(codes)}] {st}  FAILED  {type(e).__name__}: {e}", flush=True)
        time.sleep(0.4)                      # courtesy pacing; ALFRED rate-limits all-releases pulls

    if failed:
        print(f"\n  {len(failed)} state(s) failed: {', '.join(s for s, _ in failed)}")
        print("  NOT writing the merged parquet. Re-run to retry only the failures (parts are cached).")
        return 1

    out = pd.concat(frames, ignore_index=True).sort_values(["state", "date", "realtime_start"])
    out = out[["state", "date", "realtime_start", "value"]].reset_index(drop=True)
    as_of = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--as-of=")), AS_OF_DEFAULT)
    if as_of.lower() != "none":
        before = len(out)
        before_states = set(out.state.unique())
        out = out[pd.to_datetime(out.realtime_start) <= pd.Timestamp(as_of)].reset_index(drop=True)
        print(f"  as-of {as_of}: dropped {before - len(out):,} vintage rows released after the "
              f"surviving panels' own fetch horizon")
        after_states = set(out.state.unique())
        dropped_states = sorted(before_states - after_states)
        if dropped_states:
            print(f"  FATAL: --as-of={as_of} drops {len(dropped_states)} state(s) entirely from "
                  f"the panel (earlier than their own ALFRED archive start): {dropped_states}. "
                  f"Refusing to write a silently-collapsed 'immutable' panel -- choose a later "
                  f"--as-of, or explicitly accept the narrower state set by handling this "
                  f"disclosure rather than writing over it.")
            return 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    print(f"\n  wrote {OUT.relative_to(ROOT)}")
    _d = lambda ts: "NaT (empty panel)" if pd.isna(ts) else str(ts.date())
    print(f"  {len(out):,} rows | {out.state.nunique()} states | "
          f"reference months {_d(out.date.min())}..{_d(out.date.max())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

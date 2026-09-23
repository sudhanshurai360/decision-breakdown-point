"""
fred_md_sweep.py — the PREDICTIVE-LAW test at POPULATION scale (the last real shot at 'bigger').
"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
ROOT  = HERE.parent.parent                 # <repo root>/code/<subdir>/ -> <repo root>
MULTI = ROOT / "multiseries"               # data + results live HERE, not under code/
INSTR = ROOT / "code" / "instrument"       # shared estimator

sys.path.insert(0, str(HERE)); sys.path.insert(0, str(INSTR))
import m_audit as MA                 # fast h-step forecasters
import mcs_test as MCS               # mcs()
import run_multiseries as RM         # for its shared _atomic_to_csv() helper
import fragility as F                # surprise_concentration
import walkforward as WF             # _assert_no_calendar_gaps -- same guard as the curated-12 pipeline
from canonical_emit import emit, flush
DATA = MULTI / "data" / "fredmd"; RES = MULTI / "results"
MODELS = MA.EB_MODELS
M = 12; EVAL_START = 2000

FREDMD_CODES = [
    # output/income
    "RPI","INDPRO","IPFINAL","IPCONGD","IPDCONGD","IPNCONGD","IPBUSEQ","IPMAT","IPMANSICS","CUMFNS","TCU",
    # labor
    "UNRATE","UEMPMEAN","CLF16OV","CE16OV","PAYEMS","USGOOD","USCONS","MANEMP","DMANEMP","NDMANEMP","SRVPRD",
    "USTPU","USWTRADE","USTRADE","USFIRE","USGOVT","AWHMAN","AWOTMAN","UNEMPLOY",
    # housing
    "HOUST","HOUSTNE","HOUSTMW","HOUSTS","HOUSTW","PERMIT",
    "DPCERA3M086SBEA","BUSINV","RSAFS","PCE","PI","DGORDER","AMTMNO",
    # money / credit
    "M1SL","M2SL","M2REAL","BOGMBASE","TOTRESNS","NONBORRES","BUSLOANS","REALLN","NONREVSL","INVEST","TOTALSL",
    "FEDFUNDS","TB3MS","TB6MS","GS1","GS5","GS10",
    # prices
    "WPSFD49207","PPICMM","CPIAUCSL","CPIAPPSL","CPITRNSL","CPIMEDSL","CPIULFSL","PCEPI",
    "DDURRG3M086SBEA","DNDGRG3M086SBEA","DSERRG3M086SBEA","WPSID61",
    # trade
    "EXPGS","IMPGS","BOPGEXP","BOPGIMP",
]

FREDMD_TCODE = {
    "RPI": 5, "INDPRO": 5, "IPFINAL": 5, "IPCONGD": 5, "IPDCONGD": 5, "IPNCONGD": 5, "IPBUSEQ": 5,
    "IPMAT": 5, "IPMANSICS": 5, "CUMFNS": 2,
    "UNRATE": 2, "UEMPMEAN": 2, "CLF16OV": 5, "CE16OV": 5, "PAYEMS": 5, "USGOOD": 5, "USCONS": 5,
    "MANEMP": 5, "DMANEMP": 5, "NDMANEMP": 5, "SRVPRD": 5, "USTPU": 5, "USWTRADE": 5, "USTRADE": 5,
    "USFIRE": 5, "USGOVT": 5, "AWHMAN": 1, "AWOTMAN": 2,
    "HOUST": 4, "HOUSTNE": 4, "HOUSTMW": 4, "HOUSTS": 4, "HOUSTW": 4, "PERMIT": 4,
    "DPCERA3M086SBEA": 5, "BUSINV": 5, "RSAFS": 5,
    "M1SL": 6, "M2SL": 6, "M2REAL": 5, "TOTRESNS": 6, "NONBORRES": 7, "BUSLOANS": 6, "REALLN": 6,
    "NONREVSL": 6, "INVEST": 6,
    "FEDFUNDS": 2, "TB3MS": 2, "TB6MS": 2, "GS1": 2, "GS5": 2, "GS10": 2,
    "WPSFD49207": 6, "PPICMM": 6, "CPIAUCSL": 6, "CPIAPPSL": 6, "CPITRNSL": 6, "CPIMEDSL": 6,
    "CPIULFSL": 6, "PCEPI": 6, "DDURRG3M086SBEA": 6, "DNDGRG3M086SBEA": 6, "DSERRG3M086SBEA": 6,
    "WPSID61": 6,
}


def apply_tcode(y: np.ndarray, code: int) -> np.ndarray:
    """McCracken & Ng (2016) FRED-MD transformation codes, applied to a LEVEL series y."""
    if code == 1: return y
    if code == 2: return np.diff(y)
    if code == 3: return np.diff(y, n=2)
    if code == 4: return np.log(y)
    if code == 5: return np.diff(np.log(y))
    if code == 6: return np.diff(np.log(y), n=2)
    if code == 7: return np.diff(y[1:] / y[:-1] - 1.0)
    raise ValueError(f"unknown FRED-MD tcode {code}")


TCODE_OUTLIER_RATIO_THRESHOLD = 500


def tcode_outlier_ratio(y_t: np.ndarray) -> float:
    """max(|y_t|) / median(|y_t|): flags a FINITE but extreme relative-magnitude value in a
    tcode-transformed series -- the failure mode np.isfinite() cannot see."""
    return float(np.max(np.abs(y_t)) / (np.median(np.abs(y_t)) + 1e-9))


def load_fredmd():
    DATA.mkdir(parents=True, exist_ok=True)
    cache = DATA / "fredmd_api.parquet"
    fp_file = DATA / "fredmd_api_codes.txt"
    fp = "\n".join(FREDMD_CODES)
    if cache.exists() and fp_file.exists() and fp_file.read_text() == fp:
        return pd.read_parquet(cache)
    import fetch_alfred as FA
    from fredapi import Fred
    fred = Fred(api_key=FA._fred_key())
    cols = {}
    for c in FREDMD_CODES:
        try:
            s = fred.get_series(c)
            if s is not None and len(s.dropna()) > 200:
                cols[c] = s
        except Exception as e:
            print(f"  [skip] {c}: {e}", flush=True)
    df = pd.DataFrame(cols); df.index = pd.to_datetime(df.index)
    df = df[df.index >= "1959-01-01"].sort_index()
    df.to_parquet(cache); fp_file.write_text(fp)
    print(f"  fetched {df.shape[1]} series via FRED API")
    return df


def series_losses(y, dates):
    """annual-refit, 12-month-ahead each eval year -> per-year MASE/RMSSE per model."""
    yr = dates.year
    eval_years = [Y for Y in range(EVAL_START, int(yr.max()) + 1) if (yr < Y).sum() >= 60 and (yr == Y).sum() >= 6]
    if len(eval_years) < 8: return None
    models = list(MODELS); La, Ls = [], []
    for Y in eval_years:
        tr = y[yr < Y]; te = y[yr == Y]; h = len(te)
        nc = WF._not_covid(pd.Series(dates[yr < Y])) if (yr < Y).sum() else np.ones(len(tr), dtype=bool)
        d12_full = tr[M:] - tr[:-M]
        dkeep = nc[M:] & nc[:-M]
        if dkeep.any():
            d12 = np.abs(d12_full[dkeep]); sq_src = d12_full[dkeep]
        else:
            d12 = np.abs(d12_full); sq_src = d12_full   # fall back to unmasked if no non-COVID pair remains
        tr_masked = tr[nc] if nc.any() else tr
        scale_a = d12.mean() if len(d12) and d12.mean() > 0 else (np.abs(tr_masked).mean() + 1e-9)
        scale_s = (sq_src ** 2).mean() if len(sq_src) else scale_a ** 2
        if scale_s <= 0: scale_s = scale_a ** 2
        ra, rs = {}, {}
        for nm, fn in MODELS.items():
            fc = MA.safe_forecast(fn, nm, tr, M, h, context=f"eval_year={Y}")
            e = te - np.asarray(fc, float)[:h]
            ra[nm] = np.abs(e).mean() / scale_a
            rs[nm] = np.sqrt((e ** 2).mean() / scale_s)
        La.append([ra[m] for m in models]); Ls.append([rs[m] for m in models])
    return np.array(La), np.array(Ls), models, eval_years


def load_series_gap_safe(df, col):
    """Interior-gap-safe extraction of one FRED-MD series's real-valued history from the wide
    load_fredmd() panel."""
    s = df[col].interpolate(limit=1, limit_area="inside").dropna()
    if len(s) < 180:
        print(f"  [skip] {col}: only {len(s)} valid months after gap-fill (<180 required)", flush=True)
        return None
    try:
        WF._assert_no_calendar_gaps(s.index, col)
    except ValueError as e:
        print(f"  [skip] {col}: interpolation did not fully bridge the gap: {e}", flush=True)
        return None
    return s


def main():
    df = load_fredmd(); print(f"FRED-MD: {df.shape[1]} series x {df.shape[0]} months")
    rows = []
    n_tcode_outlier_skipped = 0
    for k, col in enumerate(df.columns):
        s = load_series_gap_safe(df, col)
        if s is None:
            continue
        y = s.to_numpy(float); res = series_losses(y, s.index)
        if res is None:
            print(f"  [skip] {col}: series_losses() returned no result", flush=True)
            continue
        La, Ls, models, yrs = res
        if not np.all(np.isfinite(La)):
            print(f"  [skip] {col}: non-finite loss in the levels arm (La)", flush=True)
            continue
        Ld = {m: La[:, j] for j, m in enumerate(models)}
        sc = F.surprise_concentration(Ld)
        # DATA-SIDE shock proxies (computed from the raw series, NOT the model losses -> not circular):
        dy = np.diff(y); dyc = dy - dy.mean()
        abrupt = float(np.nanmax(np.abs(np.diff(y, 2))) / (np.nanmedian(np.abs(dy)) + 1e-9))   # break-abruptness
        kurt = float((dyc ** 4).mean() / ((dyc ** 2).mean() ** 2 + 1e-12) - 3) if len(dy) > 3 else np.nan  # tail heaviness
        vol = float(np.std(dy) / (np.median(np.abs(dy)) + 1e-9))                                # dispersion
        surv_a, _ = MCS.mcs(La, alpha=0.10, seed=0); surv_s, _ = MCS.mcs(Ls, alpha=0.10, seed=0)
        # CONTINUOUS identification score: studentized best-vs-2nd-best gap. High = identified, ~0 = near-tie.
        ml = La.mean(0); order = np.argsort(ml); d = La[:, order[1]] - La[:, order[0]]
        gap_t = float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)) + 1e-12))
        MCS_abs_tcode = np.nan
        tcode = FREDMD_TCODE.get(col)
        if tcode is not None:
            try:
                y_t = apply_tcode(y, tcode)
                _outlier_ratio = tcode_outlier_ratio(y_t)
                if _outlier_ratio > TCODE_OUTLIER_RATIO_THRESHOLD:
                    n_tcode_outlier_skipped += 1
                    print(f"  [tcode skip] {col} (tcode={tcode}): transformed series has a "
                          f"{_outlier_ratio:.0f}x relative-magnitude outlier (likely a "
                          f"near-zero-crossing artifact) -- excluded from the tcode-comparison arm",
                          flush=True)
                else:
                    dates_t = s.index[-len(y_t):]
                    res_t = series_losses(y_t, dates_t)
                    if res_t is not None:
                        La_t, _, _, _ = res_t
                        if np.all(np.isfinite(La_t)):
                            surv_t, _ = MCS.mcs(La_t, alpha=0.10, seed=0)
                            MCS_abs_tcode = len(surv_t)
            except Exception as e:
                print(f"  [tcode skip] {col} (tcode={tcode}): {type(e).__name__}: {e}", flush=True)
        rows.append({"series": col, "n_years": len(yrs), "abrupt": round(abrupt, 2), "kurt": round(kurt, 2),
                     "vol": round(vol, 2), "sci_CIRC": round(sc["sci"], 3), "gap_t": round(gap_t, 3),
                     "MCS_abs": len(surv_a), "MCS_sq": len(surv_s), "best": models[int(np.argmin(ml))],
                     "MCS_abs_tcode": MCS_abs_tcode})
        if (k + 1) % 20 == 0: print(f"  ...{k+1}/{df.shape[1]} processed, {len(rows)} usable", flush=True)
    r = pd.DataFrame(rows)
    _csv_out = RES / "fred_md_sweep.csv"
    RM._atomic_to_csv(r, _csv_out)
    n = len(r)
    print(f"\n=== FRED-MD SWEEP: {n} usable series, fast tier (4 models), Hansen MCS ===")
    print(f"  identified (|MCS_abs|=1): {int((r.MCS_abs==1).sum())}/{n}   near-tie (>1): {int((r.MCS_abs>1).sum())}/{n}   mean |MCS_abs| {r.MCS_abs.mean():.2f}")
    print(f"  |MCS_abs| distribution: {dict(r.MCS_abs.value_counts().sort_index())}")
    print(f"\n=== THE PREDICTIVE LAW (does DATA-SIDE shock-proneness predict (non)identifiability "
          f"across {n} series?) ===")
    print("  (gap_t: high=IDENTIFIED; MCS_abs/MCS_sq: high=NON-identified. sci_CIRC is circular -> ignore.)")
    rho_grid = {}
    for pred in ["abrupt", "kurt", "vol", "sci_CIRC"]:
        for dv in ["gap_t", "MCS_abs", "MCS_sq"]:
            rho = r[[pred, dv]].corr(method="spearman").iloc[0, 1]
            rho_grid[(pred, dv)] = rho
            tag = "  <-- LAW" if abs(rho) >= 0.3 else ""
            print(f"  Spearman({pred:8s}, {dv:8s}) = {rho:+.3f}{tag}")
    # JOINT regression: do the data-side predictors TOGETHER explain identifiability? (OLS R^2, standardized predictors)
    P = ["abrupt", "kurt", "vol"]
    X = r[P].fillna(r[P].median()); X = (X - X.mean()) / X.std().replace(0, 1)
    X = np.column_stack([np.ones(len(X)), X.to_numpy()])
    r2 = {}
    for dv in ["gap_t", "MCS_abs", "MCS_sq"]:
        y = r[dv].to_numpy(float)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None); yhat = X @ beta
        R2 = 1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        r2[dv] = R2
        print(f"  joint OLS  [abrupt,kurt,vol] -> {dv:8s} :  R^2 = {R2:.3f}")
    print(f"\n  (interpret: |rho|>=0.3 or joint R^2>=~0.10 = a real predictive law; near 0 = a WELL-POWERED null at n={n})")
    max_rho = max(abs(rho_grid[(p, dv)]) for p in P for dv in ["gap_t", "MCS_abs", "MCS_sq"])
    emit("fredmd.law_max_abs_rho", float(max_rho), n=int(n),
         spec="largest |Spearman rho| across the {abrupt,kurt,vol} x {gap_t,MCS_abs,MCS_sq} grid (9 "
              "pairs; the fourth candidate predictor, sci_CIRC, is excluded as circular -- it is "
              "computed from the model losses themselves, not from the data, so it cannot serve as "
              "a DATA-SIDE predictor of those same losses' identifiability) -- the population-scale "
              "companion to the n=12 curated-panel shock-proneness null of Section 4.3",
         cites=["S4.3: population-scale companion check, largest of nine Spearman rho values"],
         cross_check="fred_md_sweep.csv: max |spearman(pred, dv)| over abrupt/kurt/vol x gap_t/MCS_abs/MCS_sq")
    rng = np.random.default_rng(0)
    sig = {}
    for dv in ["gap_t", "MCS_abs", "MCS_sq"]:
        y = r[dv].to_numpy(float)
        F_stat = (r2[dv] / len(P)) / ((1 - r2[dv]) / (n - len(P) - 1))
        p_f = float(stats.f.sf(F_stat, len(P), n - len(P) - 1))
        perm_r2 = np.empty(5000)
        for i in range(5000):
            yp = rng.permutation(y)
            beta, *_ = np.linalg.lstsq(X, yp, rcond=None); yhat = X @ beta
            perm_r2[i] = 1 - ((yp - yhat) ** 2).sum() / ((yp - yp.mean()) ** 2).sum()
        p_perm = float((perm_r2 >= r2[dv]).mean())
        sig[dv] = (p_f, p_perm)
        print(f"  joint OLS  [abrupt,kurt,vol] -> {dv:8s} :  F-test p={p_f:.4f}  permutation p={p_perm:.4f}")
    n_dv = 3
    for dv, key in [("gap_t", "gap_t"), ("MCS_abs", "mcs_abs"), ("MCS_sq", "mcs_sq")]:
        p_f, p_perm = sig[dv]
        emit(f"fredmd.law_r2_{key}", float(r2[dv]), n=int(n),
             spec=f"joint OLS R^2, standardized [abrupt,kurt,vol] -> {dv}, population panel",
             p_value=p_f, permutation_p=p_perm, permutation_draws=5000,
             bonferroni_p_value=min(1.0, p_f * n_dv), bonferroni_permutation_p=min(1.0, p_perm * n_dv),
             bonferroni_m=n_dv,
             cites=[f"S4.3: joint OLS regression on {dv}"],
             cross_check=f"fred_md_sweep.csv: joint OLS R^2 (+ F-test and 5000-draw permutation p, "
                         f"Bonferroni-corrected x{n_dv} for the 3 DVs tested) on {dv}")
    n_nonid = int((r.MCS_abs > 1).sum())
    emit("fredmd.nonidentified", n_nonid, of=int(n),
         spec=f"series with |MCS_abs|>1 at alpha=0.10; annual-refit walk-forward from {EVAL_START}, "
              f"models={list(MODELS)}",
         cites=["Abstract: headline population non-identification count",
                "Introduction: headline population non-identification count",
                "S4.1 body: headline population count",
                "Conclusion: headline population non-identification count"],
         cross_check="fred_md_sweep.csv: (MCS_abs > 1).sum()")
    emit("fredmd.nonidentified_pct", round(100 * n_nonid / n),
         spec="fredmd.nonidentified as a share of fredmd.total, rounded to the nearest percent",
         cites=["S4 body: headline population non-identification share"],
         cross_check="round(100 * fredmd.nonidentified / fredmd.total)")
    emit("fredmd.total", int(n),
         spec="scorable monthly series surviving the length/finiteness filters",
         cites=["Abstract", "Introduction", "S4 body"],
         cross_check="fred_md_sweep.csv: row count")
    emit("fredmd.total_candidates", len(FREDMD_CODES),
         spec="raw FRED-code candidate list pulled via the API before the length/finiteness filter "
              "drops the never-scorable-at-monthly-frequency quarterly series (fredmd.total is the "
              "post-filter count)",
         cites=["S3.2 body: FRED-MD candidate universe"],
         cross_check="len(FREDMD_CODES)")
    emit("fredmd.identified_complement", int((r.MCS_abs == 1).sum()), of=int(n),
         spec="total minus non-identified; recorded explicitly because it is a DERIVED quantity that "
              "the manuscript states separately",
         cites=["S4.2: FRED-MD identified complement"],
         cross_check="fred_md_sweep.csv: (MCS_abs == 1).sum()")
    emit("fredmd.squared_nonidentified", int((r.MCS_sq > 1).sum()), of=int(n),
         spec="series with |MCS_sq|>1 at alpha=0.10; identical protocol to fredmd.nonidentified, squared "
              "rather than absolute loss. NOTE: despite its name, this file's MCS_sq is a rooted "
              "RMSSE (sqrt per year, then pooled) and so matches mcs_test.py's 'MCS_rmsse', NOT "
              "mcs_test.py's own 'MCS_sq' column, which is deliberately un-rooted; a reader "
              "cross-referencing the two CSVs by column name alone would be misled about which "
              "construction this one uses. Not quoted in the manuscript (cites=[]).",
         cites=[],
         cross_check="fred_md_sweep.csv: (MCS_sq > 1).sum()")
    _rho, _p = stats.spearmanr(r["gap_t"], r["MCS_abs"])
    emit("fredmd.spearman_gap_mcs", float(_rho), n=int(n), p_value=float(_p),
         spec="Spearman rank correlation between the studentized best-vs-second-best gap and |MCS_abs| "
              "across the population panel",
         cites=[],
         cross_check="fred_md_sweep.csv: spearman(gap_t, MCS_abs)")
    _has_tcode = r["MCS_abs_tcode"].notna()
    n_tcode = int(_has_tcode.sum())
    n_with_code = int(r.series.isin(FREDMD_TCODE).sum())
    if n_tcode != n_with_code - n_tcode_outlier_skipped:
        raise RuntimeError(
            f"fredmd.tcode_subset ({n_tcode}) != series carrying a published code ({n_with_code}) minus "
            f"deliberately-skipped outliers ({n_tcode_outlier_skipped}) -- a transform run failed "
            f"silently (see the [tcode skip] print above) rather than the subset being purely 'has a "
            f"published code, and was not a disclosed outlier skip', which is what this key's spec asserts")
    emit("fredmd.tcode_subset", n_tcode, of=int(n),
         spec="series carrying a PUBLISHED McCracken-Ng transformation code, on which the levels and "
              "transformed specifications are both defined and therefore comparable. Excludes "
              "any series whose transformed values are "
              f"dominated by a near-zero-crossing arithmetic artifact (see tcode_outlier_ratio()) -- "
              f"{n_tcode_outlier_skipped} series excluded this run on that basis",
         cites=[],
         cross_check="fred_md_sweep.csv: MCS_abs_tcode.notna().sum()")
    emit("fredmd.nonidentified_levels_subset", int((r.MCS_abs > 1)[_has_tcode].sum()), of=n_tcode,
         spec="non-identified in LEVELS, restricted to the published-tcode subset; the same "
              "specification as the headline, on the comparable sample",
         cites=["S4.1 body: stationarity-transform sensitivity"],
         cross_check="fred_md_sweep.csv: (MCS_abs > 1).sum() where MCS_abs_tcode.notna()")
    emit("fredmd.nonidentified_tcode", int((r.MCS_abs_tcode > 1).sum()), of=n_tcode,
         movement_from_levels_subset=int((r.MCS_abs > 1)[_has_tcode].sum()) - int((r.MCS_abs_tcode > 1).sum()),
         spec="non-identified after applying each series' PUBLISHED McCracken-Ng transformation code; "
              "identical MCS protocol, alpha=0.10, only the stationarity transform differs. CAVEAT: "
              "part of the drop from fredmd.nonidentified_levels_subset is "
              "mechanical, not purely 'stationarity was an artifact' -- the tcode transform (e.g. "
              "log-difference) can turn a trending level series into a near-white-noise target, on "
              "which the `theta` model (in MODELS above) tends to degenerate toward a near-constant "
              "forecast; that alone can separate `theta` from the other three models on absolute loss "
              "and shrink |MCS|, independent of whether non-stationarity was inflating the levels-side "
              "count. Read this key beside fredmd.nonidentified_levels_subset as a bound, not a clean "
              "counterfactual",
         cites=["S4.1 body: stationarity-transform sensitivity"],
         cross_check="fred_md_sweep.csv: (MCS_abs_tcode > 1).sum()")
    print(f"wrote {RES/'fred_md_sweep.csv'}")
    flush()


if __name__ == "__main__":
    main()

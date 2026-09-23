"""
fragility.py — the RANKING-FRAGILITY diagnostic: a decision-level BREAKDOWN POINT for forecast model selection.
"""
import numpy as np

_MARGIN_REL_FLOOR = 1e-12   # a pooled-loss margin below 1e-12 x its scale is float64 residue: treat as a tie

FRAGILE_STABILITY_THRESH = 0.60     # the paper's fixed Fragile rule: winner survives < 60% of resamples ...
FRAGILE_KSTAR_RATIO_THRESH = 0.25   # ... or reverses with fewer than a quarter of the periods removed


def _margin_below_floor(la, lb, w, M):
    """True iff the weighted margin M (from `la` to `lb`) is at or below the same degenerate-margin
    floor breakdown_number() already applies -- factored out here so concentration/conc_by_opp can
    share it."""
    T = len(w)
    if T == 0:
        return True
    scale = float(np.mean(np.abs(la)) + np.mean(np.abs(lb))) / 2.0
    wscale = float(np.mean(w))
    return M <= 0 or (M / T) <= _MARGIN_REL_FLOOR * scale * wscale


def pooled_winner(L: dict, w):
    """Model with the lowest weighted-mean loss (the pooled 'best')."""
    w = np.asarray(w, float)
    if np.any(w < 0):
        raise ValueError("pooled_winner: weights must not contain negative values -- weights are "
                          "observation counts, not signed contributions.")
    if not np.any(w > 0):
        raise ValueError("pooled_winner: weights must include at least one strictly positive value "
                          "-- an all-zero weight vector has no observations to average over.")
    means = {m: float(np.average(np.asarray(L[m], float), weights=w)) for m in L}
    nan_models = sorted(m for m, v in means.items() if not np.isfinite(v))
    if nan_models:
        raise ValueError(f"pooled_winner: non-finite pooled mean for model(s) {nan_models} -- cannot "
                          f"rank against NaN/inf; filter or impute before calling.")
    best = min(means.values())
    best_m = min(sorted(means), key=lambda m: means[m])
    tied = sorted(m for m in L
                  if means[m] - best <= _MARGIN_REL_FLOOR * (abs(means[best_m]) + abs(means[m])) / 2.0)
    return tied[0]


def per_period_winner(L: dict, return_ties=False):
    """The robust winner: the model that wins the MOST periods (majority vote over per-period lowest loss)."""
    models = list(L)
    M = np.vstack([np.asarray(L[m], float) for m in models])      # models x T
    per_period_min = M.min(axis=0)
    is_unique_min = (M == per_period_min[None, :]) & (np.sum(M == per_period_min[None, :], axis=0) == 1)
    wins = is_unique_min.sum(axis=1)                                # STRICT per-period wins only
    top = int(wins.max())
    tied = sorted(models[i] for i in range(len(models)) if wins[i] == top)   # sorted -> caller-independent
    winner = tied[0]
    return (winner, tied) if return_ties else winner


def condorcet_winner(L: dict):
    """The CONDORCET winner: the model that beats every other model in pairwise per-period majority (lower loss more
    often than not)."""
    models = list(L)
    M = {m: np.asarray(L[m], float) for m in models}
    for a in models:
        if all(a == b or int((M[a] < M[b]).sum()) > int((M[b] < M[a]).sum()) for b in models):
            return a
    return None


def condorcet_status(L: dict):
    """Classify the pairwise-majority tournament among the models:
      'winner' -- a Condorcet winner exists (one model strictly beats every other in per-period majority);
      'cycle'  -- NO Condorcet winner AND the strict-beats digraph contains a directed cycle of ANY length
                  (a1>a2>...>ak>a1) -- a genuine intransitive majority cycle;
      'tie'    -- NO Condorcet winner and NO directed cycle (the absence is driven by pairwise majority TIES,
                  not intransitivity)."""
    from collections import deque
    models = list(L)
    M = {m: np.asarray(L[m], float) for m in models}
    def beats(x, y):
        return int((M[x] < M[y]).sum()) > int((M[y] < M[x]).sum())
    for a in models:
        if all(a == b or beats(a, b) for b in models):
            return "winner"
    indeg = {m: 0 for m in models}
    adj = {m: [] for m in models}
    for x in models:
        for y in models:
            if x != y and beats(x, y):
                adj[x].append(y); indeg[y] += 1
    q = deque([m for m in models if indeg[m] == 0]); ordered = 0
    while q:
        u = q.popleft(); ordered += 1
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    return "cycle" if ordered < len(models) else "tie"


def breakdown_number(la, lb, w):
    """k* = the minimum number of periods to DELETE to flip 'a beats b' on the pooled metric, where a is the current
    pooled winner of the pair."""
    la, lb, w = np.asarray(la, float), np.asarray(lb, float), np.asarray(w, float)
    if np.any(w < 0):
        raise ValueError("breakdown_number: weights must not contain negative values -- weights are "
                          "observation counts, not signed contributions.")
    c = w * (lb - la)
    M = float(c.sum())
    T = len(w)
    scale = float(np.mean(np.abs(la)) + np.mean(np.abs(lb))) / 2.0 if T else 0.0
    wscale = float(np.mean(w)) if T else 0.0
    if M <= 0 or (T and (M / T) <= _MARGIN_REL_FLOOR * scale * wscale):
        return 0, []
    order = np.argsort(-c, kind="stable")                         # largest a-favoring contribution first, ties -> earliest period
    s, removed = M, []
    for i in order:
        if c[i] <= 0:                                            # removing a b-favoring period can't help flip
            break
        s -= float(c[i]); removed.append(int(i))
        if s <= 0 or s <= _MARGIN_REL_FLOOR * scale * T * wscale:
            break
    return len(removed), removed


def _brute_force_k(la, lb, w):
    """Reference implementation for the test: smallest subset (any combination) whose deletion flips a→b. O(2^T)."""
    from itertools import combinations
    la, lb, w = np.asarray(la, float), np.asarray(lb, float), np.asarray(w, float)
    c = w * (lb - la)
    if c.sum() <= 0:
        return 0
    T = len(c)
    for k in range(1, T + 1):
        for S in combinations(range(T), k):
            if c.sum() - c[list(S)].sum() <= 0:
                return k
    return T


def decision_breakdown(L: dict, w, a=None, return_ties=False):
    """DECISION-LEVEL k*: the fewest periods whose deletion makes the pooled winner `a` lose to ANY opponent
    (not just the pooled runner-up)."""
    if a is None:
        a = pooled_winner(L, w)
    la = np.asarray(L[a], float)
    best = None                                                  # (k, opponent, removed)
    ties = []
    for m in sorted(m for m in L if m != a):                     # sorted -> caller-independent
        k, removed = breakdown_number(la, np.asarray(L[m], float), w)
        if best is None or k < best[0]:
            best = (k, m, removed); ties = [m]
        elif k == best[0]:
            ties.append(m)
    if best is None:
        return (0, None, [], []) if return_ties else (0, None, [])
    return (*best, ties) if return_ties else best


def _brute_force_k_multi(L: dict, w):
    """Reference: fewest deletions after which `a` is no longer the pooled winner (any opponent overtakes). O(2^T)."""
    from itertools import combinations
    w = np.asarray(w, float); T = len(w); a = pooled_winner(L, w); la = np.asarray(L[a], float)
    def a_still_wins(keep):                       # tie counts as 'a lost' — matches breakdown_number's M<=0 convention
        aw = float(np.average(la[keep], weights=w[keep]))
        return all(float(np.average(np.asarray(L[m], float)[keep], weights=w[keep])) > aw for m in L if m != a)
    if not a_still_wins(np.arange(T)):
        return 0
    allset = set(range(T))
    for k in range(1, T + 1):
        for S in combinations(range(T), k):
            keep = np.array(sorted(allset - set(S)), dtype=int)
            if len(keep) and not a_still_wins(keep):
                return k
    return T


def _block_resample_idx(T, block, rng, circular=True):
    """Block-bootstrap indices over evaluation periods, CIRCULAR by default (Politis & Romano 1992)."""
    if block <= 1:
        return rng.integers(0, T, size=T)  # iid case: identical under either scheme
    if not circular:
        out = []
        while len(out) < T:
            start = int(rng.integers(0, max(1, T - block + 1)))
            out.extend(range(start, min(start + block, T)))
        return np.asarray(out[:T])
    starts = rng.integers(0, T, size=int(np.ceil(T / block)))
    return np.concatenate([(s + np.arange(block)) % T for s in starts])[:T]


def winner_stability(L: dict, w, n_boot=2000, block=None, seed=0, circular=True):
    """Block-bootstrap the evaluation PERIODS and recompute the pooled winner on each resample → the distribution of
    'who wins'."""
    w = np.asarray(w, float)
    T = len(w)
    block = max(1, round(T ** (1 / 3))) if block is None else block
    if block >= T:
        raise ValueError(f"block={block} >= T={T} periods -- a circular block bootstrap with a "
                          f"block at least as long as the sample only ever produces a rotation of "
                          f"the same {T} periods, so the pooled winner cannot change across "
                          f"resamples and stability would be trivially 1.0 regardless of the data. "
                          f"Pass a smaller block (default is round(T**(1/3))).")
    if n_boot <= 0:
        raise ValueError(f"n_boot must be a positive integer; got {n_boot}.")
    obs = pooled_winner(L, w)
    _k, _b, _removed = decision_breakdown(L, w, a=obs)
    _la = np.asarray(L[obs], float)
    _lb = np.asarray(L[_b], float) if _b is not None else _la
    _M = float((w * (_lb - _la)).sum())
    if _k == 0 or _margin_below_floor(_la, _lb, w, _M):
        return float("nan"), {m: 1.0 / len(L) for m in L}
    rng = np.random.default_rng(seed)
    counts = {m: 0 for m in L}
    for _ in range(n_boot):
        idx = _block_resample_idx(T, block, rng, circular=circular)
        Lb = {m: np.asarray(L[m], float)[idx] for m in L}
        counts[pooled_winner(Lb, w[idx])] += 1
    freq = {m: counts[m] / n_boot for m in L}
    return freq[obs], freq


def exchangeable_benchmark(K, T, n_rep=200, n_boot=500, block=None, seed=0, circular=True):
    """The distribution of `winner_stability` when the winner is ARBITRARY BY CONSTRUCTION."""
    w = np.ones(T)
    out = []
    for r in range(n_rep):
        g = np.random.default_rng(seed + 1 + r)
        L = {f"m{i}": g.normal(1.0, 0.30, T) for i in range(K)}
        boot_seed = seed + 1_000_003 + r
        out.append(winner_stability(L, w, n_boot=n_boot, block=block, seed=boot_seed, circular=circular)[0])
    out = np.sort(np.asarray(out, float))
    return float(np.median(out)), float(np.percentile(out, 5)), float(np.percentile(out, 95))


def surprise_concentration(L: dict, labels=None):
    """How concentrated is the IRREDUCIBLE forecast difficulty across periods?"""
    models = list(L)
    Mmat = np.vstack([np.asarray(L[m], float) for m in models])   # models x T
    d = Mmat.min(axis=0)                                           # irreducible per-period difficulty
    b = float(np.median(d))
    s = np.maximum(0.0, d - b)
    tot = float(s.sum())
    lab = list(labels) if labels is not None else list(range(len(d)))
    d_max = float(np.max(d))
    if b == 0.0:
        d_max_over_median = 1.0 if d_max == 0.0 else (np.inf if d_max > 0.0 else -np.inf)
    elif (b < 0.0) != (d_max < 0.0):
        d_max_over_median = np.nan
    else:
        d_max_over_median = d_max / b
    return {
        "sci": float(s.max() / tot) if tot > 0 else 0.0,
        "eff_surprising_periods": float(tot ** 2 / np.sum(s ** 2)) if tot > 0 else float(len(d)),
        "hardest": lab[int(np.argmax(d))],
        "d_max_over_median": float(d_max_over_median),
    }


def fragility(L: dict, w, shock_periods=None, labels=None, n_boot=2000, seed=0, *, stability_block=None):
    """Full diagnostic."""
    if len(L) < 2:
        raise ValueError(f"fragility() requires at least 2 models to compare, got {len(L)}: {list(L)}")
    w = np.asarray(w, float)
    T = len(w)
    a = pooled_winner(L, w)
    # sorted(): on an exact pooled-loss tie, `min` otherwise returns whichever model the caller happened to
    # insert into the dict first -- same non-determinism class as pooled_winner's own tie-break above.
    pooled_runner_up = min(sorted(m for m in L if m != a),
                           key=lambda m: float(np.average(np.asarray(L[m], float), weights=w)))
    # DECISION-level k*: min over ALL opponents (the decision breaks when a loses to ANY model, not just the
    # pooled 2nd) — b is the BINDING opponent that overtakes a with the fewest deletions.
    k, b, removed, binding_ties = decision_breakdown(L, w, a, return_ties=True)
    la = np.asarray(L[a], float)
    lb = np.asarray(L[b], float) if b is not None else la          # guard: single-model dict (no opponents)
    c = w * (lb - la)
    M = float(c.sum())
    concentration = np.nan if _margin_below_floor(la, lb, w, M) else float(np.max(c) / M)
    conc_by_opp = {}
    for m in binding_ties:
        lm = np.asarray(L[m], float)
        cm = w * (lm - la)
        Mm = float(cm.sum())
        conc_by_opp[m] = np.nan if _margin_below_floor(la, lm, w, Mm) else float(np.max(cm) / Mm)
    _finite = [v for v in conc_by_opp.values() if np.isfinite(v)]
    conc_lo, conc_hi = (min(_finite), max(_finite)) if _finite else (np.nan, np.nan)
    pp, pp_tied = per_period_winner(L, return_ties=True)
    reversal_ambiguous = bool(len(pp_tied) > 1 and a in pp_tied)
    cw = condorcet_winner(L)
    cstat = condorcet_status(L)                # 'winner' | 'cycle' | 'tie' (cw is not None <=> 'winner')
    lab = list(labels) if labels is not None else list(range(T))
    resp = [lab[i] for i in removed]
    resp_by_opp = {
        m: [lab[i] for i in breakdown_number(la, np.asarray(L[m], float), w)[1]]
        for m in binding_ties
    }
    coincide = None
    if shock_periods is not None:
        sset = set(shock_periods)
        coincide = (len(resp) > 0) and all(r in sset for r in resp)
    stability, freq = winner_stability(L, w, n_boot=n_boot, block=stability_block, seed=seed)
    fragile = bool((stability < FRAGILE_STABILITY_THRESH)
                    or ((a != pp) and (k / T < FRAGILE_KSTAR_RATIO_THRESH)))
    return {
        "pooled_winner": a, "per_period_winner": pp, "condorcet_winner": cw,
        "runner_up": b, "pooled_runner_up": pooled_runner_up,
        "k_star": int(k), "k_over_T": float(k / T) if T else np.nan,
        "concentration": concentration, "reversal": (a != pp),
        "n_binding_tied": len(binding_ties), "concentration_by_opponent": conc_by_opp,
        "concentration_lo": conc_lo, "concentration_hi": conc_hi,
        "per_period_tied": pp_tied, "reversal_ambiguous": reversal_ambiguous,
        "responsible_by_opponent": resp_by_opp,
        "reversal_condorcet": (cw is not None and a != cw),
        "condorcet_status": cstat,         # 'winner' | 'cycle' | 'tie'
        "condorcet_cycle": (cstat == "cycle"),   # TRUE intransitive cycle ONLY (not tie-induced absence of a winner)
        "condorcet_undefined": (cw is None),     # no Condorcet winner (cycle OR tie) -- for the reversal denominator
        "responsible": resp, "shock_coincidence": coincide,
        "winner_stability": float(stability), "winner_freq": freq, "fragile": fragile,
    }

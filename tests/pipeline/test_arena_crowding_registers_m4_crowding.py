"""arena_crowding.py is the ONLY producer that actually computes the real-vs-independence-null
crowding comparison. What that comparison actually shows, recomputed from the committed
multiseries/results/canonical/arena_crowding.json (null minus real, i.e. how much less the real
panel flips than its independence null):

    M4  (61 submissions):  k=2  real 0.046 vs null 0.300  -> gap 0.254
                           k=32 real 0.244 vs null 0.647  -> gap 0.403
                           k=60 real 0.020 vs null 0.669  -> gap 0.649
    LLM (64 models):       k=2  real 0.064 vs null 0.253  -> gap 0.189
                           k=64 real 0.362 vs null 0.530  -> gap 0.168

So the direction is arm-specific and must not be stated as one law: the real panel flips less than
the null at EVERY pool size tested in both arms, but the M4 arm's gap WIDENS monotonically as the
pool grows (0.254 -> 0.403 -> 0.649), while only the LLM arm's gap narrows slightly (0.189 ->
0.168). Earlier revisions of this docstring asserted a single narrowing/"washes out" direction, in
quotation marks that quoted nothing; both were wrong for M4.

The nearest-looking alternative producer, density_audit.py's
density.flip_rate_min_pool/max_pool, is a real-panel flip rate only -- it computes no null at all,
which is why arena_crowding.py's own emits describe themselves as "the counterfactual that
density.flip_rate_min_pool is cited against, which density_audit.py does not itself compute".

So arena_crowding.py's main() must CAPTURE `arena("M4 competition...", LM4, ...)`'s return value and
emit the M4 arena's real/null flip rates at the min and max pool sizes tested. Discarding it would
leave the one producer that computes the mechanism emitting nothing about it (only llm.n_models,
unrelated).

(Both families currently ship with `cites: []`, so neither is quoted in the draft and claims.py's
misplaced() skips them as UNCITED rather than reporting them -- the gate offers no cover here, which
is exactly why this test exists.)
"""
import ast
import inspect
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "code" / "pipeline"))
sys.path.insert(0, str(ROOT / "code" / "instrument"))

import arena_crowding as AC  # noqa: E402

M4_CROWDING_KEYS = (
    "arena.m4_flip_real_min_pool", "arena.m4_flip_null_min_pool",
    "arena.m4_flip_real_max_pool", "arena.m4_flip_null_max_pool",
)


def _main_ast() -> ast.Module:
    return ast.parse(textwrap.dedent(inspect.getsource(AC.main)))


def _main_emit_keys() -> list[str]:
    """Every string literal passed as emit()'s first argument inside main(), read off the live
    source via AST -- not a regex over the file, and not the committed JSON on disk."""
    keys = []
    for node in ast.walk(_main_ast()):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name) and node.func.id == "emit"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)):
            keys.append(node.args[0].value)
    return keys


def test_main_source_still_emits_the_four_m4_crowding_keys():
    """THE falsifiability anchor for this file. Every other test here either exercises arena()/
    crowd() on synthetic input or reads the already-committed canonical JSON off disk, so renaming
    or deleting main()'s `emit("arena.m4_flip_*")` calls outright used to leave all of them green
    -- the one producer of the mechanism could stop emitting it and nothing in the suite noticed.
    This test reads main()'s own AST, so a deleted or renamed emit fails immediately."""
    keys = _main_emit_keys()
    assert keys, "main()'s source must contain emit() calls with literal string keys"
    missing = [k for k in M4_CROWDING_KEYS if k not in keys]
    assert not missing, (
        f"arena_crowding.main() no longer emits {missing} -- these are the M4 arena's real/null "
        f"flip rates at the min and max pool sizes, and arena_crowding.py is the only producer "
        f"that computes them. Emitted keys found: {sorted(set(keys))}"
    )


def test_main_captures_the_m4_arena_return_value():
    """The companion structural claim: main() must ASSIGN arena(...)'s five-element return, not
    call it for its printed side effects and discard the numbers -- discarding it is exactly how
    the emits above would silently lose their inputs."""
    targets = []
    for node in ast.walk(_main_ast()):
        if (isinstance(node, ast.Assign)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name) and node.value.func.id == "arena"):
            targets.append(node.targets[0])
    assert targets, "main() must capture arena()'s return value in an assignment"
    widths = {len(t.elts) for t in targets if isinstance(t, ast.Tuple)}
    assert widths == {5}, (
        f"every arena() call in main() must unpack all five returned values "
        f"(real, null, tcorr, null_sd, null_first); found tuple widths {widths or 'none'}"
    )


def test_arena_returns_nonempty_real_and_null_dicts():
    """arena()'s return value must be a genuine, non-empty (real, null, tcorr) triple the caller can
    actually use -- not silently discarded. `tcorr` in particular is computed and printed in every
    call's header, and a `return real, null` signature would drop it; see
    arena.m4_mean_cross_metric_corr's own emit() for why it is needed."""
    rng = np.random.default_rng(0)
    L = rng.normal(1.0, 0.2, (20, 4))
    # arena() returns the permutation-AVERAGED null plus its sd and the first (seed-0) draw. The
    # averaging matters: on a 61-row panel one permutation's flip rate at k=32 ranged over
    # [0.15, 0.98], so the mean over N_NULL_PERM permutations is the null the emits carry.
    real, null, tcorr, null_sd, null_first = AC.arena("synthetic test arena", L, [2, 4, 8, 16])
    assert real, "real flip-rate dict must not be empty"
    assert null, "null flip-rate dict must not be empty"
    assert set(real) == set(null) == set(null_sd) == set(null_first), "all curves share the pool sizes"
    for k, v in {**real, **null, **null_first}.items():
        assert 0.0 <= v <= 1.0, f"flip rate at k={k} out of [0,1]: {v}"
    for k, v in null_sd.items():
        assert 0.0 <= v <= 0.5, f"null sd at k={k} implausible: {v}"
    assert -1.0 <= tcorr <= 1.0, f"tcorr out of [-1,1]: {tcorr}"


def test_arena_null_is_averaged_over_permutations_and_first_draw_is_the_old_single_permutation():
    """The averaged null must not be a copy of the first draw, and the first draw must reproduce
    what a single seed-0 column permutation (the null this file used before) gives."""
    rng = np.random.default_rng(3)
    L = rng.normal(1.0, 0.2, (20, 4))
    real, null, tcorr, null_sd, null_first = AC.arena("synthetic", L, [2, 4, 8, 16])
    r2 = np.random.default_rng(0)
    Lnull0 = np.column_stack([r2.permutation(L[:, j]) for j in range(L.shape[1])])
    AC.crowd(L, r2, [2, 4, 8, 16])                       # the real curve consumes the stream next, as in arena()
    old_single = AC.crowd(Lnull0, r2, [2, 4, 8, 16])
    assert null_first == old_single
    assert any(abs(null[k] - null_first[k]) > 1e-12 for k in null), "averaging over permutations must change the null"


def test_crowd_flip_rate_is_a_real_probability():
    """crowd()'s own output, the building block of arena(), must be a genuine per-pool-size flip
    rate in [0, 1], not a placeholder or NaN."""
    rng = np.random.default_rng(1)
    L = rng.normal(1.0, 0.3, (15, 5))
    out = AC.crowd(L, rng, [2, 5, 10])
    assert set(out) == {2, 5, 10}
    for v in out.values():
        assert 0.0 <= v <= 1.0
        assert np.isfinite(v)


def test_real_canonical_json_has_the_m4_crowding_values():
    """Direct check against the real, currently-committed canonical output: the M4 real/null
    crowding values must be present, with the null contribution genuinely different from the real
    one at the small-pool end (the identification signal S9's prose describes)."""
    import json
    canon_path = ROOT / "multiseries" / "results" / "canonical" / "arena_crowding.json"
    if not canon_path.exists():
        pytest.skip("arena_crowding.json not available")
    canon = json.loads(canon_path.read_text())
    for key in ("arena.m4_flip_real_min_pool", "arena.m4_flip_null_min_pool",
                "arena.m4_flip_real_max_pool", "arena.m4_flip_null_max_pool"):
        assert key in canon, f"{key} missing from the real canonical/arena_crowding.json"
        v = canon[key]["value"]
        assert 0.0 <= v <= 1.0, f"{key}={v} out of [0,1]"
    # at the smallest pool size, this codebase's own printed output (and S9's prose) claims the
    # real panel is MORE identified (lower flip rate) than the null -- confirms the values aren't
    # just present but point the right direction
    assert canon["arena.m4_flip_real_min_pool"]["value"] < canon["arena.m4_flip_null_min_pool"]["value"], (
        "at the smallest pool size, the real M4 panel should flip LESS than its independence null "
        "(the exact claim these values exist to eventually support)"
    )

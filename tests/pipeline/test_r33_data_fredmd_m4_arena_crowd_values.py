"""Mutation-verified tests for arena_crowding.crowd(): the function behind every arena.*_flip_* and llm.crowd_* value had only key-set,
range and real<null tests, so two one-token mutations that move the paper's quoted "real flip share
of 0.02" survived the suite:
  (a) `rng.choice(len(L), k, replace=False)` -> `replace=True`   (M4 k=60: ~0.4, not ~0.02)
  (b) `np.argsort(np.argsort(sub, axis=0), axis=0)` -> `axis=1`  (M4 k=60: ~0.98, not ~0.02)
(the mutant magnitudes above were measured against a then-current baseline of 0.016; the shipped
arena.m4_flip_real_max_pool at k=60 is now 0.02, which does not change either mutant's verdict)
The fixtures below have PLANTED answers that are exact (0.0 or 1.0) whenever k equals the panel
size, because a without-replacement draw of k = n models is the whole panel in some order and the
mean-loss-vs-Borda verdict of the whole panel is a fixed fact. Every test here was verified to FAIL
on the named mutant and PASS on the real module.
"""
import numpy as np
import pytest

import arena_crowding as AC


# 3 models x 3 tasks, lower = better.  A has the lowest MEAN loss (4 vs 5 vs 9) but B wins two of
# the three tasks (5 < 6 on tasks 2 and 3), so per-task ranks are
#   task1: A0 B1 C2 | task2: B0 A1 C2 | task3: B0 A1 C2  -> mean rank A 2/3, B 1/3, C 2
# Borda winner = B, mean-loss winner = A: the whole panel FLIPS.  Pairwise: {A,B} flips, {A,C}
# and {B,C} do not.
FLIP_PANEL = np.array([[0.0, 6.0, 6.0],
                       [5.0, 5.0, 5.0],
                       [9.0, 9.0, 9.0]])

# A beats every other model on every task: no sub-panel of any size can flip.
DOMINANT_PANEL = np.array([[1.0, 1.0, 1.0],
                           [2.0, 3.0, 4.0],
                           [3.0, 2.0, 5.0],
                           [4.0, 4.0, 4.5]])


@pytest.mark.parametrize("seed", [0, 1, 7, 12345])
def test_full_panel_draw_flip_rate_is_exactly_one_on_the_planted_flip_panel(seed):
    """kills: (a) replace=True -- with-replacement draws of size 3 include (A,A,A)-type multisets
    (probability 1/9) on which mean and Borda trivially agree, so the rate drops below 1; and
    (b) axis=1 -- ranking within each model's row makes every model's mean rank (T-1)/2, the
    'Borda winner' collapses to draw position 0, and the rate becomes ~2/3 (whenever A is not
    drawn first). The correct k = n draw sees the whole panel every time: exactly 1.0, any seed."""
    out = AC.crowd(FLIP_PANEL, np.random.default_rng(seed), ks=[3])
    assert out == {3: 1.0}, out


@pytest.mark.parametrize("seed", [0, 3])
def test_full_panel_draw_flip_rate_is_exactly_zero_on_a_dominant_panel(seed):
    """A model that wins every task is both the mean-loss and the Borda winner of every sub-panel:
    exact 0.0 at every k, any seed (the companion bound; also exact for the k < n draws)."""
    out = AC.crowd(DOMINANT_PANEL, np.random.default_rng(seed), ks=[2, 3, 4])
    assert out == {2: 0.0, 3: 0.0, 4: 0.0}, out


def test_pairwise_flip_share_on_the_planted_panel_is_about_one_third():
    """kills: (b) axis=1 (pairs then flip whenever the mean-loss winner is drawn second: ~1/2)
    and (a) replace=True (draws (X,X) never flip, {A,B} is then drawn with probability 2/9 not
    1/3: ~0.22). Exactly one of the three 2-subsets ({A,B}) flips, so the 500-draw share is
    1/3 +- Monte-Carlo noise (sd 0.021); the band below is +-4 sd."""
    out = AC.crowd(FLIP_PANEL, np.random.default_rng(0), ks=[2])
    assert 0.25 <= out[2] <= 0.42, out


def test_two_model_panel_flip_is_exact_and_independent_of_draw_order():
    """kills: (b) axis=1. On the 2-model sub-panel {A, B} of the planted panel, k = 2 = n draws
    the pair in one of two orders; the verdict (B wins 2 of 3 tasks, A has the lower mean) is a
    flip in BOTH orders, so the rate is exactly 1.0. Under the row-wise rank mutant the verdict
    depends on which model is drawn first and the rate is ~0.5."""
    out = AC.crowd(FLIP_PANEL[:2], np.random.default_rng(0), ks=[2])
    assert out == {2: 1.0}, out

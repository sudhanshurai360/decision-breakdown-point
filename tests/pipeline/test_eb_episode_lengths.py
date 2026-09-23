"""_episode_lengths() (state_eb_decision.py).

THE HAZARD: computing eb.max_episode_months as (flip_months/flip_episodes).max() per state gives
the max of each state's PER-EPISODE AVERAGE length, not the true longest single episode. That is
only correct when a state's episodes are all the same length. A state with episode lengths {1, 5}
averages to 3, silently hiding the true 5-month episode -- and it would be correct in this
project's real data purely by coincidence (every one of the 24 divergent states happens to have
all-equal-length episodes), not by construction.

THE RULE: _episode_lengths() tracks each contiguous flip-run's true length directly, so
`_episode_lengths(flip).max()` reports the real longest episode regardless of whether episode lengths
are equal across a state.
"""
import pandas as pd

from state_eb_decision import _episode_lengths


def _flip_series(pattern: str) -> pd.Series:
    """'pattern' e.g. "TTFTTTTTFFT" -> a boolean Series over a matching monthly DatetimeIndex."""
    idx = pd.date_range("2000-01-01", periods=len(pattern), freq="MS")
    return pd.Series([c == "T" for c in pattern], index=idx)


def test_unequal_episode_lengths_max_is_not_the_average():
    """The exact bug this fix closes: episodes of length {1, 5} average to 3 -- the old formula would
    have reported 3, not the true max of 5."""
    flip = _flip_series("T" + "F" * 3 + "T" * 5 + "F" * 2)   # one 1-month episode, one 5-month episode
    lens = _episode_lengths(flip)
    assert sorted(lens.tolist()) == [1, 5]
    assert lens.max() == 5
    assert lens.mean() == 3.0                                 # confirms the old (wrong) formula would say 3, not 5


def test_all_equal_length_episodes_matches_the_old_approximation():
    """Sanity check: when every episode IS the same length (the real dataset's actual case), the new
    exact computation agrees with the old flip_months/flip_episodes average -- the fix changes nothing
    for the data this project actually has."""
    flip = _flip_series("TT" + "F" * 2 + "TT" + "F" * 2 + "TT")   # three 2-month episodes
    lens = _episode_lengths(flip)
    assert len(lens) == 3
    assert (lens == 2).all()
    assert lens.max() == 2
    flip_months, flip_episodes = int(flip.sum()), len(lens)
    assert lens.max() == flip_months / flip_episodes             # old approximation still correct here


def test_no_flip_episodes_returns_empty():
    flip = _flip_series("F" * 6)
    lens = _episode_lengths(flip)
    assert len(lens) == 0


def test_single_isolated_month_episodes_all_length_one():
    """The real dataset's actual shape (per the codebase's own established finding): every episode is
    exactly one isolated month, never adjacent."""
    flip = _flip_series("TFTFFTFT")
    lens = _episode_lengths(flip)
    assert len(lens) == 4
    assert (lens == 1).all()


def test_episode_touching_the_series_boundary_counts_correctly():
    """An episode that starts at month 0 (no preceding False to detect a 'start' against) must still
    be counted -- `shift(1, fill_value=False)` is what makes this work."""
    flip = _flip_series("TTTF" + "T")
    lens = _episode_lengths(flip)
    assert sorted(lens.tolist()) == [1, 3]


def test_calendar_gap_does_not_merge_two_separate_episodes():
    """A plain POSITIONAL `flip.shift(1)` with no calendar-grid reindex would treat the row
    immediately AFTER a calendar gap as if it followed the row immediately BEFORE it -- and `flip`'s
    own index does have genuine multi-month gaps (the real Oct-2025 51-state UR data upstream, via
    `u = df[s].dropna()` in scan()). Reproduced directly: True on both sides of a 4-calendar-month
    gap (Jun 2020 and Oct-Nov 2020, with Jul-Sep 2020 entirely absent from the index, not
    present-and-False) must count as two separate episodes (1 month, 2 months), never merged into
    one 3-month episode -- exactly the failure mode this codebase's own
    date_shift/date_rolling_mean/date_rolling_min (decision_protocol.py) exist to avoid for the
    identical real cause. Currently inert on this paper's own real data (flip_months ==
    flip_episodes for all 50 states; max_episode_months == 1 for every divergent state -- see
    multiseries/results/state_eb_decision.csv), but handled at the source so a future data update
    cannot silently corrupt this statistic."""
    idx = (pd.date_range("2020-01-01", "2020-06-01", freq="MS").tolist()
           + pd.date_range("2020-10-01", "2020-12-01", freq="MS").tolist())   # Jul-Sep 2020 absent
    vals = [False, True, True, False, False, True] + [True, True, False]
    flip = pd.Series(vals, index=pd.DatetimeIndex(idx))
    lens = _episode_lengths(flip)
    assert sorted(lens.tolist()) == [1, 2, 2], (
        "a calendar gap incorrectly merged two genuinely separate episodes into one"
    )

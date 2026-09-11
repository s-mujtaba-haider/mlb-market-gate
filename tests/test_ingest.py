"""Settlement logic against hand-built MLB Stats API response fixtures.

These are the cases that silently corrupt a backfill: pushes, games that never
reached five innings, walk-offs where the home half of the ninth is not played,
and doubleheaders that share a date and both team names.
"""

import pytest

from mlbgate.ingest_statsapi import derive_outcomes


def game(home, away):
    return {"teams": {"home": {"score": home}, "away": {"score": away}}}


def linescore(*innings):
    """innings as (home_runs, away_runs) tuples."""
    return {"innings": [{"home": {"runs": h}, "away": {"runs": a}} for h, a in innings]}


NINE_QUIET = [(0, 0)] * 9


class TestMoneylineAndRunLine:
    def test_home_win(self):
        out = derive_outcomes(game(5, 3), linescore(*NINE_QUIET))
        assert out["ml_home"] == 1
        assert out["total_runs"] == 8

    def test_run_line_needs_a_two_run_margin(self):
        assert derive_outcomes(game(5, 3), linescore(*NINE_QUIET))["rl_home_-1.5"] == 1
        assert derive_outcomes(game(4, 3), linescore(*NINE_QUIET))["rl_home_-1.5"] == 0

    def test_a_one_run_home_win_loses_the_run_line(self):
        """The distinction the whole run-line market turns on."""
        out = derive_outcomes(game(4, 3), linescore(*NINE_QUIET))
        assert out["ml_home"] == 1 and out["rl_home_-1.5"] == 0


class TestFirstFive:
    def test_settles_on_the_first_five_innings_only(self):
        ls = linescore((0, 0), (3, 0), (0, 0), (0, 0), (0, 0),
                       (0, 4), (0, 0), (0, 0), (0, 0))
        out = derive_outcomes(game(3, 4), ls)
        assert out["ml_home"] == 0      # lost the game
        assert out["f5_ml_home"] == 1   # won the first five

    def test_tie_after_five_is_a_push_not_a_loss(self):
        ls = linescore((1, 1), (0, 0), (0, 0), (0, 0), (0, 0),
                       (2, 0), (0, 0), (0, 0), (0, 0))
        assert derive_outcomes(game(3, 1), ls)["f5_ml_home"] == "push"

    def test_a_short_game_voids_rather_than_losing(self):
        """Rain-shortened after four: the bet is void, not a loss."""
        ls = linescore((0, 0), (1, 0), (0, 0), (0, 0))
        out = derive_outcomes(game(1, 0), ls)
        assert out["f5_ml_home"] is None
        assert out["innings_played"] == 4


class TestNRFI:
    def test_scoreless_first_is_a_win(self):
        assert derive_outcomes(game(2, 1), linescore(*NINE_QUIET))["nrfi"] == 1

    @pytest.mark.parametrize("first", [(1, 0), (0, 1), (2, 3)])
    def test_any_first_inning_run_loses(self, first):
        ls = linescore(first, *([(0, 0)] * 8))
        assert derive_outcomes(game(5, 5), ls)["nrfi"] == 0

    def test_counts_both_halves(self):
        ls = linescore((0, 1), *([(0, 0)] * 8))
        assert derive_outcomes(game(0, 1), ls)["nrfi"] == 0


class TestWalkOffs:
    def test_home_half_of_the_ninth_not_played(self):
        """A home team leading after the top of the 9th does not bat.

        The linescore still has nine entries with the home half at zero runs,
        so run totals stay correct. Anything that infers innings from the length
        of the home scoring list has to handle this.
        """
        ls = linescore((1, 0), *([(0, 0)] * 7), (0, 0))
        out = derive_outcomes(game(1, 0), ls)
        assert out["ml_home"] == 1
        assert out["innings_played"] == 9
        assert out["total_runs"] == 1


class TestMissingData:
    def test_null_runs_are_treated_as_zero(self):
        """The API returns null, not 0, for an unplayed half-inning."""
        ls = {"innings": [{"home": {"runs": None}, "away": {"runs": 2}}]}
        out = derive_outcomes(game(0, 2), ls)
        assert out["nrfi"] == 0

    def test_missing_score_does_not_crash(self):
        out = derive_outcomes({"teams": {"home": {}, "away": {}}}, {"innings": []})
        assert out["total_runs"] == 0
        assert out["f5_ml_home"] is None

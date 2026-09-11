import numpy as np
import pytest

from mlbgate.ev import (
    closing_line_value,
    expected_value,
    kelly_fraction,
    settle,
    vig_free_close,
)
from mlbgate.odds import american_to_implied
from mlbgate.stats_tests import (
    benjamini_hochberg,
    block_bootstrap_ci,
    one_sided_p_value,
    required_sample_size,
    roi_bootstrap_ci,
    roi_standard_error,
)


class TestExpectedValue:
    def test_fair_coin_at_fair_price_is_zero_ev(self):
        assert expected_value(0.5, 100) == pytest.approx(0.0)

    def test_break_even_win_rate_at_minus_110(self):
        assert expected_value(0.5238, -110) == pytest.approx(0.0, abs=1e-3)

    def test_edge_is_positive_above_break_even(self):
        assert expected_value(0.56, -110) > 0

    def test_edge_over_fair_is_not_the_same_as_positive_ev(self):
        """The most common EV bug, quantified.

        Two different probabilities get called "the market's number" and they
        are 2.4 points apart at -110:

          raw implied (0.5238)  the break-even rate. Beat THIS to have +EV.
          vig-free    (0.5000)  the market's honest opinion. Beat THIS to have
                                information the market does not.

        Reporting `p_model - p_fair` as edge and then betting on it is a losing
        strategy that looks like a winning one, because it pockets the half-hold
        gap between the two numbers as if it were profit.
        """
        raw = american_to_implied(-110)
        fair = vig_free_close(-110, -110)
        assert fair == pytest.approx(0.50, abs=1e-6)
        assert raw - fair == pytest.approx(0.0238, abs=1e-3)

        # Break-even sits exactly at the raw implied number.
        assert expected_value(raw, -110) == pytest.approx(0.0, abs=1e-9)

        # A model that is exactly right about a coin flip still loses the hold.
        assert expected_value(fair, -110) == pytest.approx(-0.0455, abs=1e-4)

        # A full point of "edge over fair" is still a losing bet.
        assert expected_value(fair + 0.01, -110) < 0

        # Edge only becomes real past the break-even number.
        assert expected_value(raw + 0.005, -110) > 0


class TestKelly:
    def test_no_edge_means_no_stake(self):
        assert kelly_fraction(0.5, 100) == pytest.approx(0.0)

    def test_negative_edge_is_floored_at_zero(self):
        assert kelly_fraction(0.40, -110) == 0.0

    def test_known_value(self):
        # p=0.6 at +100: f = (0.6*1 - 0.4)/1 = 0.2
        assert kelly_fraction(0.6, 100) == pytest.approx(0.2)

    def test_cap_is_respected(self):
        assert kelly_fraction(0.99, 500, cap=0.05) == 0.05


class TestSettlement:
    def test_win_pays_the_price(self):
        assert settle("win", -110) == pytest.approx(100 / 110)
        assert settle("win", 150) == pytest.approx(1.5)

    def test_loss_costs_the_stake(self):
        assert settle("loss", 150) == -1.0

    def test_push_returns_the_stake(self):
        """Grading pushes as losses is a quiet way to kill a live totals market."""
        assert settle("push", -110) == 0.0

    def test_stake_scales(self):
        assert settle("win", 100, stake=2.5) == pytest.approx(2.5)

    def test_unknown_result_raises(self):
        with pytest.raises(ValueError):
            settle("rained_out", -110)


class TestClosingLineValue:
    def test_line_moving_our_way_is_positive_clv(self):
        # Bet home at +120, close at -130: the market moved toward us.
        assert closing_line_value(120, -140, -130, 110) > 0

    def test_line_moving_against_us_is_negative_clv(self):
        assert closing_line_value(-130, 110, 120, -140) < 0

    def test_no_move_is_zero_clv(self):
        assert closing_line_value(-110, -110, -110, -110) == pytest.approx(0.0, abs=1e-9)

    def test_both_sides_are_devigged(self):
        """Regression test for a bug this repo actually shipped and then fixed.

        Comparing a vig-free close to a raw implied bet price subtracts half the
        hold from every bet and reports a large, confident, fake negative CLV.
        An unchanged market must score exactly zero, not minus the vig.
        """
        naive = vig_free_close(-110, -110) - american_to_implied(-110)
        assert naive < -0.02                       # what the bug produced
        assert closing_line_value(-110, -110, -110, -110) == pytest.approx(0.0, abs=1e-9)


class TestSignificance:
    """The screening question: +6% ROI on 300 picks. Is it real?"""

    @staticmethod
    def _record(n=300, roi=0.06, price=-110):
        b = 100 / abs(price)
        wins = round(n * (roi + 1) / (b + 1))
        return np.array([b] * wins + [-1.0] * (n - wins))

    def test_six_percent_on_300_is_not_significant(self):
        r = self._record()
        assert r.mean() == pytest.approx(0.06, abs=0.005)
        assert roi_standard_error(r) == pytest.approx(0.055, abs=0.005)
        assert one_sided_p_value(r) > 0.10          # nowhere near significant
        lo, _hi = roi_bootstrap_ci(r)
        assert lo < 0                                # CI contains zero

    def test_required_sample_size_for_six_percent(self):
        n = required_sample_size(0.06)
        assert 900 < n < 1050                        # ~960 bets, not 300

    def test_smaller_edges_need_far_more_bets(self):
        assert required_sample_size(0.03) > 4 * required_sample_size(0.06)

    def test_same_roi_on_a_big_sample_is_significant(self):
        r = self._record(n=5000)
        assert one_sided_p_value(r) < 0.01
        assert roi_bootstrap_ci(r)[0] > 0


class TestBlockBootstrap:
    def test_correlated_days_widen_the_interval(self):
        """Resampling bets rather than slate-days gives a CI that is too tight."""
        rng = np.random.default_rng(0)
        # 100 slate-days, 10 correlated bets each: the whole day wins or loses.
        day_outcomes = rng.normal(0.06, 0.9, size=100)
        returns = np.repeat(day_outcomes, 10)
        groups = np.repeat(np.arange(100), 10)
        naive_lo, naive_hi = roi_bootstrap_ci(returns, n_boot=3000)
        block_lo, block_hi = block_bootstrap_ci(returns, groups, n_boot=3000)
        assert (block_hi - block_lo) > (naive_hi - naive_lo) * 2


class TestMultipleTesting:
    def test_a_lone_strong_result_survives(self):
        assert benjamini_hochberg({"a": 0.001, "b": 0.6, "c": 0.7}, fdr=0.10)["a"]

    def test_the_luckiest_of_eight_does_not_survive(self):
        p = {f"m{i}": v for i, v in
             enumerate([0.04, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8])}
        assert not benjamini_hochberg(p, fdr=0.10)["m0"]

    def test_nan_p_values_do_not_survive(self):
        out = benjamini_hochberg({"a": float("nan"), "b": 0.001}, fdr=0.10)
        assert out["a"] is False and out["b"] is True

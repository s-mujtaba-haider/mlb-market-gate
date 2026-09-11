import math

import pytest

from mlbgate.odds import (
    american_to_decimal,
    american_to_implied,
    average_american_odds,
    decimal_to_american,
    devig,
    hold,
    implied_to_american,
)


@pytest.mark.parametrize("american,decimal", [
    (100, 2.0), (-100, 2.0), (-110, 1 + 100 / 110), (150, 2.5), (-200, 1.5), (380, 4.8),
])
def test_american_decimal_conversion(american, decimal):
    assert american_to_decimal(american) == pytest.approx(decimal)


@pytest.mark.parametrize("american", [-500, -200, -110, -105, 100, 110, 150, 380, 900])
def test_conversion_round_trips(american):
    assert decimal_to_american(american_to_decimal(american)) == pytest.approx(american)
    assert implied_to_american(american_to_implied(american)) == pytest.approx(american)


@pytest.mark.parametrize("bad", [0, 50, -50, 99.9, -99.9])
def test_dead_zone_is_rejected(bad):
    """There is no American price strictly between -100 and +100."""
    with pytest.raises(ValueError):
        american_to_decimal(bad)


def test_minus_110_implied_and_hold():
    assert american_to_implied(-110) == pytest.approx(0.5238, abs=1e-4)
    assert hold([-110, -110]) == pytest.approx(0.0476, abs=1e-4)


class TestAveraging:
    """You cannot average American odds. These are the three ways people try."""

    def test_naive_integer_mean_is_meaningless(self):
        # Both of these pairs average to 0, which is not a price, and they
        # describe completely different markets.
        assert (-110 + 110) / 2 == 0
        assert (-200 + 200) / 2 == 0
        assert average_american_odds([-110, 110]) == pytest.approx(100)
        assert average_american_odds([-200, 200]) == pytest.approx(100)

    def test_decimal_space_mean_is_biased_toward_the_longshot(self):
        # mean(1.5, 3.0) = 2.25 -> +125, but the true probability average is 50%.
        naive_decimal = (american_to_decimal(-200) + american_to_decimal(200)) / 2
        assert decimal_to_american(naive_decimal) == pytest.approx(125)
        assert average_american_odds([-200, 200]) == pytest.approx(100)

    def test_probability_space_mean_is_correct(self):
        avg = average_american_odds([-105, -115])
        p = american_to_implied(avg)
        assert p == pytest.approx(
            (american_to_implied(-105) + american_to_implied(-115)) / 2)

    def test_weighted_average(self):
        # All weight on one book returns that book's price.
        assert average_american_odds([-110, 200], [1.0, 0.0]) == pytest.approx(-110)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            average_american_odds([])


class TestDevig:
    @pytest.mark.parametrize("method", ["multiplicative", "additive", "power", "shin"])
    def test_probabilities_sum_to_one(self, method):
        for market in ([-110, -110], [-500, 380], [-150, 130], [200, -240]):
            probs = devig(market, method=method)
            assert sum(probs) == pytest.approx(1.0)
            assert all(0 < p < 1 for p in probs)

    @pytest.mark.parametrize("method", ["multiplicative", "additive", "power", "shin"])
    def test_fair_market_is_left_alone(self, method):
        assert devig([100, 100], method=method)[0] == pytest.approx(0.5)

    def test_methods_agree_on_a_balanced_market(self):
        results = [devig([-110, -110], m)[0]
                   for m in ("multiplicative", "additive", "power", "shin")]
        assert all(r == pytest.approx(0.5, abs=1e-6) for r in results)

    def test_methods_diverge_on_longshots(self):
        """Where the de-vig choice actually costs money."""
        mult = devig([-500, 380], "multiplicative")[0]
        shin = devig([-500, 380], "shin")[0]
        # Shin assigns the favourite more probability than proportional de-vig,
        # because it attributes the extra margin to the longshot side.
        assert shin > mult
        assert shin - mult > 0.005

    def test_devig_removes_the_hold(self):
        raw = sum(american_to_implied(p) for p in (-110, -110))
        assert raw > 1.0
        assert sum(devig([-110, -110])) == pytest.approx(1.0)

    def test_single_side_cannot_be_devigged(self):
        with pytest.raises(ValueError):
            devig([-110])

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError):
            devig([-110, -110], method="vibes")


def test_three_way_market():
    probs = devig([160, 230, 240], method="shin")
    assert sum(probs) == pytest.approx(1.0)
    assert probs[0] > probs[1] > probs[2]
    assert not any(math.isnan(p) for p in probs)

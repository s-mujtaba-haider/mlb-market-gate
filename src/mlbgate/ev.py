"""Expected value, staking, settlement and closing line value."""

from __future__ import annotations

import numpy as np

from .odds import american_to_decimal, devig

__all__ = [
    "expected_value",
    "kelly_fraction",
    "settle",
    "closing_line_value",
    "vig_free_close",
]


def expected_value(p_model: float, american: float) -> float:
    """EV per unit staked, against the price actually on offer.

    Two different numbers get called "the market's probability" and they sit
    about half the hold apart (2.4 points at -110):

      raw implied   1/decimal, 52.38% at -110. The break-even rate, and what
                    p_model has to beat for this function to return a profit.
      vig-free      50.00% at -110. The market's honest opinion, and the right
                    benchmark for asking whether a model knows something the
                    market does not (see `gate.beats_vig_free_market`).

    The expensive bug is computing `p_model - p_fair`, calling it edge, and
    betting on it. That treats the gap between the two numbers as profit, so a
    model with a full point of "edge over fair" at -110 is still -EV. Both
    numbers are needed, for different questions; only the raw implied one
    belongs in this calculation, and it is already baked into `american`.
    """
    dec = american_to_decimal(american)
    return p_model * (dec - 1.0) - (1.0 - p_model)


def kelly_fraction(p_model: float, american: float, cap: float = 1.0) -> float:
    """Full-Kelly stake fraction, floored at 0 and capped.

    Kelly is the growth-optimal stake *given that p_model is correct*. It is not
    robust to estimation error: a model 2 points overconfident sizes wildly, so
    production staking here uses a fraction of this (see KELLY_FRACTION).
    """
    b = american_to_decimal(american) - 1.0
    if b <= 0:
        return 0.0
    f = (p_model * b - (1.0 - p_model)) / b
    return float(np.clip(f, 0.0, cap))


def settle(result: str, american: float, stake: float = 1.0) -> float:
    """Profit/loss on a settled bet. Pushes return 0, not a loss.

    Mis-grading pushes as losses is one of the quieter ways a backtest goes
    wrong: it depresses ROI on totals and run lines and makes a live edge look
    dead, and it inflates the denominator used for significance testing.
    """
    if result == "win":
        return stake * (american_to_decimal(american) - 1.0)
    if result == "loss":
        return -stake
    if result == "push":
        return 0.0
    raise ValueError(f"unknown result {result!r}")


def vig_free_close(close_price: float, close_price_opp: float, method: str = "shin") -> float:
    """Fair closing probability for a side, de-vigged against its opposite."""
    return devig([close_price, close_price_opp], method=method)[0]


def closing_line_value(bet_price: float, bet_price_opp: float,
                       close_price: float, close_price_opp: float,
                       method: str = "shin") -> float:
    """CLV in probability points: fair close probability minus fair bet price.

    BOTH sides are de-vigged before comparing. Measuring a vig-free closing
    probability against the raw implied probability of the price you bet is the
    same category error as `expected_value` against a vigged number, except it
    lands with the opposite sign: it subtracts roughly half the hold from every
    bet and reports a large, confident, entirely artificial negative CLV. If a
    strategy shows CLV of -2 to -3 points on every market at once, this is
    almost always the reason rather than a genuinely bad strategy.

    Positive CLV means we took a price the market later moved past. Over a few
    hundred bets it is a far lower-variance estimate of edge than ROI, because
    it is measured on every bet rather than only through the noise of outcomes.
    Real edge with negative CLV is usually a grading bug, a stale line, or a
    backtest quoting prices that were never actually available.
    """
    fair_close = vig_free_close(close_price, close_price_opp, method=method)
    fair_bet = devig([bet_price, bet_price_opp], method=method)[0]
    return fair_close - fair_bet

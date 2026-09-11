"""American/decimal odds conversion and de-vigging.

The three things that get modelled wrong most often in betting code, and that
this module exists to make impossible to get wrong:

1. American odds are not a linear scale. -110 and +110 are adjacent prices but
   the integers are 220 apart, and there is no price between -100 and +100.
   Arithmetic on them is meaningless. See `average_american_odds`.
2. Raw implied probability is not a probability. A two-way market prices to
   more than 100%; the excess is the vig. Betting `p_model > p_implied` without
   removing vig systematically overstates edge by roughly half the hold.
3. There is more than one way to remove vig and they disagree, most on
   longshots. Multiplicative de-vig assumes the book's margin is proportional
   to price, which is empirically wrong: books load more margin onto longshots
   (favourite-longshot bias). Shin de-vig models that directly.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

__all__ = [
    "american_to_decimal",
    "decimal_to_american",
    "american_to_implied",
    "implied_to_american",
    "average_american_odds",
    "devig",
    "hold",
]

_EPS = 1e-12


def american_to_decimal(american: float) -> float:
    """Convert American odds to decimal (total return per unit staked)."""
    if -100 < american < 100:
        raise ValueError(f"{american} is not a valid American price (-100 < x < 100 is empty)")
    if american > 0:
        return 1.0 + american / 100.0
    return 1.0 + 100.0 / abs(american)


def decimal_to_american(decimal: float) -> float:
    """Convert decimal odds to American odds."""
    if decimal <= 1.0:
        raise ValueError(f"decimal odds must exceed 1.0, got {decimal}")
    if decimal >= 2.0:
        return (decimal - 1.0) * 100.0
    return -100.0 / (decimal - 1.0)


def american_to_implied(american: float) -> float:
    """Raw implied probability. NOTE: includes vig. Not a fair probability."""
    return 1.0 / american_to_decimal(american)


def implied_to_american(prob: float) -> float:
    """Inverse of `american_to_implied`."""
    if not 0.0 < prob < 1.0:
        raise ValueError(f"probability must be in (0, 1), got {prob}")
    return decimal_to_american(1.0 / prob)


def average_american_odds(prices: Sequence[float], weights: Sequence[float] | None = None) -> float:
    """Average a set of American prices *correctly*, via probability space.

    Averaging the American integers is always wrong, and wrong in a way that
    looks plausible: mean(-110, +110) = 0, which is not a price at all, and
    mean(-200, +200) = 0 again, even though those two pairs describe completely
    different markets.

    Averaging in *decimal* space is also wrong, just less obviously. Decimal
    odds are 1/p, and the mean of reciprocals is not the reciprocal of the mean
    (Jensen's inequality), so a decimal average is biased toward the longshot.
    mean(1.50, 3.00) = 2.25 -> +125, but the two prices imply 66.7% and 33.3%,
    whose true average is 50.0% -> +100.

    Consensus lines are averages of *probabilities*, so that is what we do here.
    For a true consensus you should de-vig each book first, then average.
    """
    if not prices:
        raise ValueError("no prices to average")
    probs = [american_to_implied(p) for p in prices]
    if weights is None:
        mean_p = sum(probs) / len(probs)
    else:
        if len(weights) != len(probs):
            raise ValueError("weights and prices must be the same length")
        total_w = sum(weights)
        if total_w <= 0:
            raise ValueError("weights must sum to a positive number")
        mean_p = sum(p * w for p, w in zip(probs, weights)) / total_w
    return implied_to_american(mean_p)


def hold(prices: Sequence[float]) -> float:
    """Book hold (overround) for a complete market, as a fraction.

    A -110/-110 two-way market holds ~4.55% of handle in the long run.
    """
    return sum(american_to_implied(p) for p in prices) - 1.0


def _devig_multiplicative(q: list[float]) -> list[float]:
    total = sum(q)
    return [x / total for x in q]


def _devig_additive(q: list[float]) -> list[float]:
    excess = (sum(q) - 1.0) / len(q)
    out = [x - excess for x in q]
    if any(x <= 0 for x in out):
        # Additive de-vig can drive a heavy longshot non-positive. Fall back
        # rather than emit an impossible probability.
        return _devig_multiplicative(q)
    return out


def _devig_power(q: list[float], tol: float = 1e-12, max_iter: int = 200) -> list[float]:
    """Find k such that sum(q_i ** k) == 1. k > 1 for an overround market."""
    lo, hi = 1.0, 4.0
    while sum(x**hi for x in q) > 1.0 and hi < 64.0:
        hi *= 2.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        s = sum(x**mid for x in q)
        if abs(s - 1.0) < tol:
            break
        if s > 1.0:
            lo = mid
        else:
            hi = mid
    k = 0.5 * (lo + hi)
    out = [x**k for x in q]
    total = sum(out)
    return [x / total for x in out]


def _shin_probs(q: list[float], z: float) -> list[float]:
    qsum = sum(q)
    out = []
    for x in q:
        disc = z * z + 4.0 * (1.0 - z) * x * x / qsum
        out.append((math.sqrt(max(disc, 0.0)) - z) / (2.0 * (1.0 - z)))
    return out


def _devig_shin(q: list[float], tol: float = 1e-12, max_iter: int = 200) -> list[float]:
    """Shin (1993): treat the overround as compensation for insider trading.

    `z` is the implied fraction of informed money. Recovers more probability
    from favourites and less from longshots than multiplicative de-vig, which
    matches how books actually price. Reduces to multiplicative as z -> 0.
    """
    if sum(q) <= 1.0 + _EPS:
        return _devig_multiplicative(q)
    lo, hi = 0.0, 0.99
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        s = sum(_shin_probs(q, mid))
        if abs(s - 1.0) < tol:
            break
        if s > 1.0:
            lo = mid
        else:
            hi = mid
    probs = _shin_probs(q, 0.5 * (lo + hi))
    total = sum(probs)
    return [p / total for p in probs]


_DEVIG_METHODS = {
    "multiplicative": _devig_multiplicative,
    "additive": _devig_additive,
    "power": _devig_power,
    "shin": _devig_shin,
}


def devig(prices: Sequence[float], method: str = "shin") -> list[float]:
    """Strip vig from a complete market, returning fair probabilities.

    `prices` must be every outcome of the market (both sides of a total, all
    three of a 3-way). De-vigging a single side is not defined: you cannot know
    how much margin sits on a price you cannot see.
    """
    if len(prices) < 2:
        raise ValueError("de-vig needs a complete market (>= 2 outcomes)")
    if method not in _DEVIG_METHODS:
        raise ValueError(f"unknown de-vig method {method!r}; options: {sorted(_DEVIG_METHODS)}")
    q = [american_to_implied(p) for p in prices]
    probs = _DEVIG_METHODS[method](q)
    total = sum(probs)
    return [p / total for p in probs]

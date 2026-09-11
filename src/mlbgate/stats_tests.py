"""Significance machinery for betting records.

A betting P&L curve breaks most of the assumptions of a t-test taken off the
shelf: returns are bimodal rather than normal, bets placed on the same game or
the same day are correlated, and when you test eight markets at once the best
one is selected *because* it got lucky. Each function here addresses one of
those.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .odds import american_to_decimal

__all__ = [
    "roi_standard_error",
    "roi_bootstrap_ci",
    "block_bootstrap_ci",
    "required_sample_size",
    "benjamini_hochberg",
    "one_sided_p_value",
]

RNG_SEED = 20260911


def one_sided_p_value(returns: np.ndarray) -> float:
    """P(observe ROI this good | true edge is zero), normal approximation."""
    returns = np.asarray(returns, dtype=float)
    n = len(returns)
    if n < 2:
        return float("nan")
    se = returns.std(ddof=1) / np.sqrt(n)
    if se == 0:
        return float("nan")
    from math import erfc, sqrt
    t = returns.mean() / se
    return float(0.5 * erfc(t / sqrt(2.0)))


def roi_standard_error(returns: np.ndarray) -> float:
    returns = np.asarray(returns, dtype=float)
    return float(returns.std(ddof=1) / np.sqrt(len(returns)))


def roi_bootstrap_ci(returns: np.ndarray, alpha: float = 0.05, n_boot: int = 20_000,
                     seed: int = RNG_SEED) -> tuple[float, float]:
    """Percentile bootstrap CI for mean return per unit staked (ROI)."""
    returns = np.asarray(returns, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(returns), size=(n_boot, len(returns)))
    means = returns[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def block_bootstrap_ci(returns: np.ndarray, groups: np.ndarray, alpha: float = 0.05,
                       n_boot: int = 20_000, seed: int = RNG_SEED) -> tuple[float, float]:
    """Bootstrap that resamples whole *groups* (e.g. slate-days), not bets.

    Bets on the same day share a starting pitcher pool, weather and a common
    market state, so they are not independent draws. Resampling individual bets
    understates the true variance and produces a CI that is too tight, which is
    how a dead strategy passes a naive significance check.
    """
    returns = np.asarray(returns, dtype=float)
    groups = np.asarray(groups)
    uniq = pd.unique(groups)
    by_group = {g: returns[groups == g] for g in uniq}
    sizes = np.array([len(by_group[g]) for g in uniq])
    sums = np.array([by_group[g].sum() for g in uniq])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    boot_sum = sums[idx].sum(axis=1)
    boot_n = sizes[idx].sum(axis=1)
    means = boot_sum / np.maximum(boot_n, 1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def required_sample_size(roi_target: float, price: float = -110.0, z: float = 1.96) -> int:
    """Bets needed for `roi_target` to sit z standard errors above zero.

    Assumes flat stakes at a single price and that the edge is real and constant.
    This is a floor, not an estimate: correlated bets and varying prices push the
    real requirement higher.
    """
    dec = american_to_decimal(price)
    b = dec - 1.0
    # Win rate implying the target ROI at this price.
    win_rate = (roi_target + 1.0) / (b + 1.0)
    var = win_rate * b**2 + (1 - win_rate) * 1.0 - roi_target**2
    return int(np.ceil((z * np.sqrt(var) / roi_target) ** 2))


def benjamini_hochberg(p_values: dict[str, float], fdr: float = 0.10) -> dict[str, bool]:
    """Benjamini-Hochberg step-up. Returns {key: survives_correction}.

    Running eight markets through a gate and shipping whichever clears p < 0.05
    means roughly one false pass every two full passes of the slate. Controlling
    false discovery rate across the family is the difference between "this market
    is good" and "this market was the luckiest of eight".
    """
    items = [(k, v) for k, v in p_values.items() if not np.isnan(v)]
    if not items:
        return {k: False for k in p_values}
    items.sort(key=lambda kv: kv[1])
    m = len(items)
    survives = {k: False for k in p_values}
    k_max = 0
    for i, (_, p) in enumerate(items, start=1):
        if p <= fdr * i / m:
            k_max = i
    for i, (key, _) in enumerate(items, start=1):
        survives[key] = i <= k_max
    return survives

"""A model shows +6% ROI on 300 picks. What do you check before believing it?

This script answers the arithmetic half of that question. The other half --
leakage, price availability, grading, multiple testing -- is what the gate in
`02_run_gate.py` does, and what the eight markets in this repo demonstrate.

Run: python scripts/03_roi_significance.py
"""
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlbgate.odds import american_to_decimal, american_to_implied  # noqa: E402
from mlbgate.stats_tests import (  # noqa: E402
    one_sided_p_value,
    required_sample_size,
    roi_bootstrap_ci,
    roi_standard_error,
)

N_PICKS = 300
ROI = 0.06
PRICE = -110


def synthetic_record(n: int, roi: float, price: float) -> np.ndarray:
    """The flat-staked record that would produce exactly this ROI at this price."""
    b = american_to_decimal(price) - 1.0
    wins = round(n * (roi + 1.0) / (b + 1.0))
    return np.array([b] * wins + [-1.0] * (n - wins))


def main() -> None:
    b = american_to_decimal(PRICE) - 1.0
    breakeven = american_to_implied(PRICE)
    r = synthetic_record(N_PICKS, ROI, PRICE)
    wins = int((r > 0).sum())
    se = roi_standard_error(r)
    t = r.mean() / se
    p = one_sided_p_value(r)
    lo, hi = roi_bootstrap_ci(r)

    print(f"Claim: {ROI:+.0%} ROI on {N_PICKS} picks at {PRICE:+.0f}\n")
    print("1. Is it even distinguishable from zero?")
    print(f"   record needed        : {wins}-{N_PICKS - wins} "
          f"({wins / N_PICKS:.1%} win rate)")
    print(f"   break-even at {PRICE:+.0f}  : {breakeven:.1%}")
    print(f"   edge over break-even : {wins / N_PICKS - breakeven:+.1%} "
          f"({(wins / N_PICKS - breakeven) * N_PICKS:+.1f} wins)")
    print(f"   per-bet std dev      : {r.std(ddof=1):.3f}")
    print(f"   standard error       : {se:.3%}")
    print(f"   t-statistic          : {t:.2f}")
    print(f"   one-sided p-value    : {p:.3f}")
    print(f"   95% bootstrap CI     : [{lo:+.2%}, {hi:+.2%}]")
    print(f"   -> the whole result is {t:.1f} standard errors from zero, and "
          f"the interval contains it.")
    print(f"   -> {abs(round((wins / N_PICKS - breakeven) * N_PICKS))} coin flips "
          f"going the other way erases the entire edge.\n")

    print("2. How much evidence would it take?")
    for target in (0.10, 0.06, 0.03, 0.02):
        print(f"   a real {target:+.0%} edge needs "
              f"{required_sample_size(target):>7,} bets to clear two standard errors")
    print(f"   -> at {ROI:+.0%}, {required_sample_size(ROI):,} picks, not {N_PICKS}. "
          f"This sample is {required_sample_size(ROI) / N_PICKS:.1f}x too small.\n")

    print("3. What else would move my belief, in priority order?")
    for i, line in enumerate([
        "CLV. Mean closing line value over the same 300 bets is a far lower-"
        "variance edge estimate than ROI, because it is measured on every bet "
        "rather than through the noise of outcomes. Positive CLV on an "
        "insignificant ROI means keep collecting. Negative CLV means the ROI "
        "is noise regardless of what the p-value says.",
        "Were those prices real? Point-in-time odds at bet timestamp, at a book "
        "and a limit actually reachable. Backtests that quote openers, closers "
        "or a best-of-N line across books invent most of a 6% edge on their own.",
        "Leakage. Every feature's as-of time at or before the decision time, and "
        "the EV threshold and calibration fitted on train folds only. Choosing "
        "the bet threshold on the test set is the quiet one.",
        "Grading. Pushes graded as pushes, not losses. Correct settlement on "
        "rain-shortened and suspended games.",
        "Selection. How many markets, thresholds and parameter sets were tried "
        "to arrive at this one? 300 picks is one season of one market; if it is "
        "the best of eight, the effective p-value is far worse than 0.13.",
        "Concentration. Strip the best 10 bets and the best single week. If the "
        "edge is carried by a handful of longshots or one hot fortnight, it is a "
        "variance story, not an edge story.",
    ], start=1):
        print(f"   {i}. {line}")
    print()

    print("4. Verdict")
    print(f"   Not yet believable, and not yet disproved. {ROI:+.0%} on {N_PICKS} "
          f"picks is the expected spread of a break-even bettor, so the record is")
    print("   consistent with both a real edge and no edge at all. CLV and the "
          "leakage audit decide it long before the ROI number can.")


if __name__ == "__main__":
    main()

# MLB market validation gate

[![tests](https://github.com/s-mujtaba-haider/mlb-market-gate/actions/workflows/tests.yml/badge.svg)](https://github.com/s-mujtaba-haider/mlb-market-gate/actions/workflows/tests.yml)

A backtesting harness that decides whether a betting market is good enough to
ship, and refuses to be fooled by the usual ways a backtest lies.

Eight MLB markets go in. Each comes out with **PASS**, **FAIL** or **VETO**, a
named cause, and a specific improvement lever. Two pass.

```
PASS  rl_home_-1.5           n=1852  roi= +9.07%  CI [+3.54%, +14.45%]  CLV t=36.5
PASS  f5_ml_home             n=1749  roi=+10.32%  CI [+6.86%, +14.13%]  CLV t=49.5
FAIL  ml_home                n=2412  roi= -0.20%  no edge over the vig-free price
FAIL  total_over             n=2144  roi= -1.44%  no edge over the vig-free price
FAIL  nrfi                   n= 182  roi= -1.04%  fires too rarely to prove anything
FAIL  pitcher_ks_over        n=1842  roi= +7.42%  edge died in 2023, one fold carries it
FAIL  batter_hits_over_0.5   n=3713  roi= +5.13%  ECE 0.051, probabilities are stale
VETO  team_total_over_home   n=2014  roi=+110.19% post-decision feature. Not a result.
```

That last line is the point of the whole repo. A market showing +110% ROI,
Brier skill of +0.66 and a bootstrap CI nowhere near zero gets **vetoed**,
because one of its features carries an as-of timestamp twelve hours after the
pick fires. Every statistical check agrees it is brilliant. It is worthless.

---

## The honest bit, up front

**The data in this repo is generated, not real.** The machine this was built on
had no reachable sports data API and no odds vendor, so neither a Stats API
pull nor a price backfill was possible.

Rather than hand-wave that, I turned it into the design:
`src/mlbgate/simulate.py` generates each market with a **known pathology** —
two with real exploitable edge, two priced efficiently, one too thin to measure,
one whose edge is arbitraged away mid-sample, one whose calibration drifts, one
with planted leakage. The gate never sees the ground truth. `tests/test_gate.py`
asserts that the gate reached the right verdict **for the right reason**.

That is a stronger claim than a scrape would support. A validator run only on
data where nobody knows the right answer has not been validated — you are just
trusting a second model. Here the validator has a test suite.

The real path is written and ready: `src/mlbgate/ingest_statsapi.py` pulls
schedules and linescores from the MLB Stats API with on-disk caching, and
derives settlement for moneyline, run line, first-five, NRFI and team totals —
including the cases that quietly corrupt a backfill (pushes, rain-shortened
games voiding F5, walk-offs, doubleheaders sharing a date and both team names).
It is covered by fixture tests. It raises a clear error offline instead of
pretending. Odds are the part that needs a vendor, and
`ingest_statsapi.ODDS_NOTE` says exactly what to buy and what join to use.

---

## The gate

Eleven checks, run in priority order. The first failure becomes the reported
cause, because the causes are not independent — a leaking market also looks
beautifully calibrated and wildly significant, and printing "significant" next
to "leaking" is how bad markets get shipped.

| # | Check | Bar | What it stops |
|--:|---|---|---|
| 1 | Feature as-of audit | no feature timestamped after decision time | **Hard veto.** A feature that did not exist at pick time cannot be used at pick time. |
| 2 | Target shuffle | OOS AUC on globally permuted labels ≤ 0.53 | **Hard veto.** Pre-split target encoding, a calibrator fitted on test labels, the same game in both folds. |
| 3 | Grading completeness | ≥ 99% resolved, pushes graded as pushes | Pushes scored as losses quietly kill live totals and run-line markets. |
| 4 | Sample | ≥ 300 OOS picks across ≥ 2 seasons | Below this, ROI cannot be separated from variance. |
| 5 | Beats the vig-free market | Brier skill > 0 vs the de-vigged price | The benchmark is the market, not a coin flip. Most models clear the coin flip and lose to the price. |
| 6 | Calibration | ECE ≤ 0.035 | Miscalibrated probabilities break every EV number and Kelly stake downstream. |
| 7 | CLV | mean CLV t ≥ 1.64 | Lower-variance evidence of edge than ROI, measured on every bet rather than through outcome noise. |
| 8 | ROI significance | day-blocked bootstrap 95% CI excludes zero | Bets on one slate share a pitcher pool and a market state. Resampling bets instead of days gives a CI that is too tight. |
| 9 | Multiple testing | survives Benjamini–Hochberg at FDR 10% | Eight markets tested and the best one shipped is a false pass every other slate. |
| 10 | Fold stability | ≥ 60% of folds profitable, no fold > 80% of P&L | Separates a live edge from one that has been arbitraged away. |
| 11 | Price haircut | ROI > 0 at 10¢ worse prices | The backtest's price is not the price you get at real limits. |

Leakage control is structural, not a single check:

- **Expanding-window walk-forward by season.** A model is only ever scored on
  seasons that did not exist when it was fit. Four seasons, three OOS folds.
- **3-day embargo** before each test fold, because rolling features computed
  near the boundary see into the next one.
- **EV threshold fitted on the train fold.** The bet cutoff is a fitted
  parameter; choosing it to maximise test-set profit is the single most common
  silent leak in betting backtests, and it never shows up in a feature audit.
- **Calibration fitted on the train fold**, never refit on test.
- **Disjointness assertion** — `LeakageError` if a `game_id` lands on both sides.

---

## Odds math

`src/mlbgate/odds.py`, with `tests/test_odds.py` as the argument.

**You cannot average American odds.** Three ways people try:

| Method | -110 and +110 | -200 and +200 | |
|---|---|---|---|
| Mean the integers | `0` | `0` | not a price; two different markets collapse to the same answer |
| Mean the decimals | +100 | **+125** | biased toward the longshot — decimals are `1/p`, and the mean of reciprocals is not the reciprocal of the mean |
| Mean the probabilities | +100 | **+100** | correct (66.7% and 33.3% average to 50%) |

For a real consensus line you de-vig each book *first*, then average the fair
probabilities.

**De-vigging is a modelling choice, not a formula.** Four methods on a
-500/+380 market:

| Method | Favourite | |
|---|--:|---|
| Multiplicative | 80.00% | assumes margin is proportional to price |
| Additive | 81.25% | |
| Power | 81.95% | |
| **Shin** | 81.25% | models margin as compensation for informed money |

They agree exactly at -110/-110 and diverge on longshots, which is where the
money is — books load more margin onto the longshot side, so multiplicative
de-vig systematically understates favourites. Shin is the default here.

**Raw implied is not fair.** At -110 there are two different "market numbers"
2.4 points apart, and conflating them is the most expensive bug in betting code:

- **raw implied, 52.38%** — the break-even rate. Beat this to have +EV.
- **vig-free, 50.00%** — the market's honest opinion. Beat this to have
  information the market lacks.

Reporting `p_model − p_fair` as "edge" and betting on it pockets the half-hold
gap as if it were profit. A full point of edge over fair is still a losing bet.

---

## "A model shows +6% ROI on 300 picks. What do you check?"

`python scripts/03_roi_significance.py` — the arithmetic, computed rather than
asserted:

```
record needed        : 167-133 (55.7% win rate)
break-even at -110   : 52.4%
edge over break-even : +3.3%  (+9.9 wins)
standard error       : 5.485%
t-statistic          : 1.14
one-sided p-value    : 0.126
95% bootstrap CI     : [-4.55%, +17.09%]

a real +10% edge needs     342 bets to clear two standard errors
a real  +6% edge needs     961 bets
a real  +3% edge needs   3,865 bets
```

The entire result is **1.1 standard errors from zero**. Ten coin flips going the
other way erase it completely. 300 picks is roughly a third of the sample that
edge would need.

So the ROI number cannot settle it, and the things that can are all upstream:
CLV over the same 300 bets, whether those prices were actually available at the
bet timestamp, whether the feature as-of times and the EV threshold are clean,
whether pushes were graded as pushes, and how many markets and thresholds were
tried before this one was shown to me.

---

## Bugs this harness caught in itself

Worth listing, because they are all bugs I would expect to find in a live
platform, and finding them is the job:

1. **CLV compared a vig-free close to a vigged bet price.** Subtracted half the
   hold from every bet and reported a large, confident CLV of −14 to −58 t across
   *every* market simultaneously. That uniformity is the tell: a strategy can be
   bad, but not identically bad in eight independent markets. Fixed by de-vigging
   both sides; regression-tested by asserting an unchanged market scores exactly
   zero, not minus the vig.
2. **The price haircut walked into the ±100 dead zone.** Worsening +105 by 10¢
   produced "+95", which is not a price. The odds validator raised rather than
   silently continuing. The correct answer is −105: the haircut has to step
   across the gap.
3. **The target-shuffle test could never fire.** It permuted train and test
   labels independently, which returns ~0.5 on every dataset, leaking or not —
   a check that looks reassuring while testing nothing. Fixed to permute once
   globally, and given a `rebuild` hook so feature construction re-runs on the
   permuted labels: *a permutation test that starts downstream of feature
   construction cannot see leakage introduced during feature construction.*
   `tests/test_leakage_guard.py` now contains both a true positive (pre-split
   target encoding) and a demonstration that the check goes blind without the
   hook.

---

## Run it

```bash
pip install -r requirements.txt

python scripts/01_build_dataset.py      # 77k rows, 8 markets, 4 seasons
python scripts/02_run_gate.py           # -> reports/market_report.md
python scripts/03_roi_significance.py   # the +6% question
python -m pytest tests/ -q              # 115 tests, ~6s
```

Output lands in `reports/market_report.md` (reviewable) and
`reports/gate_results.csv` (machine-readable, one row per market with every
metric, failed check, cause and lever).

## Layout

```
src/mlbgate/
  odds.py              American/decimal/implied, averaging, 4 de-vig methods
  ev.py                EV, Kelly, settlement, CLV
  stats_tests.py       bootstrap, day-blocked bootstrap, BH, sample sizing
  model.py             purged walk-forward, train-fold thresholds, shuffle test
  gate.py              the 11 checks, priority order, cause + lever per failure
  schema.py            market registry, feature manifest, ground truth
  simulate.py          generator with known pathologies
  ingest_statsapi.py   real MLB Stats API path + settlement derivation
  report.py            markdown + CSV rendering
tests/                 115 tests
```

## Mapping to MLB Phase 1

This is a standalone harness, not a port of anyone's platform, but it is built
around the same spine as the Phase 1 brief: backfill and grade, validate every
market through a gate with no-leakage integrity, an honest pass or fail per
market, a named cause and a practical lever for every failure, ship the winners
and veto-filter the rest. `gate_results.csv` is the artifact that drives the
ship/kill decision; `market_report.md` is the artifact a reviewer reads.

Swap `simulate.py` for `ingest_statsapi.py` plus an odds vendor and the rest of
the pipeline is unchanged — which is the reason the data layer is behind a
seam in the first place.

---

MIT licensed. See `LICENSE`.

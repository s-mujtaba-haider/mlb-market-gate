# MLB market gate - results

`2` of `8` markets cleared the gate. Every number below is out-of-sample, from expanding-window walk-forward folds with a 3-day embargo. No market is judged on data that existed when its model was fit.

## Summary

| Market | Verdict | Picks | ROI | 95% CI | CLV t | Brier skill | ECE |
|---|---|--:|--:|:--:|--:|--:|--:|
| Moneyline - home | FAIL | 2412 | -0.20% | [-4.98%, +4.85%] | 14.99 | -0.0164 | 0.018 |
| Run line - home -1.5 | **PASS** | 1852 | +9.07% | [+3.54%, +14.45%] | 36.48 | 0.0014 | 0.017 |
| Game total - over | FAIL | 2144 | -1.44% | [-6.58%, +3.47%] | 17.20 | -0.0137 | 0.014 |
| First 5 innings - home ML | **PASS** | 1749 | +10.32% | [+6.86%, +14.13%] | 49.47 | 0.0125 | 0.019 |
| No runs first inning | FAIL | 182 | -1.04% | [-15.20%, +13.83%] | 10.79 | 0.0015 | 0.010 |
| Home team total - over | **VETO** | 2014 | +110.19% | [+105.26%, +115.14%] | 3.98 | 0.6594 | 0.008 |
| Starting pitcher strikeouts - over | FAIL | 1842 | +7.42% | [+0.78%, +14.02%] | 31.73 | 0.0097 | 0.012 |
| Batter to record a hit | FAIL | 3713 | +5.13% | [+2.59%, +7.52%] | 49.76 | 0.0201 | 0.051 |

## Gate policy

A market ships only if it clears every check. They run in priority order and the first failure is reported as the primary cause, because the causes are not independent: a leaking market also looks beautifully calibrated and highly significant.

| # | Check | Bar | Why it is in the gate |
|--:|---|---|---|
| 1 | Feature as-of audit | no feature timestamped after decision time | A feature that did not exist at pick time cannot be used at pick time. Hard veto. |
| 2 | Target shuffle | OOS AUC on randomised labels <= 0.53 | Catches structural leaks no manifest can see. Hard veto. |
| 3 | Grading completeness | >= 99% resolved, pushes graded as pushes | Pushes scored as losses quietly kill live totals and run-line markets. |
| 4 | Sample | >= 300 OOS picks across >= 2 seasons | Below this, ROI cannot be separated from variance. |
| 5 | Beats the vig-free market | Brier skill > 0 vs the de-vigged price | The benchmark is the market, not a coin flip. |
| 6 | Calibration | ECE <= 0.035 | Miscalibrated probabilities break every EV number and stake downstream. |
| 7 | CLV | mean CLV t >= 1.64 | Lower-variance evidence of edge than ROI, measured on every bet. |
| 8 | ROI significance | day-blocked bootstrap 95% CI excludes zero | Bets on one slate are correlated; resampling bets rather than days gives a CI that is too tight. |
| 9 | Multiple testing | survives Benjamini-Hochberg at FDR 10% | Eight markets tested and the best one shipped is a false pass every other slate. |
| 10 | Fold stability | >= 60% of folds profitable, no fold > 80% of P&L | Separates a live edge from one that has been arbitraged away. |
| 11 | Price haircut | ROI > 0 at 10c worse prices | The backtest's price is not the price you get at real limits. |

## Per-market findings

### Moneyline - home - FAIL

`ml_home` | 2412 picks on 7,194 scored games (33.5% fire rate) | ROI -0.20% (SE 2.6%) | AUC 0.705 | mean CLV +0.69%

**Cause.** Brier skill against the vig-free price is -0.0164. The model carries no information the market has not already priced; it beats a coin flip, which is not the benchmark.

**Lever.** This needs a genuinely new input, not a better fit on the same ones. Highest-value candidates: posted-lineup and bullpen-availability deltas inside the final hour, park-adjusted Statcast quality of contact, and umpire assignment. If none of those move Brier skill positive, kill the market rather than tuning it.

Also failing: `roi_ci_excludes_zero`, `fold_stability`, `price_haircut_survival`, `roi_bh_corrected`.

### Run line - home -1.5 - **PASS**

`rl_home_-1.5` | 1852 picks on 7,194 scored games (25.7% fire rate) | ROI +9.07% (SE 2.8%) | AUC 0.731 | mean CLV +2.80%

Cleared every check. Ship behind the standard live guards: quarter-Kelly staking, a minimum-EV floor above the price haircut, and a rolling CLV monitor that pulls the market if mean CLV turns negative over a 200-pick window.

### Game total - over - FAIL

`total_over` | 2144 picks on 7,194 scored games (29.8% fire rate) | ROI -1.44% (SE 2.7%) | AUC 0.673 | mean CLV +0.84%

**Cause.** Brier skill against the vig-free price is -0.0137. The model carries no information the market has not already priced; it beats a coin flip, which is not the benchmark.

**Lever.** This needs a genuinely new input, not a better fit on the same ones. Highest-value candidates: posted-lineup and bullpen-availability deltas inside the final hour, park-adjusted Statcast quality of contact, and umpire assignment. If none of those move Brier skill positive, kill the market rather than tuning it.

Also failing: `roi_ci_excludes_zero`, `fold_stability`, `price_haircut_survival`, `roi_bh_corrected`.

### First 5 innings - home ML - **PASS**

`f5_ml_home` | 1749 picks on 7,194 scored games (24.3% fire rate) | ROI +10.32% (SE 1.9%) | AUC 0.718 | mean CLV +4.04%

Cleared every check. Ship behind the standard live guards: quarter-Kelly staking, a minimum-EV floor above the price haircut, and a rolling CLV monitor that pulls the market if mean CLV turns negative over a 200-pick window.

### No runs first inning - FAIL

`nrfi` | 182 picks on 7,194 scored games (2.5% fire rate) | ROI -1.04% (SE 7.2%) | AUC 0.711 | mean CLV +2.54%

**Cause.** Only 182 graded out-of-sample picks. At this price a real +5% edge needs about 1,387 bets to clear two standard errors, so any ROI measured here is indistinguishable from noise.

**Lever.** Backfill more seasons, or widen selection: this market fires on only 2.5% of scored games. Loosen the EV threshold and re-test, or pool it with a structurally correlated market to borrow sample.

Also failing: `roi_ci_excludes_zero`, `fold_stability`, `price_haircut_survival`, `roi_bh_corrected`.

### Home team total - over - **VETO**

`team_total_over_home` | 2014 picks on 7,194 scored games (28.0% fire rate) | ROI +110.19% (SE 2.5%) | AUC 0.963 | mean CLV +0.21%

**Cause.** 1 feature(s) carry an as-of time after the decision timestamp: season_team_ops. The model is reading information that did not exist when the pick fires.

**Lever.** Rebuild these as point-in-time aggregates closed at the decision timestamp (trailing windows ending the prior day), then re-run. Add the as-of assertion to CI so the feature store rejects them at write time rather than the gate catching them at read time.

### Starting pitcher strikeouts - over - FAIL

`pitcher_ks_over` | 1842 picks on 7,194 scored games (25.6% fire rate) | ROI +7.42% (SE 3.3%) | AUC 0.751 | mean CLV +3.09%

**Cause.** Profit is not stable across walk-forward folds (2022: +143.1u, 2023: -40.3u, 2024: +34.0u); the best fold carries 105% of total P&L. That is the signature of an edge that existed in one regime and has since been priced in.

**Lever.** Refit on a trailing window rather than an expanding one and re-gate on the most recent season alone. If the edge is gone it is gone: veto-filter the market instead of averaging a dead recent season against a live old one to manufacture a pass.

### Batter to record a hit - FAIL

`batter_hits_over_0.5` | 3713 picks on 7,194 scored games (51.6% fire rate) | ROI +5.13% (SE 1.2%) | AUC 0.709 | mean CLV +2.93%

**Cause.** Expected calibration error 0.051 against a 0.035 bar. Ranking is fine (AUC 0.709) but the probabilities themselves are off, so every EV number and Kelly stake downstream is wrong by the same margin.

**Lever.** Refit the isotonic calibration on a rolling recent window instead of the full expanding history and carry the season base rate as an offset. The drift here is in the intercept, not the slope: discrimination is intact, only the level is stale.

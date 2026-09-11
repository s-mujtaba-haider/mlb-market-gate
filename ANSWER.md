# Application — MLB Phase 1, Milestone 1

## 2. "A model shows +6% ROI on 300 picks. What do you check before believing it?"

> **First, whether it's even measurable.** At -110 that's a 167-133 record, 55.7%
> against a 52.4% break-even — about 1.1 standard errors from zero (one-sided
> p ≈ 0.13, 95% CI roughly -4.5% to +17%). Ten flips the other way erase the
> entire edge, and a genuine +6% needs ~960 bets to clear two SE. So 300 picks
> can't confirm it or kill it.
>
> **So I'd go to CLV first**, since it's a much lower-variance estimate of edge —
> it's measured on every bet rather than through outcome noise. Alongside that:
> were those prices actually available at bet time, at a reachable limit? A
> backtest quoting closers, openers, or best-of-N across books manufactures most
> of a 6% edge by itself.
>
> **Then leakage.** Every feature's as-of timestamp at or before the decision
> time, and — the quiet one — the EV threshold and the calibration fitted on
> train folds only, never chosen on the test set. Plus pushes graded as pushes,
> not losses.
>
> **Finally selection.** How many markets, thresholds and parameter sets were
> tried to arrive at this one? If it's the best of eight, the effective p-value
> is far worse than 0.13 and needs a family-wise correction before anyone
> believes the 6%.

## 1. Project

**https://github.com/s-mujtaba-haider/mlb-market-gate** — an MLB market
validation gate. Eight markets in,
PASS / FAIL / VETO out, with a named cause and a specific improvement lever for
every failure.

Three things in it that are relevant to Milestone 1:

- **It vetoes a market showing +110% ROI** with a bootstrap CI nowhere near
  zero, because one feature carries an as-of timestamp after the pick fires.
  Every statistical check says it's brilliant; it's worthless. Leakage outranks
  significance in the gate for exactly this reason.
- **The validator itself is tested.** Every market is generated with a known
  pathology — real edge, efficient pricing, too-thin sample, an edge that dies
  mid-sample, calibration drift, planted leakage — and the test suite asserts
  the gate reaches the right verdict *for the right reason*. 115 tests.
- **The odds math is the foundation, not an afterthought**: four de-vig methods
  (Shin by default, since books load margin onto longshots), and a hard
  separation between raw implied (the break-even number) and vig-free (the
  market's opinion) — conflating those two is the most expensive bug in betting
  code.

Data caveat stated up front in the README: it's generated, because the machine
had no reachable sports data API and no odds vendor. The real MLB Stats API
ingestor is written, tested
against response fixtures, and handles the settlement cases that actually
corrupt backfills — pushes, rain-shortened games voiding first-five, walk-offs,
doubleheaders sharing a date and both team names.

---

## Notes before sending

- [ ] The answer is the part they'll read first — it's written to be pasted
      directly into the DM or WhatsApp. Send it as prose, drop the `>` marks.
- [ ] WhatsApp is text only per the posting: 0336 8468248.
- [ ] They asked for 3–4 lines. The four bolded points are the four lines; if
      you want it shorter, the first two carry the answer on their own.

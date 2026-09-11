"""The validation gate: PASS or FAIL per market, with cause and lever.

Checks run in priority order and the first failure is reported as the primary
cause, because the causes are not independent. A market with leakage will also
look brilliantly calibrated and wildly significant; reporting "significant"
next to "leaking" invites someone to ship it. Leakage is a hard veto: it is not
a result, it is the absence of one.

Thresholds live in one place so a reviewer can argue with the policy rather
than reverse-engineer it from the code.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from .stats_tests import (
    benjamini_hochberg,
    block_bootstrap_ci,
    one_sided_p_value,
    required_sample_size,
    roi_standard_error,
)

# ---------------------------------------------------------------- thresholds
MIN_PICKS = 300              # below this, ROI cannot be distinguished from noise
MIN_SEASONS = 2              # one season is one regime, not a sample
MIN_GRADED_FRAC = 0.99       # ungraded rows silently bias ROI
MAX_SHUFFLED_AUC = 0.53      # a shuffled target must be unpredictable
MAX_ECE = 0.035              # expected calibration error
MIN_BRIER_SKILL = 0.0        # vs the vig-free market price
MIN_CLV_T = 1.64             # one-sided 95% on mean closing line value
FDR = 0.10                   # Benjamini-Hochberg across the market family
MIN_PROFITABLE_FOLDS = 0.60  # fraction of walk-forward folds in the black
MAX_FOLD_CONCENTRATION = 0.80
PRICE_HAIRCUT_CENTS = 10     # must survive 10c worse prices than the backtest got


@dataclass
class Check:
    name: str
    passed: bool
    value: float
    threshold: float
    cause: str = ""
    lever: str = ""


@dataclass
class MarketVerdict:
    market: str
    label: str
    verdict: str = "PASS"
    primary_cause: str = ""
    lever: str = ""
    checks: list[Check] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    def failed(self) -> list[Check]:
        return [c for c in self.checks if not c.passed]


def _ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error: mean |confidence - accuracy| over bins."""
    if len(p) == 0:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        total += (m.sum() / len(p)) * abs(p[m].mean() - y[m].mean())
    return float(total)


def _apply_haircut(price: float, cents: int) -> float:
    """Move an American price `cents` against us (worse for the bettor).

    There is no price between -100 and +100, so a haircut that would land in
    that gap has to step across it: +105 worsened by 10 cents is -105, not the
    nonexistent +95. Subtracting blindly is the same class of error as averaging
    American odds, and it silently flatters any haircut test run near pick'em.
    """
    if price > 0:
        moved = price - cents
        return moved if moved >= 100 else moved - 200
    return price - cents


def evaluate_market(spec, picks: pd.DataFrame, oos: pd.DataFrame,
                    manifest: pd.DataFrame, shuffled_auc: float) -> MarketVerdict:
    from .ev import settle

    v = MarketVerdict(market=spec.key, label=spec.label)

    # ---- 1. Leakage audit (hard veto) --------------------------------------
    illegal = manifest[(manifest["market"] == spec.key)
                       & (manifest["as_of_offset_min"] > 0)]
    n_illegal = len(illegal)
    names = ", ".join(illegal["feature"]) if n_illegal else ""
    v.checks.append(Check(
        "leakage_feature_timestamps", n_illegal == 0, float(n_illegal), 0.0,
        cause=(f"{n_illegal} feature(s) carry an as-of time after the decision "
               f"timestamp: {names}. The model is reading information that did "
               f"not exist when the pick fires.") if n_illegal else "",
        lever=("Rebuild these as point-in-time aggregates closed at the decision "
               "timestamp (trailing windows ending the prior day), then re-run. "
               "Add the as-of assertion to CI so the feature store rejects them "
               "at write time rather than the gate catching them at read time.")
        if n_illegal else "",
    ))
    v.checks.append(Check(
        "leakage_target_shuffle", shuffled_auc <= MAX_SHUFFLED_AUC,
        float(shuffled_auc), MAX_SHUFFLED_AUC,
        cause=(f"Refitting on randomised labels still yields AUC {shuffled_auc:.3f} "
               f"out of sample. The leak is structural, not in a single feature.")
        if shuffled_auc > MAX_SHUFFLED_AUC else "",
        lever=("Audit the split boundary: scaler or imputer fitted before splitting, "
               "duplicated game rows, or an identifier that encodes the outcome.")
        if shuffled_auc > MAX_SHUFFLED_AUC else "",
    ))

    # ---- 2. Sample sufficiency ---------------------------------------------
    n_picks = len(picks)
    n_seasons = int(picks["test_season"].nunique()) if n_picks else 0
    graded = float(picks["result"].isin(["win", "loss", "push"]).mean()) if n_picks else 0.0
    need = required_sample_size(0.05)
    fire_rate = n_picks / max(len(oos), 1)
    v.checks.append(Check(
        "sample_size", n_picks >= MIN_PICKS, float(n_picks), float(MIN_PICKS),
        cause=(f"Only {n_picks} graded out-of-sample picks. At this price a real "
               f"+5% edge needs about {need:,} bets to clear two standard errors, so "
               f"any ROI measured here is indistinguishable from noise.")
        if n_picks < MIN_PICKS else "",
        lever=(f"Backfill more seasons, or widen selection: this market fires on only "
               f"{fire_rate:.1%} of scored games. Loosen the EV threshold and re-test, "
               f"or pool it with a structurally correlated market to borrow sample.")
        if n_picks < MIN_PICKS else "",
    ))
    v.checks.append(Check(
        "season_coverage", n_seasons >= MIN_SEASONS, float(n_seasons), float(MIN_SEASONS),
        cause=(f"Picks span only {n_seasons} test season(s); one season is one regime, "
               f"not a sample.") if n_seasons < MIN_SEASONS else "",
        lever="Backfill additional seasons before re-gating." if n_seasons < MIN_SEASONS else "",
    ))
    v.checks.append(Check(
        "grading_completeness", graded >= MIN_GRADED_FRAC, graded, MIN_GRADED_FRAC,
        cause=f"{(1 - graded):.1%} of picks are ungraded or unresolved."
        if graded < MIN_GRADED_FRAC else "",
        lever=("Fix the settlement join: pushes, suspended games and rain-shortened "
               "finals are the usual culprits.") if graded < MIN_GRADED_FRAC else "",
    ))

    # ---- 3. Calibration and 4. beating the vig-free market -----------------
    ece = brier_model = brier_mkt = bss = auc = float("nan")
    if len(oos):
        y = oos["y"].to_numpy()
        pm = oos["p_model"].to_numpy()
        pk = oos["p_fair_open"].to_numpy()
        ece = _ece(pm, y)
        brier_model = float(brier_score_loss(y, pm))
        brier_mkt = float(brier_score_loss(y, pk))
        bss = 1.0 - brier_model / brier_mkt if brier_mkt > 0 else float("nan")
        auc = float(roc_auc_score(y, pm)) if len(np.unique(y)) > 1 else float("nan")

    v.checks.append(Check(
        "calibration_ece", (not np.isnan(ece)) and ece <= MAX_ECE, ece, MAX_ECE,
        cause=(f"Expected calibration error {ece:.3f} against a {MAX_ECE:.3f} bar. "
               f"Ranking is fine (AUC {auc:.3f}) but the probabilities themselves are "
               f"off, so every EV number and Kelly stake downstream is wrong by the "
               f"same margin.") if not ((not np.isnan(ece)) and ece <= MAX_ECE) else "",
        lever=("Refit the isotonic calibration on a rolling recent window instead of "
               "the full expanding history and carry the season base rate as an "
               "offset. The drift here is in the intercept, not the slope: "
               "discrimination is intact, only the level is stale.")
        if not ((not np.isnan(ece)) and ece <= MAX_ECE) else "",
    ))
    beat_ok = (not np.isnan(bss)) and bss > MIN_BRIER_SKILL
    v.checks.append(Check(
        "beats_vig_free_market", beat_ok, bss, MIN_BRIER_SKILL,
        cause=(f"Brier skill against the vig-free price is {bss:+.4f}. The model "
               f"carries no information the market has not already priced; it beats a "
               f"coin flip, which is not the benchmark.") if not beat_ok else "",
        lever=("This needs a genuinely new input, not a better fit on the same ones. "
               "Highest-value candidates: posted-lineup and bullpen-availability "
               "deltas inside the final hour, park-adjusted Statcast quality of "
               "contact, and umpire assignment. If none of those move Brier skill "
               "positive, kill the market rather than tuning it.") if not beat_ok else "",
    ))

    # ---- 5. Closing line value ---------------------------------------------
    clv_t = float("nan")
    if n_picks > 1:
        clv = picks["clv"].to_numpy()
        se = clv.std(ddof=1) / np.sqrt(len(clv))
        clv_t = float(clv.mean() / se) if se > 0 else float("nan")
    clv_ok = (not np.isnan(clv_t)) and clv_t >= MIN_CLV_T
    v.checks.append(Check(
        "closing_line_value", clv_ok, clv_t, MIN_CLV_T,
        cause=(f"Mean CLV t-stat is {clv_t:.2f}. The prices these picks take are not "
               f"beaten by the close, so there is no evidence they contain information "
               f"the market later agrees with.") if not clv_ok else "",
        lever=("Move pick generation later, after lineups post, so the model prices the "
               "same information the closing market does; and confirm the backtest is "
               "quoting prices actually available at decision time rather than a stale "
               "opener.") if not clv_ok else "",
    ))

    # ---- 6. ROI significance (BH correction applied in finalise) -----------
    roi = p_val = lo = hi = float("nan")
    if n_picks:
        ret = (picks["pnl"] / picks["stake"]).to_numpy()
        roi = float(ret.mean())
        p_val = one_sided_p_value(ret)
        lo, hi = block_bootstrap_ci(ret, picks["game_date"].to_numpy(), n_boot=4000)
    roi_ok = (not np.isnan(lo)) and lo > 0
    v.checks.append(Check(
        "roi_ci_excludes_zero", roi_ok, lo, 0.0,
        cause=(f"ROI is {roi:+.2%} but the day-blocked bootstrap 95% CI is "
               f"[{lo:+.2%}, {hi:+.2%}], which contains zero.") if not roi_ok else "",
        lever=("Either the edge is not there or the sample is too small to show it. "
               "Read CLV first: positive CLV with an insignificant CI means keep "
               "collecting, negative CLV with the same CI means kill.")
        if not roi_ok else "",
    ))

    # ---- 7. Robustness ------------------------------------------------------
    frac_profitable = concentration = haircut_roi = float("nan")
    fold_detail = ""
    if n_picks:
        by_fold = picks.groupby("test_season")["pnl"].sum()
        frac_profitable = float((by_fold > 0).mean())
        total = by_fold.sum()
        concentration = float(by_fold.max() / total) if total > 0 else float("inf")
        fold_detail = ", ".join(f"{s}: {p:+.1f}u" for s, p in by_fold.items())
        hair = [settle(r, _apply_haircut(px, PRICE_HAIRCUT_CENTS), s)
                for r, px, s in zip(picks["result"], picks["price_side"], picks["stake"])]
        haircut_roi = float(np.sum(hair) / picks["stake"].sum())

    fold_ok = ((not np.isnan(frac_profitable))
               and frac_profitable >= MIN_PROFITABLE_FOLDS
               and concentration <= MAX_FOLD_CONCENTRATION)
    v.checks.append(Check(
        "fold_stability", fold_ok, frac_profitable, MIN_PROFITABLE_FOLDS,
        cause=(f"Profit is not stable across walk-forward folds ({fold_detail}); the "
               f"best fold carries {concentration:.0%} of total P&L. That is the "
               f"signature of an edge that existed in one regime and has since been "
               f"priced in.") if not fold_ok else "",
        lever=("Refit on a trailing window rather than an expanding one and re-gate on "
               "the most recent season alone. If the edge is gone it is gone: "
               "veto-filter the market instead of averaging a dead recent season "
               "against a live old one to manufacture a pass.") if not fold_ok else "",
    ))
    hair_ok = (not np.isnan(haircut_roi)) and haircut_roi > 0
    v.checks.append(Check(
        "price_haircut_survival", hair_ok, haircut_roi, 0.0,
        cause=(f"ROI falls to {haircut_roi:+.2%} at {PRICE_HAIRCUT_CENTS} cents worse "
               f"prices. The edge is thinner than the gap between the backtest's price "
               f"and what is actually reachable at real limits.") if not hair_ok else "",
        lever=("Price the strategy against the worst of the book set rather than the "
               "best available, and set a minimum-EV floor above the haircut before "
               "shipping.") if not hair_ok else "",
    ))

    v.metrics = {
        "n_picks": n_picks,
        "n_oos_scored": len(oos),
        "n_seasons": n_seasons,
        "bet_rate": fire_rate if len(oos) else float("nan"),
        "roi": roi,
        "roi_ci_low": lo,
        "roi_ci_high": hi,
        "p_value": p_val,
        "roi_se": float(roi_standard_error((picks["pnl"] / picks["stake"]).to_numpy()))
        if n_picks > 1 else float("nan"),
        "auc": auc,
        "ece": ece,
        "brier_model": brier_model,
        "brier_market": brier_mkt,
        "brier_skill": bss,
        "clv_mean": float(picks["clv"].mean()) if n_picks else float("nan"),
        "clv_t": clv_t,
        "shuffled_auc": shuffled_auc,
        "frac_folds_profitable": frac_profitable,
        "fold_concentration": concentration,
        "roi_after_haircut": haircut_roi,
    }
    return v


# Ordered by priority: the first failing check becomes the primary cause.
CHECK_PRIORITY = [
    "leakage_feature_timestamps",
    "leakage_target_shuffle",
    "grading_completeness",
    "season_coverage",
    "sample_size",
    "beats_vig_free_market",
    "calibration_ece",
    "closing_line_value",
    "fold_stability",
    "roi_ci_excludes_zero",
    "roi_bh_corrected",
    "price_haircut_survival",
]


def finalise(verdicts: list[MarketVerdict], fdr: float = FDR) -> list[MarketVerdict]:
    """Apply family-wise correction across markets, then assign final verdicts."""
    p_values = {v.market: v.metrics.get("p_value", float("nan")) for v in verdicts}
    survives = benjamini_hochberg(p_values, fdr=fdr)
    n_tested = sum(1 for p in p_values.values() if not np.isnan(p))

    for v in verdicts:
        ok = bool(survives.get(v.market, False))
        p = p_values.get(v.market, float("nan"))
        v.checks.append(Check(
            "roi_bh_corrected", ok, p, fdr,
            cause=(f"Raw one-sided p = {p:.3f} does not survive Benjamini-Hochberg at "
                   f"FDR {fdr:.0%} across the {n_tested} markets tested together. "
                   f"Testing this many markets and shipping the best one produces a "
                   f"false pass roughly every other slate.") if not ok else "",
            lever=("Collect more picks so the raw p-value falls, or test fewer markets "
                   "simultaneously. Do not re-test the same market with a tweaked "
                   "threshold and report the better run.") if not ok else "",
        ))
        order = {n: i for i, n in enumerate(CHECK_PRIORITY)}
        failed = sorted(v.failed(), key=lambda c: order.get(c.name, 99))
        if failed:
            top = failed[0]
            v.verdict = "VETO" if top.name.startswith("leakage") else "FAIL"
            v.primary_cause = top.cause
            v.lever = top.lever
        else:
            v.verdict = "PASS"
            v.primary_cause = ""
            v.lever = ""
    return verdicts

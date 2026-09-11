"""Render gate results to a reviewable markdown report plus a machine CSV."""

from __future__ import annotations

import pathlib

import pandas as pd

from .gate import (
    FDR,
    MAX_ECE,
    MAX_SHUFFLED_AUC,
    MIN_CLV_T,
    MIN_PICKS,
    PRICE_HAIRCUT_CENTS,
)

_BADGE = {"PASS": "**PASS**", "FAIL": "FAIL", "VETO": "**VETO**"}


def _fmt(x, pct=False, dp=3):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "n/a"
    return f"{x:+.2%}" if pct else f"{x:.{dp}f}"


def _bare(x, dp=3):
    """Percentage without a forced sign, for rates rather than returns."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "n/a"
    return f"{x:.1%}"


def results_frame(verdicts) -> pd.DataFrame:
    return pd.DataFrame([
        {"market": v.market, "label": v.label, "verdict": v.verdict,
         **v.metrics,
         "failed_checks": "|".join(c.name for c in v.failed()),
         "primary_cause": v.primary_cause, "lever": v.lever}
        for v in verdicts
    ])


def write_reports(verdicts, picks: pd.DataFrame, out_dir: pathlib.Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = results_frame(verdicts)
    df.to_csv(out_dir / "gate_results.csv", index=False)
    if len(picks):
        picks.to_parquet(out_dir / "graded_picks.parquet", index=False)

    n_pass = int((df["verdict"] == "PASS").sum())
    lines = [
        "# MLB market gate - results",
        "",
        f"`{n_pass}` of `{len(df)}` markets cleared the gate. Every number below is "
        "out-of-sample, from expanding-window walk-forward folds with a 3-day "
        "embargo. No market is judged on data that existed when its model was fit.",
        "",
        "## Summary",
        "",
        "| Market | Verdict | Picks | ROI | 95% CI | CLV t | Brier skill | ECE |",
        "|---|---|--:|--:|:--:|--:|--:|--:|",
    ]
    for v in verdicts:
        m = v.metrics
        ci = f"[{_fmt(m['roi_ci_low'], pct=True)}, {_fmt(m['roi_ci_high'], pct=True)}]"
        lines.append(
            f"| {v.label} | {_BADGE[v.verdict]} | {m['n_picks']} | "
            f"{_fmt(m['roi'], pct=True)} | {ci} | {_fmt(m['clv_t'], dp=2)} | "
            f"{_fmt(m['brier_skill'], dp=4)} | {_fmt(m['ece'])} |"
        )

    lines += [
        "",
        "## Gate policy",
        "",
        "A market ships only if it clears every check. They run in priority order and "
        "the first failure is reported as the primary cause, because the causes are "
        "not independent: a leaking market also looks beautifully calibrated and "
        "highly significant.",
        "",
        "| # | Check | Bar | Why it is in the gate |",
        "|--:|---|---|---|",
        "| 1 | Feature as-of audit | no feature timestamped after decision time | "
        "A feature that did not exist at pick time cannot be used at pick time. "
        "Hard veto. |",
        f"| 2 | Target shuffle | OOS AUC on randomised labels <= {MAX_SHUFFLED_AUC} | "
        "Catches structural leaks no manifest can see. Hard veto. |",
        "| 3 | Grading completeness | >= 99% resolved, pushes graded as pushes | "
        "Pushes scored as losses quietly kill live totals and run-line markets. |",
        f"| 4 | Sample | >= {MIN_PICKS} OOS picks across >= 2 seasons | "
        "Below this, ROI cannot be separated from variance. |",
        "| 5 | Beats the vig-free market | Brier skill > 0 vs the de-vigged price | "
        "The benchmark is the market, not a coin flip. |",
        f"| 6 | Calibration | ECE <= {MAX_ECE} | "
        "Miscalibrated probabilities break every EV number and stake downstream. |",
        f"| 7 | CLV | mean CLV t >= {MIN_CLV_T} | "
        "Lower-variance evidence of edge than ROI, measured on every bet. |",
        "| 8 | ROI significance | day-blocked bootstrap 95% CI excludes zero | "
        "Bets on one slate are correlated; resampling bets rather than days gives a "
        "CI that is too tight. |",
        f"| 9 | Multiple testing | survives Benjamini-Hochberg at FDR {FDR:.0%} | "
        "Eight markets tested and the best one shipped is a false pass every other "
        "slate. |",
        "| 10 | Fold stability | >= 60% of folds profitable, no fold > 80% of P&L | "
        "Separates a live edge from one that has been arbitraged away. |",
        f"| 11 | Price haircut | ROI > 0 at {PRICE_HAIRCUT_CENTS}c worse prices | "
        "The backtest's price is not the price you get at real limits. |",
        "",
        "## Per-market findings",
        "",
    ]

    for v in verdicts:
        m = v.metrics
        lines += [
            f"### {v.label} - {_BADGE[v.verdict]}",
            "",
            f"`{v.market}` | {m['n_picks']} picks on {m['n_oos_scored']:,} scored "
            f"games ({_bare(m['bet_rate'])} fire rate) | "
            f"ROI {_fmt(m['roi'], pct=True)} (SE {_bare(m['roi_se'])}) | "
            f"AUC {_fmt(m['auc'])} | mean CLV {_fmt(m['clv_mean'], pct=True)}",
            "",
        ]
        if v.verdict == "PASS":
            lines += [
                "Cleared every check. Ship behind the standard live guards: "
                "quarter-Kelly staking, a minimum-EV floor above the price haircut, "
                "and a rolling CLV monitor that pulls the market if mean CLV turns "
                "negative over a 200-pick window.",
                "",
            ]
        else:
            lines += [
                f"**Cause.** {v.primary_cause}",
                "",
                f"**Lever.** {v.lever}",
                "",
            ]
            others = [c for c in v.failed() if c.cause != v.primary_cause]
            if others:
                lines += ["Also failing: "
                          + ", ".join(f"`{c.name}`" for c in others) + ".", ""]

    (out_dir / "market_report.md").write_text("\n".join(lines), encoding="utf-8")

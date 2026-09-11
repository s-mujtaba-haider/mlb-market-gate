"""Market registry and ground-truth spec for the synthetic MLB dataset.

`truth` on each MarketSpec is what the generator was told to build. The gate
never sees it; it exists only so `tests/test_gate.py` can assert that the gate
reached the right verdict for the right reason. That is the point of the whole
repo: a validator you cannot check is just a second model you have to trust.
"""

from __future__ import annotations

from dataclasses import dataclass, field

SEASONS = (2021, 2022, 2023, 2024)

# Decision time: how long before first pitch the pick is generated and priced.
DECISION_LEAD_MIN = 90


@dataclass(frozen=True)
class MarketSpec:
    key: str
    label: str
    base_rate: float           # unconditional P(side wins)
    signal_strength: float     # how much latent form moves the true probability
    efficiency: float | dict   # how much of the signal the market already prices
    price_center: float        # typical American price on the side we bet
    vig: float                 # book hold on the two-way market
    bet_rate: float            # fraction of games that clear the EV threshold
    push_rate: float = 0.0     # integer-line markets can push
    base_rate_drift: float = 0.0   # per-season drift -> calibration decay
    leak_feature: bool = False     # plant a post-decision feature
    truth: str = ""            # expected verdict, asserted in tests


MARKETS: tuple[MarketSpec, ...] = (
    MarketSpec(
        key="ml_home",
        label="Moneyline - home",
        base_rate=0.540, signal_strength=0.85, efficiency=1.00,
        price_center=-125, vig=0.042, bet_rate=0.34,
        truth="FAIL:no_edge",
    ),
    MarketSpec(
        key="rl_home_-1.5",
        label="Run line - home -1.5",
        base_rate=0.360, signal_strength=1.05, efficiency=0.64,
        price_center=145, vig=0.047, bet_rate=0.26,
        truth="PASS",
    ),
    MarketSpec(
        key="total_over",
        label="Game total - over",
        base_rate=0.495, signal_strength=0.70, efficiency=0.99,
        price_center=-110, vig=0.045, bet_rate=0.30, push_rate=0.06,
        truth="FAIL:no_edge",
    ),
    MarketSpec(
        key="f5_ml_home",
        label="First 5 innings - home ML",
        base_rate=0.520, signal_strength=1.00, efficiency=0.48,
        price_center=-115, vig=0.052, bet_rate=0.24,
        truth="PASS",
    ),
    MarketSpec(
        key="nrfi",
        label="No runs first inning",
        base_rate=0.560, signal_strength=0.95, efficiency=0.60,
        price_center=-120, vig=0.055, bet_rate=0.030,
        truth="FAIL:insufficient_sample",
    ),
    MarketSpec(
        key="team_total_over_home",
        label="Home team total - over",
        base_rate=0.490, signal_strength=0.80, efficiency=0.97,
        price_center=-112, vig=0.048, bet_rate=0.28, push_rate=0.04,
        leak_feature=True,
        truth="FAIL:leakage",
    ),
    MarketSpec(
        key="pitcher_ks_over",
        label="Starting pitcher strikeouts - over",
        # Edge was real in 2021 and fully arbitraged away by 2023.
        base_rate=0.500, signal_strength=1.10,
        efficiency={2021: 0.34, 2022: 0.34, 2023: 0.97, 2024: 1.00},
        price_center=-118, vig=0.060, bet_rate=0.31, push_rate=0.05,
        truth="FAIL:regime_instability",
    ),
    MarketSpec(
        key="batter_hits_over_0.5",
        label="Batter to record a hit",
        base_rate=0.660, signal_strength=0.90, efficiency=0.40,
        price_center=-155, vig=0.058, bet_rate=0.33,
        base_rate_drift=-0.038,   # league contact rate drifts season over season
        truth="FAIL:calibration",
    ),
)

MARKETS_BY_KEY = {m.key: m for m in MARKETS}

# Feature registry. `as_of_offset_min` is measured from the decision timestamp;
# it must be <= 0 for a feature to be legal. The generator writes this manifest
# and the gate audits against it, exactly as a production feature store would.
FEATURES = [
    ("form_10g",        -240, "Rolling 10-game team form, closed as of yesterday"),
    ("pitcher_form",    -240, "Starter's trailing 5-start rate stat"),
    ("bullpen_fatigue", -180, "Relief innings thrown over trailing 3 days"),
    ("park_weather",     -120, "Park factor x forecast wind/temperature"),
    ("lineup_strength",  -95,  "Projected lineup wOBA, posted lineups only"),
]

LEAK_FEATURE = (
    "season_team_ops",
    +720,
    "Full-season team OPS -- computed at season end, includes this game's result",
)

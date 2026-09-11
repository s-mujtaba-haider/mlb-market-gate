"""Deterministic synthetic MLB dataset with known ground truth.

THIS IS NOT REAL MLB DATA. It is a generator whose parameters are known, built
so that the gate's verdicts can be checked against truth. `ingest_statsapi.py`
is the real-data path; this module exists because a validator that has only
ever seen data where nobody knows the right answer has not been validated.

What is modelled, per game and market:
  z            latent true form signal, N(0,1)
  p_true       true win probability for the side, sigmoid(logit(base) + s*z)
  p_mkt        the market's fair view, which prices only `efficiency` * s * z
  price        p_mkt inflated by vig and quoted as American odds
  p_close      the market drifting partway toward p_true before first pitch
  features     noisy observations of z, each with an as-of timestamp

When efficiency < 1 the market is systematically underweighting a signal that
IS recoverable from the features, so a model that recovers z has genuine edge
and earns positive CLV. When efficiency == 1 no model can beat the price, and
any backtest profit is noise. That is the distinction the gate must make.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .odds import decimal_to_american, devig
from .schema import (
    DECISION_LEAD_MIN,
    FEATURES,
    LEAK_FEATURE,
    MARKETS,
    SEASONS,
    MarketSpec,
)

SEED = 20260911
GAMES_PER_SEASON = 2430
CLOSE_DRIFT = 0.38      # how far the market moves toward truth by first pitch
MARKET_NOISE = 0.16     # irreducible pricing noise in logit space

TEAMS = [
    "ARI", "ATL", "BAL", "BOS", "CHC", "CIN", "CLE", "COL", "CWS", "DET",
    "HOU", "KC", "LAA", "LAD", "MIA", "MIL", "MIN", "NYM", "NYY", "OAK",
    "PHI", "PIT", "SD", "SEA", "SF", "STL", "TB", "TEX", "TOR", "WSH",
]


def _logit(p: np.ndarray | float) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def build_schedule(rng: np.random.Generator) -> pd.DataFrame:
    """A plausible MLB regular-season calendar: ~2430 games, late March to Sept."""
    rows = []
    game_id = 0
    for season in SEASONS:
        start = pd.Timestamp(f"{season}-03-28")
        n_days = 178
        per_day = GAMES_PER_SEASON / n_days
        for d in range(n_days):
            day = start + pd.Timedelta(days=d)
            n_games = rng.poisson(per_day)
            for _ in range(int(n_games)):
                home, away = rng.choice(len(TEAMS), size=2, replace=False)
                # First pitch clusters in the evening, local time folded into UTC.
                start_ts = day + pd.Timedelta(hours=int(rng.integers(17, 24)),
                                              minutes=int(rng.choice([5, 7, 10, 35, 40])))
                rows.append({
                    "game_id": f"{season}{game_id:05d}",
                    "season": season,
                    "game_date": day.normalize(),
                    "start_ts": start_ts,
                    "decision_ts": start_ts - pd.Timedelta(minutes=DECISION_LEAD_MIN),
                    "home_team": TEAMS[home],
                    "away_team": TEAMS[away],
                })
                game_id += 1
    return pd.DataFrame(rows)


def _efficiency_for(spec: MarketSpec, seasons: np.ndarray) -> np.ndarray:
    if isinstance(spec.efficiency, dict):
        return np.array([spec.efficiency[s] for s in seasons], dtype=float)
    return np.full(len(seasons), float(spec.efficiency))


def _price_two_way(p_fair: np.ndarray, vig: float, rng: np.random.Generator
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Quote a two-way market at `vig` hold, split unevenly as books actually do."""
    tilt = rng.uniform(0.42, 0.58, size=len(p_fair))
    q_side = p_fair * (1 + vig * tilt)
    q_opp = (1 - p_fair) * (1 + vig * (1 - tilt))
    q_side = np.clip(q_side, 0.01, 0.985)
    q_opp = np.clip(q_opp, 0.01, 0.985)
    side = np.array([decimal_to_american(1.0 / q) for q in q_side])
    opp = np.array([decimal_to_american(1.0 / q) for q in q_opp])
    # Books quote to the nearest 5 cents.
    return np.round(side / 5) * 5, np.round(opp / 5) * 5


def simulate_market(spec: MarketSpec, sched: pd.DataFrame,
                    rng: np.random.Generator) -> pd.DataFrame:
    n = len(sched)
    seasons = sched["season"].to_numpy()

    z = rng.normal(size=n)
    season_idx = seasons - min(SEASONS)
    base = np.clip(spec.base_rate + spec.base_rate_drift * season_idx, 0.05, 0.95)

    true_logit = _logit(base) + spec.signal_strength * z
    p_true = _sigmoid(true_logit)

    eff = _efficiency_for(spec, seasons)
    # The market knows the current base rate (it sets the line); what it may
    # underweight is the game-level signal z.
    mkt_logit = (_logit(base)
                 + spec.signal_strength * eff * z
                 + rng.normal(scale=MARKET_NOISE, size=n))
    p_mkt = _sigmoid(mkt_logit)

    price_side, price_opp = _price_two_way(p_mkt, spec.vig, rng)

    # By first pitch the market has absorbed part of the gap to truth.
    close_logit = (mkt_logit + CLOSE_DRIFT * (true_logit - mkt_logit)
                   + rng.normal(scale=MARKET_NOISE * 0.5, size=n))
    close_side, close_opp = _price_two_way(_sigmoid(close_logit), spec.vig * 0.85, rng)

    # Outcome is drawn from the TRUE probability, not the market's view.
    win = rng.random(n) < p_true
    result = np.where(win, "win", "loss").astype(object)
    if spec.push_rate > 0:
        result[rng.random(n) < spec.push_rate] = "push"

    out = pd.DataFrame({
        "game_id": sched["game_id"].to_numpy(),
        "market": spec.key,
        "season": seasons,
        "game_date": sched["game_date"].to_numpy(),
        "decision_ts": sched["decision_ts"].to_numpy(),
        "start_ts": sched["start_ts"].to_numpy(),
        "home_team": sched["home_team"].to_numpy(),
        "away_team": sched["away_team"].to_numpy(),
        "price_side": price_side,
        "price_opp": price_opp,
        "close_side": close_side,
        "close_opp": close_opp,
        "result": result,
        "y": win.astype(int),
    })

    # Observable features: noisy, partially redundant views of the latent signal.
    # Loadings are high on purpose. The optimal combination recovers ~91% of
    # var(z), which is the regime where the question "can a model beat the
    # price?" is actually about market efficiency rather than about the model
    # being starved of inputs. With weak features every market fails check 5 for
    # the same uninteresting reason.
    loadings = [0.90, 0.86, 0.80, 0.72, 0.65]
    for (name, _off, _doc), load in zip(FEATURES, loadings):
        out[name] = load * z + rng.normal(scale=np.sqrt(max(1 - load**2, 0.05)), size=n)

    if spec.leak_feature:
        # Season-aggregate stat: contains this game's own result. Predictive in a
        # way no point-in-time feature could be, and worthless in production.
        name = LEAK_FEATURE[0]
        out[name] = 0.55 * z + 1.25 * (win.astype(float) - 0.5) + rng.normal(scale=0.5, size=n)

    return out


def build_dataset(seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (rows, feature_manifest)."""
    rng = np.random.default_rng(seed)
    sched = build_schedule(rng)
    frames = [simulate_market(spec, sched, rng) for spec in MARKETS]
    rows = pd.concat(frames, ignore_index=True)

    manifest = []
    for spec in MARKETS:
        for name, off, doc in FEATURES:
            manifest.append({"market": spec.key, "feature": name,
                             "as_of_offset_min": off, "description": doc})
        if spec.leak_feature:
            name, off, doc = LEAK_FEATURE
            manifest.append({"market": spec.key, "feature": name,
                             "as_of_offset_min": off, "description": doc})
    return rows, pd.DataFrame(manifest)

"""Ingest real MLB results from the public MLB Stats API.

This is the production data path. It is written against the live API shape and
caches every response to disk so a backfill is re-runnable and a re-run costs
nothing. It could not be executed while this repo was built -- the build
environment has no network egress -- which is exactly why `simulate.py` exists
and why the gate is validated against generated data with known ground truth
rather than against data where nobody knows the right answer.

What this module gives you, and what it does not:

  results          yes. Final scores, per-inning linescores, and from those the
                   settlement for moneyline, run line, first-five, NRFI and
                   team totals.
  point-in-time    yes, for the game clock. `gameDate` is scheduled first pitch
                   in UTC, so decision_ts = gameDate - DECISION_LEAD_MIN.
  odds             NO. The Stats API carries no prices. Odds come from a
                   separate vendor (see `ODDS_NOTE`) and are the part of the
                   backfill that actually costs money and needs care, because
                   you need the price that was *available at the decision
                   timestamp*, not the opener and not the close.

Rate limits: the API is public and unauthenticated but not free to hammer.
`MIN_INTERVAL_S` keeps a backfill polite; a full four-season pull is roughly
10k requests, so run it once into the cache and work from there.
"""

from __future__ import annotations

import json
import pathlib
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from datetime import date, timedelta

import pandas as pd

from .schema import DECISION_LEAD_MIN

BASE = "https://statsapi.mlb.com/api"
SPORT_ID = 1                      # MLB
MIN_INTERVAL_S = 0.12
TIMEOUT_S = 30
USER_AGENT = "mlb-market-gate/0.1 (backfill; contact: repo owner)"

ODDS_NOTE = """
Odds are not in the Stats API. Point-in-time prices have to come from a vendor
that stores snapshots rather than a single settled line. Practical options:

  the-odds-api.com      historical endpoint returns snapshots at a timestamp.
                        Costs credits per snapshot per market, so decide the
                        snapshot cadence before backfilling four seasons.
  sportsbookreview      scraped consensus, opener and close only. Usable for a
                        CLV baseline, not for "what could I actually have bet
                        90 minutes out".
  a book's own feed     best fidelity, needs an account and usually a
                        relationship.

Whatever the source, store the snapshot timestamp alongside the price and make
the join `price at the last snapshot at or before decision_ts`. A join that
picks the closing price is the single most effective way to manufacture a
backtest edge that does not exist.
"""

_last_call = 0.0


class StatsAPIError(RuntimeError):
    pass


def _get(path: str, cache_dir: pathlib.Path, **params) -> dict:
    """GET with on-disk caching and polite rate limiting."""
    global _last_call

    qs = "&".join(f"{k}={v}" for k, v in sorted(params.items()) if v is not None)
    url = f"{BASE}{path}" + (f"?{qs}" if qs else "")
    key = url.replace(BASE, "").replace("/", "_").replace("?", "__").replace("&", "_")
    cache_file = cache_dir / f"{key[:180]}.json"

    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))

    wait = MIN_INTERVAL_S - (time.monotonic() - _last_call)
    if wait > 0:
        time.sleep(wait)

    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise StatsAPIError(
            f"could not reach {url}: {exc}. This module needs network access; "
            f"use mlbgate.simulate for an offline dataset."
        ) from exc
    finally:
        _last_call = time.monotonic()

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def iter_schedule(start: date, end: date, cache_dir: pathlib.Path,
                  chunk_days: int = 30) -> Iterator[dict]:
    """Yield finalised regular-season games between two dates."""
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=chunk_days - 1), end)
        payload = _get("/v1/schedule", cache_dir,
                       sportId=SPORT_ID, gameType="R",
                       startDate=cursor.isoformat(), endDate=chunk_end.isoformat())
        for day in payload.get("dates", []):
            for game in day.get("games", []):
                # Only settled games. Postponed games reappear on their makeup
                # date with a new gamePk; taking them here would double-count.
                if game.get("status", {}).get("codedGameState") != "F":
                    continue
                yield game
        cursor = chunk_end + timedelta(days=1)


def linescore(game_pk: int, cache_dir: pathlib.Path) -> dict:
    return _get(f"/v1/game/{game_pk}/linescore", cache_dir)


def _inning_runs(ls: dict) -> tuple[list[int], list[int]]:
    home, away = [], []
    for inning in ls.get("innings", []):
        home.append(int(inning.get("home", {}).get("runs") or 0))
        away.append(int(inning.get("away", {}).get("runs") or 0))
    return home, away


def derive_outcomes(game: dict, ls: dict) -> dict:
    """Settle every market this repo covers that needs no price line.

    Totals and team totals need the posted line from the odds vendor, so they
    return the raw run counts and are graded at join time. Pushes must survive
    that join as pushes; grading them as losses is a quiet way to bury a live
    totals market.
    """
    home_runs = int(game["teams"]["home"].get("score") or 0)
    away_runs = int(game["teams"]["away"].get("score") or 0)
    h_by_inning, a_by_inning = _inning_runs(ls)

    f5_home = sum(h_by_inning[:5])
    f5_away = sum(a_by_inning[:5])
    first_inning_runs = (h_by_inning[0] if h_by_inning else 0) + \
                        (a_by_inning[0] if a_by_inning else 0)

    # A home win in a regulation game means the home half of the 9th may not
    # have been played. Innings played matters for first-five and for any
    # market settled on a partial game.
    innings_played = len(h_by_inning)

    return {
        "home_runs": home_runs,
        "away_runs": away_runs,
        "total_runs": home_runs + away_runs,
        "innings_played": innings_played,
        "ml_home": int(home_runs > away_runs),
        # Run line settles on margin, and a -1.5 line cannot push.
        "rl_home_-1.5": int(home_runs - away_runs >= 2),
        # First five is a push if tied, and void if the game did not reach 5.
        "f5_ml_home": (None if innings_played < 5
                       else "push" if f5_home == f5_away
                       else int(f5_home > f5_away)),
        "nrfi": int(first_inning_runs == 0),
        # Graded at join time against the posted line.
        "home_team_runs_for_team_total": home_runs,
    }


def backfill(start: date, end: date, cache_dir: pathlib.Path,
             with_linescore: bool = True) -> pd.DataFrame:
    """Pull a date range into a tidy frame, one row per game."""
    rows = []
    for game in iter_schedule(start, end, cache_dir):
        game_pk = game["gamePk"]
        start_ts = pd.Timestamp(game["gameDate"])       # scheduled first pitch, UTC
        row = {
            "game_id": str(game_pk),
            "season": int(game["season"]),
            "game_date": start_ts.normalize(),
            "start_ts": start_ts,
            "decision_ts": start_ts - pd.Timedelta(minutes=DECISION_LEAD_MIN),
            "home_team": game["teams"]["home"]["team"]["name"],
            "away_team": game["teams"]["away"]["team"]["name"],
            "doubleheader": game.get("doubleHeader", "N"),
            "game_number": game.get("gameNumber", 1),
        }
        if with_linescore:
            row.update(derive_outcomes(game, linescore(game_pk, cache_dir)))
        rows.append(row)

    df = pd.DataFrame(rows)
    if len(df):
        # Doubleheaders share a date and both teams; gamePk is the only safe key.
        assert df["game_id"].is_unique, "duplicate gamePk in backfill"
    return df


if __name__ == "__main__":
    import sys

    cache = pathlib.Path(__file__).resolve().parents[2] / "data" / "statsapi_cache"
    try:
        df = backfill(date(2024, 4, 1), date(2024, 4, 7), cache)
    except StatsAPIError as exc:
        print(f"offline: {exc}", file=sys.stderr)
        print(ODDS_NOTE, file=sys.stderr)
        raise SystemExit(1)
    print(df.head().to_string())
    print(f"\n{len(df)} games ingested; cache at {cache}")

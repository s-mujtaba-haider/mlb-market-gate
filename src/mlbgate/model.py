"""Point-in-time modelling: purged walk-forward, threshold fitting, grading.

Every design choice here exists to stop a specific way backtests lie:

  expanding-window folds   a model is only ever scored on seasons that did not
                           exist when it was fit
  embargo                  rolling features computed near the fold boundary see
                           a few days of the next fold; those days are dropped
  train-fold thresholds    the EV cutoff is itself a fitted parameter. Choosing
                           it to maximise test-set profit is the single most
                           common silent leak in betting backtests
  train-fold calibration   isotonic/Platt mapping fit on train only
  disjointness assertion   no game_id may appear in both sides of a split
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .ev import closing_line_value, expected_value, kelly_fraction, settle
from .odds import american_to_implied, devig

EMBARGO_DAYS = 3
KELLY_FRACTION = 0.25
DEVIG_METHOD = "shin"


class LeakageError(RuntimeError):
    """Raised when a split would put the same game on both sides."""


def legal_features(manifest: pd.DataFrame, market: str) -> list[str]:
    """Features whose as-of time is at or before the decision time."""
    m = manifest[manifest["market"] == market]
    return m.loc[m["as_of_offset_min"] <= 0, "feature"].tolist()


def registered_features(manifest: pd.DataFrame, market: str) -> list[str]:
    """Everything the feature store offers, legal or not.

    The gate runs the model on this set deliberately. A pipeline that silently
    hoovers up every column in the feature store is the realistic failure mode,
    and the timestamp audit is what is supposed to catch it.
    """
    return manifest.loc[manifest["market"] == market, "feature"].tolist()


def walk_forward_splits(df: pd.DataFrame, embargo_days: int = EMBARGO_DAYS):
    """Expanding-window splits by season, with a purge gap before each test."""
    seasons = sorted(df["season"].unique())
    for i in range(1, len(seasons)):
        test_season = seasons[i]
        train_seasons = seasons[:i]
        test = df[df["season"] == test_season]
        train = df[df["season"].isin(train_seasons)]
        if len(test) == 0 or len(train) == 0:
            continue
        cutoff = test["game_date"].min() - pd.Timedelta(days=embargo_days)
        train = train[train["game_date"] <= cutoff]
        overlap = set(train["game_id"]) & set(test["game_id"])
        if overlap:
            raise LeakageError(f"{len(overlap)} game_ids in both train and test")
        yield test_season, train, test


def _fit_fold(train: pd.DataFrame, feats: list[str]):
    scaler = StandardScaler().fit(train[feats].to_numpy())
    clf = LogisticRegression(C=1.0, max_iter=1000)
    clf.fit(scaler.transform(train[feats].to_numpy()), train["y"].to_numpy())
    raw = clf.predict_proba(scaler.transform(train[feats].to_numpy()))[:, 1]
    # Calibration map fitted on train only; deliberately NOT refit on test.
    iso = IsotonicRegression(out_of_bounds="clip").fit(raw, train["y"].to_numpy())
    return scaler, clf, iso


def _predict(scaler, clf, iso, frame: pd.DataFrame, feats: list[str]) -> np.ndarray:
    raw = clf.predict_proba(scaler.transform(frame[feats].to_numpy()))[:, 1]
    return np.clip(iso.predict(raw), 1e-4, 1 - 1e-4)


def _fair_prob(frame: pd.DataFrame) -> np.ndarray:
    """Vig-free probability of the side we are considering betting."""
    return np.array([
        devig([a, b], method=DEVIG_METHOD)[0]
        for a, b in zip(frame["price_side"], frame["price_opp"])
    ])


def run_market(df: pd.DataFrame, feats: list[str], bet_rate: float
               ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk-forward a single market.

    Returns (graded picks, every out-of-sample row with a prediction). The
    second frame matters: calibration and "did we beat the market" have to be
    judged on all scored games, not only on the ones the EV filter selected,
    or the selection itself flatters the metric.
    """
    picks = []
    oos = []
    for test_season, train, test in walk_forward_splits(df):
        scaler, clf, iso = _fit_fold(train, feats)

        # Fit the EV threshold on TRAIN so the test set never chooses its own bar.
        p_train = _predict(scaler, clf, iso, train, feats)
        ev_train = np.array([expected_value(p, px)
                             for p, px in zip(p_train, train["price_side"])])
        threshold = float(np.quantile(ev_train, 1.0 - bet_rate))

        p_test = _predict(scaler, clf, iso, test, feats)
        ev_test = np.array([expected_value(p, px)
                            for p, px in zip(p_test, test["price_side"])])

        fold = test.copy()
        fold["p_model"] = p_test
        fold["p_fair_open"] = _fair_prob(test)
        fold["p_implied_raw"] = [american_to_implied(px) for px in test["price_side"]]
        fold["ev"] = ev_test
        fold["ev_threshold"] = threshold
        fold["test_season"] = test_season
        oos.append(fold.copy())
        fold = fold[fold["ev"] > threshold].copy()
        if fold.empty:
            continue

        fold["stake"] = 1.0
        fold["kelly_stake"] = [KELLY_FRACTION * kelly_fraction(p, px)
                               for p, px in zip(fold["p_model"], fold["price_side"])]
        fold["pnl"] = [settle(r, px, s) for r, px, s
                       in zip(fold["result"], fold["price_side"], fold["stake"])]
        fold["clv"] = [closing_line_value(bp, bo, cs, co, method=DEVIG_METHOD)
                       for bp, bo, cs, co in zip(fold["price_side"], fold["price_opp"],
                                                 fold["close_side"], fold["close_opp"])]
        picks.append(fold)

    oos_all = pd.concat(oos, ignore_index=True) if oos else pd.DataFrame()
    picks_all = pd.concat(picks, ignore_index=True) if picks else pd.DataFrame()
    return picks_all, oos_all


def shuffled_label_auc(df: pd.DataFrame, feats: list[str], seed: int = 7,
                       rebuild=None) -> float:
    """Permute the target ONCE across the whole dataset, then walk forward.

    Out-of-sample AUC must collapse to ~0.5. If it does not, information is
    flowing from the evaluation labels back into training, which is what this
    test exists to find: target encoding or an imputer fitted before the split,
    a calibrator fitted on test labels, the same game present in both folds
    under two different ids.

    The permutation has to be global. Shuffling train and test labels
    independently returns ~0.5 on *every* dataset, leaking or not, which makes
    the check look reassuring while testing nothing.

    `rebuild` is the feature-construction step, and passing it matters more
    than it looks. Any feature derived from the label column -- target
    encoding, a "team form" column built from past results, a rolling win rate
    -- has to be recomputed from the permuted labels, or the permutation washes
    out of the features and the test reports 0.5 no matter how badly the
    pipeline leaks. A permutation test that starts downstream of feature
    construction cannot see leakage introduced during feature construction.

    Two further limits, stated plainly because a check whose scope nobody
    understands is worse than no check:
      - it cannot see a leaking *raw* feature. Permuting the target destroys the
        relationship a post-decision input has with the outcome, so the planted
        market comes back clean here. The as-of manifest audit is what catches
        that, which is why the gate runs both.
      - its power scales with model capacity. A linear model cannot memorise
        individual duplicated rows; a gradient-boosted tree can. Read a clean
        result as "no leak this model could exploit", not "no leak".
    """
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    shuffled = df.copy()
    shuffled["y"] = rng.permutation(shuffled["y"].to_numpy())
    if rebuild is not None:
        shuffled = rebuild(shuffled)

    aucs = []
    for _season, train, test in walk_forward_splits(shuffled):
        scaler, clf, iso = _fit_fold(train, feats)
        p = _predict(scaler, clf, iso, test, feats)
        if test["y"].nunique() > 1:
            aucs.append(roc_auc_score(test["y"], p))
    return float(np.mean(aucs)) if aucs else float("nan")

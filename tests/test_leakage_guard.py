import numpy as np
import pandas as pd
import pytest

from mlbgate.model import (
    EMBARGO_DAYS,
    LeakageError,
    legal_features,
    registered_features,
    shuffled_label_auc,
    walk_forward_splits,
)
from mlbgate.schema import MARKETS_BY_KEY


class TestWalkForwardSplits:
    def test_no_game_appears_on_both_sides(self, dataset):
        rows, _ = dataset
        df = rows[rows["market"] == "ml_home"].reset_index(drop=True)
        for _season, train, test in walk_forward_splits(df):
            assert not (set(train["game_id"]) & set(test["game_id"]))

    def test_train_is_strictly_in_the_past(self, dataset):
        rows, _ = dataset
        df = rows[rows["market"] == "ml_home"].reset_index(drop=True)
        for _season, train, test in walk_forward_splits(df):
            assert train["game_date"].max() < test["game_date"].min()

    def test_embargo_gap_is_respected(self, dataset):
        """Rolling features near the boundary see into the next fold."""
        rows, _ = dataset
        df = rows[rows["market"] == "ml_home"].reset_index(drop=True)
        for _season, train, test in walk_forward_splits(df):
            gap = (test["game_date"].min() - train["game_date"].max()).days
            assert gap >= EMBARGO_DAYS

    def test_every_fold_tests_exactly_one_unseen_season(self, dataset):
        rows, _ = dataset
        df = rows[rows["market"] == "ml_home"].reset_index(drop=True)
        folds = list(walk_forward_splits(df))
        assert len(folds) == 3            # 4 seasons -> 3 out-of-sample folds
        for season, train, test in folds:
            assert test["season"].unique().tolist() == [season]
            assert season not in set(train["season"])

    def test_duplicated_game_is_caught(self, dataset):
        """The realistic version of this bug is a bad join fanning out rows."""
        rows, _ = dataset
        df = rows[rows["market"] == "ml_home"].reset_index(drop=True)
        seasons = sorted(df["season"].unique())
        leaked = df[df["season"] == seasons[1]].head(5).copy()
        leaked["season"] = seasons[0]
        leaked["game_date"] = df[df["season"] == seasons[0]]["game_date"].min()
        poisoned = pd.concat([df, leaked], ignore_index=True)
        with pytest.raises(LeakageError):
            list(walk_forward_splits(poisoned))


class TestFeatureManifest:
    def test_clean_markets_expose_only_legal_features(self, dataset):
        _rows, manifest = dataset
        for key, spec in MARKETS_BY_KEY.items():
            if spec.leak_feature:
                continue
            assert legal_features(manifest, key) == registered_features(manifest, key)

    def test_the_planted_market_has_an_illegal_feature(self, dataset):
        _rows, manifest = dataset
        key = "team_total_over_home"
        assert set(registered_features(manifest, key)) - set(legal_features(manifest, key)) \
            == {"season_team_ops"}

    def test_every_legal_feature_closes_before_the_decision(self, dataset):
        _rows, manifest = dataset
        legal = manifest[manifest["as_of_offset_min"] <= 0]
        assert len(legal) > 0
        assert (legal["as_of_offset_min"] <= 0).all()


class TestTargetShuffle:
    def test_shuffled_labels_are_unpredictable_on_a_clean_market(self, dataset):
        """If a randomised target is still predictable, the pipeline leaks."""
        rows, manifest = dataset
        df = rows[rows["market"] == "ml_home"].reset_index(drop=True)
        auc = shuffled_label_auc(df, registered_features(manifest, "ml_home"))
        assert 0.45 < auc < 0.55

    def test_shuffle_test_is_blind_to_feature_leakage(self, dataset):
        """Deliberate limitation, documented so nobody relies on the wrong check.

        Shuffling the target destroys the relationship a leaking *feature* has
        with the outcome, so this test comes back clean on the planted market.
        It catches structural leaks (a scaler fitted across the split, duplicated
        rows) -- the as-of manifest audit is what catches leaking features. Both
        are in the gate because neither subsumes the other.
        """
        rows, manifest = dataset
        key = "team_total_over_home"
        df = rows[rows["market"] == key].reset_index(drop=True)
        auc = shuffled_label_auc(df, registered_features(manifest, key))
        assert 0.45 < auc < 0.55

    def test_pre_split_target_encoding_is_caught(self, dataset):
        """Positive control: the leak this check is actually for.

        Target encoding computed over the whole dataset before splitting is one
        of the most common real leaks in production feature stores, and it is
        invisible to a timestamp audit -- the feature legitimately closes before
        the decision, it was just *fitted* on the future. Because the encoding
        is derived from the label column, a global permutation carries straight
        through it and out-of-sample AUC stays far above chance.

        A check with no demonstrated true positive is decoration. This is the
        test that stops `leakage_target_shuffle` from being decoration.
        """
        rows, manifest = dataset
        df = rows[rows["market"] == "ml_home"].reset_index(drop=True).copy()

        def poison(frame):
            # Group-mean of y over small groups spanning the whole dataset,
            # each row's own outcome included. Exactly what `groupby().mean()`
            # before a train/test split produces.
            grp = np.arange(len(frame)) // 4
            frame = frame.copy()
            frame["team_form_target_enc"] = (
                pd.Series(frame["y"].to_numpy()).groupby(grp).transform("mean").to_numpy()
            )
            return frame

        from mlbgate.gate import MAX_SHUFFLED_AUC

        feats = registered_features(manifest, "ml_home") + ["team_form_target_enc"]
        auc = shuffled_label_auc(poison(df), feats, seed=0, rebuild=poison)
        assert auc > MAX_SHUFFLED_AUC, f"shuffle test failed to fire, AUC={auc:.3f}"

    def test_the_check_is_blind_without_the_rebuild_hook(self, dataset):
        """Why `rebuild` is not optional in practice.

        Run the identical leaking pipeline but permute only the labels, leaving
        the already-built features alone, and the test comes back a comfortable
        0.5. Same leak, same data, clean bill of health. This is the failure
        mode that makes teams trust a permutation test that is testing nothing.
        """
        rows, manifest = dataset
        df = rows[rows["market"] == "ml_home"].reset_index(drop=True).copy()
        grp = np.arange(len(df)) // 4
        df["team_form_target_enc"] = (
            pd.Series(df["y"].to_numpy()).groupby(grp).transform("mean").to_numpy()
        )
        feats = registered_features(manifest, "ml_home") + ["team_form_target_enc"]
        auc = shuffled_label_auc(df, feats, seed=0)     # no rebuild hook
        assert 0.45 < auc < 0.53

    def test_the_clean_market_passes_the_same_construction(self, dataset):
        """Control for the control: with no encoded feature at all, AUC is ~0.5."""
        rows, manifest = dataset
        df = rows[rows["market"] == "ml_home"].reset_index(drop=True)
        auc = shuffled_label_auc(df, registered_features(manifest, "ml_home"), seed=0)
        assert 0.45 < auc < 0.53

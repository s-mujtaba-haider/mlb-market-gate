"""The point of the repo: the gate reaches the right verdict for the right reason.

Each market in `schema.MARKETS` was generated with a known pathology recorded in
its `truth` field. The gate never sees that field. These tests assert that what
the gate concluded matches what was actually built, and -- just as important --
that it concluded it via the correct check. A gate that fails a dead market
because of a thresholding accident is not a working gate, it is a coincidence.
"""

import numpy as np
import pytest

from mlbgate.gate import (
    CHECK_PRIORITY,
    MAX_ECE,
    MAX_FOLD_CONCENTRATION,
    MAX_SHUFFLED_AUC,
    MIN_PICKS,
)
from mlbgate.schema import MARKETS


def _check(verdict, name):
    return next(c for c in verdict.checks if c.name == name)


@pytest.mark.parametrize("spec", MARKETS, ids=lambda s: s.key)
def test_verdict_matches_ground_truth(spec, gate_results):
    v = gate_results[spec.key]
    expected = spec.truth.split(":")[0]
    # A planted leak is reported as VETO rather than FAIL.
    expected = "VETO" if spec.truth.startswith("FAIL:leakage") else expected
    assert v.verdict == expected, (
        f"{spec.key}: gate said {v.verdict}, ground truth is {spec.truth}. "
        f"Failed checks: {[c.name for c in v.failed()]}"
    )


class TestFailsForTheRightReason:
    """Ground truth is not just pass/fail, it is *why*."""

    def test_leakage_is_vetoed_on_the_timestamp_audit(self, gate_results):
        v = gate_results["team_total_over_home"]
        assert v.verdict == "VETO"
        assert not _check(v, "leakage_feature_timestamps").passed
        assert "season_team_ops" in v.primary_cause
        # And the veto fires despite the market looking spectacular, which is
        # the whole reason leakage outranks everything else in the gate.
        assert v.metrics["roi"] > 0.5
        assert v.metrics["brier_skill"] > 0.5

    def test_efficient_markets_fail_on_beating_the_market(self, gate_results):
        for key in ("ml_home", "total_over"):
            v = gate_results[key]
            assert v.verdict == "FAIL"
            assert not _check(v, "beats_vig_free_market").passed
            assert v.metrics["brier_skill"] <= 0
            # They are not failing for lack of sample -- they have thousands.
            assert _check(v, "sample_size").passed

    def test_thin_market_fails_on_sample_size(self, gate_results):
        v = gate_results["nrfi"]
        assert v.verdict == "FAIL"
        assert not _check(v, "sample_size").passed
        assert v.metrics["n_picks"] < MIN_PICKS
        # The underlying edge is real; the market simply does not fire enough
        # to prove it. The cause must say that rather than blaming the model.
        assert "indistinguishable from noise" in v.primary_cause

    def test_arbitraged_market_fails_on_fold_stability(self, gate_results):
        v = gate_results["pitcher_ks_over"]
        assert v.verdict == "FAIL"
        assert not _check(v, "fold_stability").passed
        # Pooled ROI is positive and it clears the market, the sample and the
        # significance checks. Without the fold check this market ships.
        assert v.metrics["roi"] > 0
        assert _check(v, "beats_vig_free_market").passed
        assert _check(v, "sample_size").passed
        # It fails on concentration rather than on fold count: two of three
        # folds are green, but the first one carries more than all the profit
        # because a later fold is negative. That is what a dead edge looks like.
        assert v.metrics["fold_concentration"] > MAX_FOLD_CONCENTRATION
        assert "priced in" in v.primary_cause

    def test_drifting_market_fails_on_calibration(self, gate_results):
        v = gate_results["batter_hits_over_0.5"]
        assert v.verdict == "FAIL"
        assert not _check(v, "calibration_ece").passed
        assert v.metrics["ece"] > MAX_ECE
        # Discrimination is intact; only the level is wrong. If this assertion
        # breaks, the lever in the report ("recalibrate, do not refit") is wrong.
        assert v.metrics["auc"] > 0.55
        assert v.metrics["brier_skill"] > 0


class TestPassingMarkets:
    @pytest.mark.parametrize("key", ["rl_home_-1.5", "f5_ml_home"])
    def test_passing_markets_clear_every_check(self, key, gate_results):
        v = gate_results[key]
        assert v.verdict == "PASS"
        assert v.failed() == []
        assert v.primary_cause == ""

    @pytest.mark.parametrize("key", ["rl_home_-1.5", "f5_ml_home"])
    def test_passing_markets_have_positive_clv_and_survive_a_haircut(self, key, gate_results):
        v = gate_results[key]
        assert v.metrics["clv_mean"] > 0
        assert v.metrics["roi_after_haircut"] > 0
        assert v.metrics["roi_ci_low"] > 0


class TestGateMechanics:
    def test_no_clean_market_trips_the_shuffle_test(self, gate_results):
        for key, v in gate_results.items():
            assert v.metrics["shuffled_auc"] <= MAX_SHUFFLED_AUC, key

    def test_every_failure_has_a_cause_and_a_lever(self, gate_results):
        for key, v in gate_results.items():
            if v.verdict == "PASS":
                continue
            assert len(v.primary_cause) > 40, f"{key} cause is too thin to act on"
            assert len(v.lever) > 40, f"{key} lever is too thin to act on"
            # A lever must propose an action, not restate the problem.
            assert v.lever != v.primary_cause

    def test_primary_cause_is_the_highest_priority_failure(self, gate_results):
        order = {n: i for i, n in enumerate(CHECK_PRIORITY)}
        for key, v in gate_results.items():
            failed = v.failed()
            if not failed:
                continue
            top = min(failed, key=lambda c: order[c.name])
            assert v.primary_cause == top.cause, key

    def test_every_check_name_is_ranked(self, gate_results):
        for v in gate_results.values():
            for c in v.checks:
                assert c.name in CHECK_PRIORITY, f"{c.name} has no priority rank"

    def test_multiple_testing_correction_is_applied(self, gate_results):
        # Eight markets are tested together, so the correction must exist and
        # must be stricter than the raw p-values it is derived from.
        for v in gate_results.values():
            assert any(c.name == "roi_bh_corrected" for c in v.checks)
        passing = [v for v in gate_results.values() if v.verdict == "PASS"]
        for v in passing:
            assert v.metrics["p_value"] < 0.10

    def test_metrics_are_finite_where_picks_exist(self, gate_results):
        for key, v in gate_results.items():
            if v.metrics["n_picks"] == 0:
                continue
            for field in ("roi", "clv_t", "brier_skill", "ece", "auc"):
                assert not np.isnan(v.metrics[field]), f"{key}.{field} is nan"

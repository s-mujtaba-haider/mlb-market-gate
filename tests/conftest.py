import pathlib
import sys

import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="session")
def dataset():
    """Build (or reuse) the ground-truth dataset once for the whole session."""
    from mlbgate.simulate import build_dataset

    rows_path = ROOT / "data" / "rows.parquet"
    manifest_path = ROOT / "data" / "feature_manifest.csv"
    if rows_path.exists() and manifest_path.exists():
        return pd.read_parquet(rows_path), pd.read_csv(manifest_path)
    return build_dataset()


@pytest.fixture(scope="session")
def gate_results(dataset):
    """Run every market through the gate once; tests assert against the verdicts."""
    from mlbgate.gate import evaluate_market, finalise
    from mlbgate.model import registered_features, run_market, shuffled_label_auc
    from mlbgate.schema import MARKETS

    rows, manifest = dataset
    verdicts = {}
    for spec in MARKETS:
        df = rows[rows["market"] == spec.key].reset_index(drop=True)
        feats = registered_features(manifest, spec.key)
        picks, oos = run_market(df, feats, spec.bet_rate)
        auc = shuffled_label_auc(df, feats)
        verdicts[spec.key] = evaluate_market(spec, picks, oos, manifest, auc)
    finalise(list(verdicts.values()))
    return verdicts

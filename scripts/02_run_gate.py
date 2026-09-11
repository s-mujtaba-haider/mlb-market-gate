"""Run every MLB market through the gate. Writes reports/."""
import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlbgate.gate import evaluate_market, finalise  # noqa: E402
from mlbgate.model import registered_features, run_market, shuffled_label_auc  # noqa: E402
from mlbgate.report import write_reports  # noqa: E402
from mlbgate.schema import MARKETS  # noqa: E402


def main() -> None:
    rows = pd.read_parquet(ROOT / "data" / "rows.parquet")
    manifest = pd.read_csv(ROOT / "data" / "feature_manifest.csv")

    verdicts, all_picks = [], []
    for spec in MARKETS:
        df = rows[rows["market"] == spec.key].reset_index(drop=True)
        # Deliberately hand the model everything the feature store registered,
        # legal or not. Catching that is the gate's job, not the caller's.
        feats = registered_features(manifest, spec.key)
        picks, oos = run_market(df, feats, spec.bet_rate)
        auc_shuffled = shuffled_label_auc(df, feats)
        v = evaluate_market(spec, picks, oos, manifest, auc_shuffled)
        verdicts.append(v)
        if len(picks):
            all_picks.append(picks)
        print(f"  scored {spec.key:<24} picks={len(picks):>5}  "
              f"roi={v.metrics['roi']:+.2%}  clv_t={v.metrics['clv_t']:5.2f}  "
              f"bss={v.metrics['brier_skill']:+.4f}")

    verdicts = finalise(verdicts)
    picks_df = pd.concat(all_picks, ignore_index=True) if all_picks else pd.DataFrame()
    write_reports(verdicts, picks_df, ROOT / "reports")

    print()
    for v in verdicts:
        print(f"{v.verdict:<5} {v.market:<24} n={v.metrics['n_picks']:>5} "
              f"roi={v.metrics['roi']:+.2%}  "
              f"failed={'|'.join(c.name for c in v.failed()) or '-'}")


if __name__ == "__main__":
    main()

"""Build the dataset. Writes data/rows.parquet and data/feature_manifest.csv."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from mlbgate.simulate import build_dataset  # noqa: E402

DATA = pathlib.Path(__file__).resolve().parents[1] / "data"


def main() -> None:
    DATA.mkdir(exist_ok=True)
    rows, manifest = build_dataset()
    rows.to_parquet(DATA / "rows.parquet", index=False)
    manifest.to_csv(DATA / "feature_manifest.csv", index=False)
    print(f"rows            : {len(rows):,}")
    print(f"markets         : {rows['market'].nunique()}")
    print(f"seasons         : {sorted(rows['season'].unique())}")
    print(f"games/season    : {rows.groupby('season')['game_id'].nunique().to_dict()}")
    print(f"feature manifest: {len(manifest)} entries, "
          f"{(manifest['as_of_offset_min'] > 0).sum()} illegal")
    print(f"written to      : {DATA}")


if __name__ == "__main__":
    main()

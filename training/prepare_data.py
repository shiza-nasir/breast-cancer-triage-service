
import argparse
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "raw" / "breast_cancer.csv"


def prepare_data(output_path, drop_rows=0):
    dataset = load_breast_cancer(as_frame=True)
    dataframe = dataset.frame.copy()

    dataframe.columns = [
        column.replace(" ", "_") for column in dataframe.columns
    ]

    if drop_rows < 0 or drop_rows >= len(dataframe):
        raise ValueError("drop_rows must be between 0 and 568")

    if drop_rows:
        dataframe = dataframe.iloc[:-drop_rows].copy()

    feature_columns = [
        column for column in dataframe.columns if column != "target"
    ]

    if len(feature_columns) != 30:
        raise ValueError(
            f"Expected 30 features, found {len(feature_columns)}"
        )

    if dataframe.isnull().values.any():
        raise ValueError("Dataset contains null values")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)

    saved = pd.read_csv(output_path)

    if saved.isnull().values.any():
        raise ValueError("Saved CSV contains null values")

    saved_features = [
        column for column in saved.columns if column != "target"
    ]

    if len(saved_features) != 30:
        raise ValueError(
            f"Expected 30 saved features, found {len(saved_features)}"
        )

    print(f"Output file: {output_path}")
    print(f"Rows: {len(saved)}")
    print(f"Feature count: {len(saved_features)}")
    print(f"Target column present: {'target' in saved.columns}")
    print(f"Null values: {int(saved.isnull().sum().sum())}")
    print(f"File size: {output_path.stat().st_size} bytes")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--drop-rows", type=int, default=0)
    args = parser.parse_args()
    prepare_data(args.output, args.drop_rows)


if __name__ == "__main__":
    main()
"""Load, validate, clean, and split the Dry Bean dataset."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "dry_bean.csv"

# The source file has two typos in its column names. They are fixed on load.
COLUMN_FIXES = {"AspectRation": "AspectRatio", "roundness": "Roundness"}

FEATURES = [
    "Area", "Perimeter", "MajorAxisLength", "MinorAxisLength",
    "AspectRatio", "Eccentricity", "ConvexArea", "EquivDiameter",
    "Extent", "Solidity", "Roundness", "Compactness",
    "ShapeFactor1", "ShapeFactor2", "ShapeFactor3", "ShapeFactor4",
]
TARGET = "Class"
CLASSES = {"BARBUNYA", "BOMBAY", "CALI", "DERMASON", "HOROZ", "SEKER", "SIRA"}
MIN_ROWS = 1000


def load_data(path=DATA_PATH):
    """Read the CSV and fix the column name typos."""
    return pd.read_csv(path).rename(columns=COLUMN_FIXES)


def validate_data(df):
    """Check the data before training. Raises ValueError listing every problem found."""
    problems = []

    missing_cols = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing_cols:
        # The other checks need these columns, so stop here.
        raise ValueError(f"Missing columns: {missing_cols}")

    if len(df) < MIN_ROWS:
        problems.append(f"Only {len(df)} rows, expected at least {MIN_ROWS}")

    n_missing = int(df[FEATURES + [TARGET]].isna().sum().sum())
    if n_missing:
        problems.append(f"{n_missing} missing values")

    n_not_positive = int((df[FEATURES] <= 0).sum().sum())
    if n_not_positive:
        problems.append(f"{n_not_positive} zero or negative measurements")

    unknown = sorted(set(df[TARGET].dropna()) - CLASSES)
    if unknown:
        problems.append(f"Unknown classes: {unknown}")

    if problems:
        raise ValueError("Data validation failed: " + "; ".join(problems))

    return {
        "rows": len(df),
        "duplicate_rows": int(df.duplicated().sum()),
        "class_counts": df[TARGET].value_counts().to_dict(),
    }


def clean_data(df):
    """Remove exact duplicate rows."""
    return df.drop_duplicates().reset_index(drop=True)


def split_data(df, test_size=0.2, seed=42):
    """Stratified train/test split, so each bean class keeps its share."""
    train, test = train_test_split(
        df, test_size=test_size, stratify=df[TARGET], random_state=seed
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)
"""Tests for loading, validating, cleaning, and splitting the data."""

import pandas as pd
import pytest

from src.data import (
    DATA_PATH, FEATURES, MIN_ROWS, TARGET,
    clean_data, load_data, split_data, validate_data,
)


def test_real_file_loads_with_fixed_column_names():
    df = load_data(DATA_PATH)
    assert list(df.columns) == FEATURES + [TARGET]
    assert df.shape == (13611, 17)


def test_real_file_passes_validation():
    summary = validate_data(load_data(DATA_PATH))
    assert summary["rows"] == 13611
    assert summary["duplicate_rows"] == 68
    assert len(summary["class_counts"]) == 7


def test_load_fixes_typos_in_column_names(raw_csv):
    df = load_data(raw_csv)
    assert "AspectRatio" in df.columns and "Roundness" in df.columns
    assert "AspectRation" not in df.columns and "roundness" not in df.columns


def test_validation_fails_on_missing_column(beans):
    with pytest.raises(ValueError, match="Missing columns"):
        validate_data(beans.drop(columns=["Area"]))


def test_validation_lists_every_problem(beans):
    bad = beans.copy()
    bad.loc[0, "Area"] = -5
    bad.loc[1, "Perimeter"] = None
    bad.loc[2, TARGET] = "PINTO"
    with pytest.raises(ValueError) as error:
        validate_data(bad)
    message = str(error.value)
    assert "missing values" in message
    assert "zero or negative" in message
    assert "Unknown classes: ['PINTO']" in message


def test_validation_fails_on_too_few_rows(beans):
    assert len(beans) < MIN_ROWS
    with pytest.raises(ValueError, match="Only 400 rows"):
        validate_data(beans)


def test_clean_removes_only_duplicates(beans):
    with_duplicates = pd.concat([beans, beans.head(3)], ignore_index=True)
    cleaned = clean_data(with_duplicates)
    assert len(cleaned) == len(beans)


def test_split_is_stratified_and_reproducible(beans):
    train, test = split_data(beans, test_size=0.2, seed=1)
    assert len(test) == 80
    # each class keeps its 25% share in the test set
    assert set(test[TARGET].value_counts()) == {20}
    train_again, test_again = split_data(beans, test_size=0.2, seed=1)
    assert test.equals(test_again)
"""Shared test setup: make dags/src importable and build a small synthetic bean dataset."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

DAGS_DIR = Path(__file__).resolve().parents[1] / "dags"
sys.path.insert(0, str(DAGS_DIR))

from src.data import COLUMN_FIXES, FEATURES, TARGET  # noqa: E402

N_GROUPS = 4
CLASS_NAMES = ["BOMBAY", "CALI", "HOROZ", "SIRA"]


@pytest.fixture
def beans():
    """400 synthetic beans in 4 well-separated groups, with clean column names."""
    rng = np.random.default_rng(0)
    centers = rng.uniform(1, 10, size=(N_GROUPS, len(FEATURES)))
    rows = [c * rng.normal(1, 0.05, size=(100, len(FEATURES))) for c in centers]
    df = pd.DataFrame(np.vstack(rows), columns=FEATURES)
    df[TARGET] = np.repeat(CLASS_NAMES, 100)
    return df


@pytest.fixture
def raw_csv(tmp_path, beans):
    """The synthetic beans saved as a CSV with the source file's original column names."""
    original_names = {fixed: raw for raw, fixed in COLUMN_FIXES.items()}
    path = tmp_path / "beans.csv"
    beans.rename(columns=original_names).to_csv(path, index=False)
    return path
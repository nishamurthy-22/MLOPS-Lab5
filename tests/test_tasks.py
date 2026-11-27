"""Tests for the task functions that the DAG calls."""

import json
from pathlib import Path

import pytest

from src import tasks

EXPECTED_FILES = {
    "validation.json", "train.csv", "test.csv", "k_comparison.csv",
    "k_comparison.png", "model.joblib", "metrics.json", "cluster_profile.csv", "report.md",
}


def test_run_folder_name_is_safe(tmp_path):
    run_dir = tasks.make_run_dir("manual__2026-11-27T02:20:16+00:00", root=tmp_path)
    assert Path(run_dir).is_dir()
    assert Path(run_dir).name == "manual__2026-11-27T02_20_16_00_00"


def test_full_pipeline_writes_every_output(tmp_path, raw_csv, monkeypatch):
    monkeypatch.setattr(tasks.data, "MIN_ROWS", 100)  # the synthetic set is small
    run_dir = tasks.make_run_dir("test_run", root=tmp_path)

    validation = tasks.validate(run_dir, data_path=raw_csv)
    counts = tasks.prepare(run_dir, test_size=0.2, data_path=raw_csv)
    choice = tasks.select_k(run_dir, k_min=2, k_max=8)
    tasks.train(run_dir, choice["k"])
    metrics = tasks.evaluate(run_dir)
    clusters = tasks.profile(run_dir)
    report_path = tasks.report(run_dir, validation, counts, choice, metrics, clusters)

    assert EXPECTED_FILES <= {p.name for p in Path(run_dir).iterdir()}
    assert counts == {"clean_rows": 400, "train_rows": 320, "test_rows": 80}
    assert choice["k"] == 4
    assert json.loads((Path(run_dir) / "metrics.json").read_text()) == metrics
    assert "Chosen k: 4" in Path(report_path).read_text()


def test_validate_task_fails_on_bad_data(tmp_path, raw_csv):
    # 400 rows is below the 1000-row minimum, so the task must fail
    with pytest.raises(ValueError, match="Data validation failed"):
        tasks.validate(str(tmp_path), data_path=raw_csv)


def test_select_k_rejects_an_empty_range(tmp_path):
    with pytest.raises(ValueError, match="must be smaller"):
        tasks.select_k(str(tmp_path), k_min=8, k_max=8)
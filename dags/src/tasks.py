"""The work done by each Airflow task.

Each function reads its inputs from the run folder and writes its outputs there,
so tasks pass small file paths and summaries through XCom instead of pickled data.
None of this code imports Airflow, so it can be tested on its own.
"""

import json
import os
import re
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # no screen inside the container
import matplotlib.pyplot as plt
import pandas as pd

from src import data, model

ARTIFACTS_DIR = Path(os.environ.get("ARTIFACTS_DIR", "/opt/airflow/artifacts"))


def make_run_dir(run_id, root=None):
    """Create a separate output folder for each DAG run."""
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", run_id).strip("_")
    run_dir = Path(root or ARTIFACTS_DIR) / safe_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return str(run_dir)


def _write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2))


def validate(run_dir, data_path=data.DATA_PATH):
    """Check the raw data. Fails the task, and stops the pipeline, if anything is wrong."""
    summary = data.validate_data(data.load_data(data_path))
    _write_json(Path(run_dir) / "validation.json", summary)
    print(f"Validation passed: {summary['rows']} rows, {summary['duplicate_rows']} duplicates")
    return summary


def prepare(run_dir, test_size, data_path=data.DATA_PATH):
    """Clean the data and save the train and test sets."""
    clean = data.clean_data(data.load_data(data_path))
    train, test = data.split_data(clean, test_size=test_size)
    train.to_csv(Path(run_dir) / "train.csv", index=False)
    test.to_csv(Path(run_dir) / "test.csv", index=False)
    counts = {"clean_rows": len(clean), "train_rows": len(train), "test_rows": len(test)}
    print(f"Prepared data: {counts}")
    return counts


def _plot_k_comparison(results, chosen_k, path):
    ks = [r["k"] for r in results]
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(ks, [r["sse"] for r in results], "o-", color="tab:blue", label="SSE (elbow)")
    ax1.set_xlabel("Number of clusters (k)")
    ax1.set_ylabel("SSE", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(ks, [r["silhouette"] for r in results], "s--", color="tab:orange", label="Silhouette")
    ax2.set_ylabel("Silhouette score", color="tab:orange")
    ax1.axvline(chosen_k, color="gray", linestyle=":", label=f"Chosen k = {chosen_k}")
    ax1.set_xticks(ks)
    lines = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax1.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax1.legend(lines, labels, loc="upper right")
    ax1.set_title("Choosing k: elbow and silhouette")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def select_k(run_dir, k_min, k_max):
    """Compare k values, save the results and plot, and return the chosen k."""
    if k_min >= k_max:
        raise ValueError(f"k_min ({k_min}) must be smaller than k_max ({k_max})")
    train = pd.read_csv(Path(run_dir) / "train.csv")
    results = model.compare_k(train, k_min, k_max)
    choice = model.choose_k(results)
    pd.DataFrame(results).to_csv(Path(run_dir) / "k_comparison.csv", index=False)
    _plot_k_comparison(results, choice["k"], Path(run_dir) / "k_comparison.png")
    print(f"Chosen k: {choice['k']} (by {choice['method']}); best silhouette at k = {choice['best_silhouette_k']}")
    return choice


def train(run_dir, k):
    """Train the final model with the chosen k and save it."""
    train_df = pd.read_csv(Path(run_dir) / "train.csv")
    fitted = model.train_model(train_df, k)
    model_path = Path(run_dir) / "model.joblib"
    joblib.dump(fitted, model_path)
    print(f"Saved model with k = {k} to {model_path}")
    return str(model_path)


def evaluate(run_dir):
    """Score the saved model on the held-out test set."""
    fitted = joblib.load(Path(run_dir) / "model.joblib")
    test = pd.read_csv(Path(run_dir) / "test.csv")
    metrics = model.evaluate_model(fitted, test)
    _write_json(Path(run_dir) / "metrics.json", metrics)
    print(f"Test metrics: {metrics}")
    return metrics


def profile(run_dir):
    """Describe each cluster on the test set and save the table."""
    fitted = joblib.load(Path(run_dir) / "model.joblib")
    test = pd.read_csv(Path(run_dir) / "test.csv")
    table = model.cluster_profile(fitted, test)
    table.to_csv(Path(run_dir) / "cluster_profile.csv", index=False)
    print(table.to_string(index=False))
    return table.to_dict(orient="records")


def report(run_dir, validation, counts, choice, metrics, clusters):
    """Write a short Markdown summary of the whole run."""
    lines = [
        "# Dry Bean clustering run",
        "",
        f"- Raw rows: {validation['rows']} ({validation['duplicate_rows']} duplicates removed)",
        f"- Train / test rows: {counts['train_rows']} / {counts['test_rows']}",
        f"- Chosen k: {choice['k']} (by {choice['method']}), best silhouette at k = {choice['best_silhouette_k']}",
        f"- Test silhouette: {metrics['silhouette']:.3f}",
        f"- Test adjusted Rand index: {metrics['adjusted_rand_index']:.3f}",
        "",
        "| Cluster | Beans | Main class | Main class share |",
        "| ------- | ----- | ---------- | ---------------- |",
    ]
    lines += [
        f"| {c['cluster']} | {c['beans']} | {c['main_class']} | {c['main_class_share']:.1%} |"
        for c in clusters
    ]
    report_path = Path(run_dir) / "report.md"
    report_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return str(report_path)
"""Tests for the K-Means pipeline and model selection."""

import numpy as np

from src import model
from src.data import FEATURES, TARGET


def test_scaling_is_saved_inside_the_model(beans):
    # The original lab trained on scaled data but predicted on raw data.
    # Here, raw rows go straight into the model and get the same labels as in training.
    fitted = model.train_model(beans, k=4)
    assert np.array_equal(fitted.predict(beans[FEATURES]), fitted["kmeans"].labels_)


def test_final_model_uses_the_chosen_k(beans):
    # The original lab saved the k=49 model instead of the elbow's k.
    fitted = model.train_model(beans, k=3)
    assert fitted["kmeans"].n_clusters == 3


def test_compare_k_scores_every_k(beans):
    results = model.compare_k(beans, k_min=2, k_max=6)
    assert [r["k"] for r in results] == [2, 3, 4, 5, 6]
    sse = [r["sse"] for r in results]
    assert sse == sorted(sse, reverse=True)
    assert all(-1 <= r["silhouette"] <= 1 for r in results)


def test_elbow_finds_the_four_groups(beans):
    choice = model.choose_k(model.compare_k(beans, k_min=2, k_max=8))
    assert choice["k"] == 4
    assert choice["method"] == "elbow"
    assert choice["best_silhouette_k"] == 4


def test_falls_back_to_silhouette_when_no_elbow(monkeypatch):
    class NoElbow:
        def __init__(self, *args, **kwargs):
            self.elbow = None

    monkeypatch.setattr(model, "KneeLocator", NoElbow)
    results = [
        {"k": 2, "sse": 30.0, "silhouette": 0.3},
        {"k": 3, "sse": 20.0, "silhouette": 0.6},
        {"k": 4, "sse": 10.0, "silhouette": 0.4},
    ]
    assert model.choose_k(results) == {
        "k": 3, "method": "silhouette", "elbow_k": None, "best_silhouette_k": 3,
    }


def test_evaluation_on_perfectly_separated_groups(beans):
    metrics = model.evaluate_model(model.train_model(beans, k=4), beans)
    assert metrics["test_rows"] == len(beans)
    assert metrics["adjusted_rand_index"] == 1.0
    assert metrics["silhouette"] > 0.5


def test_cluster_profile_has_one_row_per_cluster(beans):
    profile = model.cluster_profile(model.train_model(beans, k=4), beans)
    assert len(profile) == 4
    assert profile["beans"].sum() == len(beans)
    assert set(profile["main_class"]) == set(beans[TARGET])
    assert (profile["main_class_share"] == 1.0).all()
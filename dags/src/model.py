"""Build, compare, and evaluate K-Means models for the Dry Bean data."""

import numpy as np
import pandas as pd
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.data import FEATURES, TARGET

# Size features are right-skewed (BOMBAY beans are much larger than the rest),
# so they are log-transformed before scaling. Shape features are ratios and stay as they are.
SIZE_FEATURES = [
    "Area", "Perimeter", "MajorAxisLength", "MinorAxisLength",
    "ConvexArea", "EquivDiameter",
]
SHAPE_FEATURES = [f for f in FEATURES if f not in SIZE_FEATURES]

SEED = 42
SILHOUETTE_SAMPLE = 3000  # silhouette is slow on all rows, so a fixed sample is used


def build_pipeline(k, seed=SEED):
    """Preprocessing and K-Means in one pipeline, so the same scaling is used for training and prediction."""
    preprocess = ColumnTransformer([
        ("log_size", FunctionTransformer(np.log, feature_names_out="one-to-one"), SIZE_FEATURES),
        ("shape", "passthrough", SHAPE_FEATURES),
    ])
    return Pipeline([
        ("preprocess", preprocess),
        ("scale", StandardScaler()),
        ("kmeans", KMeans(n_clusters=k, n_init=10, random_state=seed)),
    ])


def _silhouette(X_scaled, labels, seed=SEED):
    sample = min(SILHOUETTE_SAMPLE, len(X_scaled))
    return float(silhouette_score(X_scaled, labels, sample_size=sample, random_state=seed))


def compare_k(train, k_min=2, k_max=15, seed=SEED):
    """Fit K-Means for each k and return the SSE (for the elbow) and silhouette score."""
    results = []
    for k in range(k_min, k_max + 1):
        pipe = build_pipeline(k, seed).fit(train[FEATURES])
        X_scaled = pipe[:-1].transform(train[FEATURES])
        results.append({
            "k": k,
            "sse": float(pipe["kmeans"].inertia_),
            "silhouette": _silhouette(X_scaled, pipe["kmeans"].labels_, seed),
        })
    return results


def choose_k(results):
    """Pick k with the elbow method. Falls back to the best silhouette if no elbow is found."""
    ks = [r["k"] for r in results]
    sse = [r["sse"] for r in results]
    elbow = KneeLocator(ks, sse, curve="convex", direction="decreasing").elbow
    best_silhouette = max(results, key=lambda r: r["silhouette"])["k"]
    if elbow is None:
        return {"k": best_silhouette, "method": "silhouette", "elbow_k": None, "best_silhouette_k": best_silhouette}
    return {"k": int(elbow), "method": "elbow", "elbow_k": int(elbow), "best_silhouette_k": best_silhouette}


def train_model(train, k, seed=SEED):
    """Fit the final pipeline with the chosen k."""
    return build_pipeline(k, seed).fit(train[FEATURES])


def evaluate_model(model, test, seed=SEED):
    """Score the model on held-out beans it has never seen."""
    clusters = model.predict(test[FEATURES])
    X_scaled = model[:-1].transform(test[FEATURES])
    return {
        "test_rows": len(test),
        "silhouette": _silhouette(X_scaled, clusters, seed),
        # How well the clusters match the real bean types (1 = perfect, 0 = random)
        "adjusted_rand_index": float(adjusted_rand_score(test[TARGET], clusters)),
    }


def cluster_profile(model, df):
    """Size of each cluster, its most common bean type, and what share of the cluster that type makes up."""
    labeled = df.assign(cluster=model.predict(df[FEATURES]))
    counts = pd.crosstab(labeled["cluster"], labeled[TARGET])
    profile = pd.DataFrame({
        "beans": counts.sum(axis=1),
        "main_class": counts.idxmax(axis=1),
        "main_class_share": (counts.max(axis=1) / counts.sum(axis=1)).round(3),
    })
    means = labeled.groupby("cluster")[["Area", "AspectRatio", "Roundness"]].mean().round(3)
    return profile.join(means).reset_index()
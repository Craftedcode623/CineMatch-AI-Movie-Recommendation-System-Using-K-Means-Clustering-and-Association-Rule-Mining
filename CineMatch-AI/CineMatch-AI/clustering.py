"""
clustering.py
-------------
K-Means clustering of movies (Scikit-learn).

Idea: movies with similar rating level, popularity, release year and genres
end up in the same cluster. We then recommend movies from the same cluster.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


def scale_features(features):
    """Standardise features (mean 0, std 1) so no single feature dominates.

    Without scaling, 'year' (values ~2000) would completely overpower
    genre columns (0 or 1).
    """
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features.values)
    return pd.DataFrame(scaled, index=features.index, columns=features.columns)


def find_optimal_clusters(X_scaled, k_min=2, k_max=10):
    """Elbow Method + silhouette score for every k in [k_min, k_max].

    Returns a DataFrame with columns: k, inertia, silhouette
        inertia    = sum of squared distances to the nearest cluster centre
                     (lower is better, but always falls as k grows -> look for the "elbow")
        silhouette = how well separated the clusters are (-1 to 1, higher is better)
    """
    if len(X_scaled) <= k_max:
        raise ValueError("Not enough movies for this range of k.")
    rows = []
    for k in range(k_min, k_max + 1):
        model = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
        labels = model.fit_predict(X_scaled)
        rows.append({
            "k": k,
            "inertia": float(model.inertia_),
            "silhouette": float(silhouette_score(X_scaled, labels)),
        })
    return pd.DataFrame(rows)


def suggest_k(elbow_df, k_floor=4, tolerance=0.005):
    """Suggest a value of k using the silhouette score.

    - k below `k_floor` is ignored: k = 2 or 3 often scores highest but only
      splits movies into 2-3 huge groups, which is too coarse for recommendations.
    - Among the remaining values we take the SMALLEST k whose silhouette is
      within `tolerance` of the best one (simpler model, same quality).
    This is only a suggestion; the user can choose any k in the app.
    """
    candidates = elbow_df[elbow_df["k"] >= k_floor]
    if candidates.empty:
        candidates = elbow_df
    best = candidates["silhouette"].max()
    close = candidates[candidates["silhouette"] >= best - tolerance]
    return int(close["k"].min())


def perform_clustering(X_scaled, k):
    """Run K-Means with k clusters.

    Returns
    -------
    labels : pandas Series (index = movieId, value = cluster number)
    model  : fitted KMeans
    score  : silhouette score for this k
    """
    if k < 2 or k >= len(X_scaled):
        raise ValueError("k must be at least 2 and smaller than the number of movies.")
    model = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
    labels = model.fit_predict(X_scaled)
    score = float(silhouette_score(X_scaled, labels))
    return pd.Series(labels, index=X_scaled.index, name="cluster"), model, score


def add_cluster_labels(table, labels):
    """Attach the cluster number to the movie table (only clustered movies are kept)."""
    out = table.merge(labels.rename("cluster"), left_on="movieId", right_index=True, how="inner")
    return out


def summarize_clusters(clustered, genres, top_n=3):
    """Cluster-wise statistics + a readable name made from the top genres."""
    rows = []
    for cluster_id, group in clustered.groupby("cluster"):
        genre_share = {g: group["genre_list"].apply(lambda lst, g=g: g in lst).mean() for g in genres}
        top = sorted(genre_share, key=genre_share.get, reverse=True)[:top_n]
        rows.append({
            "cluster": cluster_id,
            "movies": len(group),
            "avg_rating": round(group["avg_rating"].mean(), 2),
            "avg_rating_count": round(group["rating_count"].mean(), 1),
            "avg_year": round(group["year"].mean()),
            "top_genres": ", ".join(top),
        })
    summary = pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)
    summary["name"] = summary.apply(
        lambda r: f"Cluster {r['cluster']}: {r['top_genres'].split(', ')[0]} / {r['top_genres'].split(', ')[1]}"
        if ", " in r["top_genres"] else f"Cluster {r['cluster']}", axis=1)
    return summary


def pca_projection(X_scaled, n_components=2):
    """Reduce features to 2 dimensions (PCA) so clusters can be plotted."""
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_scaled)
    df = pd.DataFrame(coords, index=X_scaled.index, columns=[f"PC{i + 1}" for i in range(n_components)])
    return df, [float(v) for v in pca.explained_variance_ratio_]

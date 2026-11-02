"""
recommendation.py
-----------------
The recommendation engine. Every function works on real tables produced by the
other modules - nothing is hard-coded.

Tables used
    table      : one row per movie (from data_processing.build_movie_table + add_weighted_rating)
    clustered  : table rows that were clustered, with a 'cluster' column
    X_scaled   : scaled K-Means feature matrix (index = movieId)
    rules      : Apriori rules (from association_rules.generate_association_rules)
"""

import numpy as np
import pandas as pd

from association_rules import rules_for_movie

DISPLAY_COLUMNS = ["movieId", "title", "genres", "year", "avg_rating", "rating_count"]


def _show(df, extra=None):
    cols = DISPLAY_COLUMNS + (extra or [])
    return df[[c for c in cols if c in df.columns]].reset_index(drop=True)


# ----------------------------------------------------------------------------
# 6. SEARCH
# ----------------------------------------------------------------------------
def search_movies(table, query, limit=50):
    """Case-insensitive title search. Returns best-rated matches first."""
    query = (query or "").strip()
    if not query:
        return pd.DataFrame(columns=DISPLAY_COLUMNS)
    found = table[table["title"].str.contains(query, case=False, regex=False, na=False)]
    found = found.sort_values("rating_count", ascending=False).head(limit)
    return _show(found)


def filter_movies(table, query="", genres=None, min_rating=0.0, min_ratings=0, year_range=None):
    """Filter used by the Explore page: title + genres + rating + year."""
    df = table
    if query and query.strip():
        df = df[df["title"].str.contains(query.strip(), case=False, regex=False, na=False)]
    if genres:
        df = df[df["genre_list"].apply(lambda lst: all(g in lst for g in genres))]
    df = df[(df["avg_rating"].fillna(0) >= min_rating) & (df["rating_count"] >= min_ratings)]
    if year_range:
        df = df[df["year"].between(year_range[0], year_range[1])]
    return df.sort_values(["weighted_rating"], ascending=False)


# ----------------------------------------------------------------------------
# 4. TOP RATED
# ----------------------------------------------------------------------------
def get_top_rated_movies(table, n=10, min_ratings=50, genre=None):
    """Highest average rating among movies with at least `min_ratings` ratings."""
    df = table[table["rating_count"] >= min_ratings]
    if genre:
        df = df[df["genre_list"].apply(lambda lst: genre in lst)]
    df = df.sort_values(["avg_rating", "rating_count"], ascending=False).head(n)
    return _show(df)


def get_most_rated_movies(table, n=10):
    """Movies with the largest number of ratings (most popular)."""
    return _show(table.sort_values("rating_count", ascending=False).head(n))


# ----------------------------------------------------------------------------
# 1. SIMILAR MOVIES (K-Means)
# ----------------------------------------------------------------------------
def recommend_similar_movies(movie_id, clustered, X_scaled, n=10):
    """Recommend movies from the SAME cluster as the selected movie.

    Inside the cluster, movies are ranked by Euclidean distance to the selected
    movie in the scaled feature space (smaller distance = more similar).

    Returns (DataFrame, message). DataFrame is empty if the movie cannot be used.
    """
    if movie_id not in X_scaled.index:
        return pd.DataFrame(), ("This movie has too few ratings to be clustered, so similar "
                                "movies cannot be found. Try a movie with more ratings.")
    cluster_id = clustered.loc[clustered["movieId"] == movie_id, "cluster"].iloc[0]
    members = clustered[(clustered["cluster"] == cluster_id) & (clustered["movieId"] != movie_id)].copy()
    if members.empty:
        return pd.DataFrame(), "No other movies in this cluster."
    target = X_scaled.loc[movie_id].values
    members["distance"] = np.linalg.norm(X_scaled.loc[members["movieId"]].values - target, axis=1)
    members = members.sort_values("distance").head(n)
    return _show(members, ["cluster", "distance"]), f"Showing movies from cluster {cluster_id}."


# ----------------------------------------------------------------------------
# 3. ASSOCIATION-BASED (Apriori)
# ----------------------------------------------------------------------------
def recommend_associated_movies(movie_id, rules, table, n=10):
    """Movies that users who liked `movie_id` also liked (from Apriori rules).

    Rules of the form {selected movie} -> {other movie} are used first. If the
    movie only appears together with other movies in the 'if' part, those rules
    are used instead. A movie that appears in several rules keeps its
    best (highest-lift) rule.
    Returns (DataFrame, message).
    """
    matches = rules_for_movie(rules, movie_id, single_only=True)
    message = "Based on Apriori rules: users who liked this movie also liked these."
    if matches.empty:
        matches = rules_for_movie(rules, movie_id)
        message = ("Based on Apriori rules where this movie is liked together with other movies "
                   "(no rule with this movie alone was found).")
    if matches.empty:
        return pd.DataFrame(), ("No association rule found for this movie with the current "
                                "settings. Try a movie from the 'Movies that appear in rules' list "
                                "or lower the support / confidence thresholds.")
    rows = []
    for _, rule in matches.iterrows():
        for consequent in rule["consequents"]:
            if consequent != movie_id:
                rows.append({"movieId": consequent, "support": rule["support"],
                             "confidence": rule["confidence"], "lift": rule["lift"]})
    rec = pd.DataFrame(rows).sort_values(["lift", "confidence"], ascending=False)
    rec = rec.drop_duplicates("movieId").head(n)
    rec = rec.merge(table, on="movieId", how="left")
    return _show(rec, ["support", "confidence", "lift"]), message


# ----------------------------------------------------------------------------
# 2. GENRE-BASED
# ----------------------------------------------------------------------------
def recommend_by_genre(table, genres, n=10, min_ratings=30, match_all=False):
    """Best movies (by weighted rating) in the chosen genres.

    match_all=False -> movie must contain at least one chosen genre
    match_all=True  -> movie must contain all chosen genres
    """
    if not genres:
        return pd.DataFrame(columns=DISPLAY_COLUMNS)
    check = all if match_all else any
    df = table[table["rating_count"] >= min_ratings]
    df = df[df["genre_list"].apply(lambda lst: check(g in lst for g in genres))]
    df = df.sort_values("weighted_rating", ascending=False).head(n)
    return _show(df, ["weighted_rating"])


# ----------------------------------------------------------------------------
# 5. PERSONALIZED
# ----------------------------------------------------------------------------
def recommend_personalized(table, genres, min_avg_rating=3.5, min_ratings=30,
                           year_range=None, n=10):
    """Recommendations from the user's stated preferences.

    Score = 0.6 * genre match + 0.4 * quality
        genre match = share of the user's chosen genres the movie contains
        quality     = weighted rating scaled to 0-1 (rating / 5)
    Only movies with avg rating >= min_avg_rating and enough ratings are considered.
    """
    if not genres:
        return pd.DataFrame(columns=DISPLAY_COLUMNS)
    df = table[(table["rating_count"] >= min_ratings) & (table["avg_rating"] >= min_avg_rating)].copy()
    if year_range:
        df = df[df["year"].between(year_range[0], year_range[1])]
    df["genre_match"] = df["genre_list"].apply(lambda lst: sum(g in lst for g in genres) / len(genres))
    df = df[df["genre_match"] > 0]
    df["score"] = 0.6 * df["genre_match"] + 0.4 * (df["weighted_rating"] / 5.0)
    df = df.sort_values("score", ascending=False).head(n)
    df["score"] = df["score"].round(3)
    return _show(df, ["genre_match", "score"])


def highlights(table, rules, id_to_title):
    """Small helper for the home page: one popular movie, one top-rated, strongest rule."""
    pop = get_most_rated_movies(table, 1).iloc[0]
    top = get_top_rated_movies(table, 1, min_ratings=50).iloc[0]
    out = {"popular": pop["title"], "top_rated": top["title"], "rule": None}
    if rules is not None and not rules.empty:
        r = rules.iloc[0]
        out["rule"] = (
            " + ".join(id_to_title.get(i, str(i)) for i in r["antecedents"]),
            " + ".join(id_to_title.get(i, str(i)) for i in r["consequents"]),
            float(r["lift"]),
        )
    return out

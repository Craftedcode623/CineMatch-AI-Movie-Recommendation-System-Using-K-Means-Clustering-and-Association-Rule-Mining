"""
data_processing.py
------------------
Everything related to loading and preparing the MovieLens data.

Main steps:
    1. load_data()               -> read the CSV files with Pandas
    2. clean_data()              -> remove duplicates / missing values, extract year and genres
    3. build_movie_table()       -> one row per movie with average rating and rating count
    4. prepare_movie_features()  -> numeric table used by K-Means
    5. prepare_transactions()    -> user "baskets" used by Apriori
"""

import os

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

NO_GENRE_LABEL = "(no genres listed)"


# ----------------------------------------------------------------------------
# 1. LOADING
# ----------------------------------------------------------------------------
def load_data(data_dir=DATA_DIR):
    """Load movies.csv, ratings.csv and (optionally) tags.csv and links.csv.

    Returns
    -------
    movies, ratings, tags, links : DataFrames (tags / links are empty if missing)
    """
    movies_path = os.path.join(data_dir, "movies.csv")
    ratings_path = os.path.join(data_dir, "ratings.csv")

    for path in (movies_path, ratings_path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required file not found: {path}\n"
                "Place movies.csv and ratings.csv from MovieLens 'ml-latest-small' "
                "inside the 'data' folder (or run: python download_data.py)."
            )

    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)

    # tags.csv and links.csv are optional
    tags_path = os.path.join(data_dir, "tags.csv")
    links_path = os.path.join(data_dir, "links.csv")
    tags = pd.read_csv(tags_path) if os.path.exists(tags_path) else pd.DataFrame()
    links = pd.read_csv(links_path) if os.path.exists(links_path) else pd.DataFrame()

    # Check that the expected columns exist
    _require_columns(movies, ["movieId", "title", "genres"], "movies.csv")
    _require_columns(ratings, ["userId", "movieId", "rating", "timestamp"], "ratings.csv")
    return movies, ratings, tags, links


def _require_columns(df, columns, name):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


# ----------------------------------------------------------------------------
# 2. CLEANING
# ----------------------------------------------------------------------------
def clean_data(movies, ratings, tags=None):
    """Clean the raw tables.

    - drop rows with missing values in important columns
    - drop duplicate movies and duplicate (user, movie) ratings
    - keep only ratings that belong to a known movie
    - extract release year from the title:  "Toy Story (1995)" -> 1995
    - split the genres string into a Python list
    - convert the timestamp to a date
    """
    movies = movies.dropna(subset=["movieId", "title"]).copy()
    movies["genres"] = movies["genres"].fillna(NO_GENRE_LABEL)
    movies = movies.drop_duplicates(subset="movieId")

    ratings = ratings.dropna(subset=["userId", "movieId", "rating"]).copy()
    ratings["date"] = pd.to_datetime(ratings["timestamp"], unit="s")
    # If a user rated the same movie twice, keep the latest rating
    ratings = ratings.sort_values("date").drop_duplicates(subset=["userId", "movieId"], keep="last")
    ratings = ratings[ratings["movieId"].isin(movies["movieId"])]
    ratings = ratings[(ratings["rating"] >= 0.5) & (ratings["rating"] <= 5.0)]

    # Year and clean title
    movies["year"] = movies["title"].str.extract(r"\((\d{4})\)\s*$")[0].astype(float)
    movies["title_clean"] = movies["title"].str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True).str.strip()

    # Genre list ("no genres listed" -> empty list)
    movies["genre_list"] = movies["genres"].apply(
        lambda g: [] if g == NO_GENRE_LABEL else g.split("|")
    )

    if tags is not None and len(tags) > 0:
        tags = tags.dropna(subset=["tag"]).drop_duplicates().copy()
    return movies.reset_index(drop=True), ratings.reset_index(drop=True), tags


# ----------------------------------------------------------------------------
# 3. MOVIE TABLE (movies + ratings merged)
# ----------------------------------------------------------------------------
def build_movie_table(movies, ratings):
    """Return one row per movie with avg_rating and rating_count.

    Movies that nobody rated get rating_count = 0 and avg_rating = NaN.
    """
    stats = ratings.groupby("movieId")["rating"].agg(avg_rating="mean", rating_count="count")
    table = movies.merge(stats, on="movieId", how="left")
    table["rating_count"] = table["rating_count"].fillna(0).astype(int)
    table["avg_rating"] = table["avg_rating"].round(3)
    return table


def get_genre_list(movies):
    """Sorted list of all distinct genres (without the 'no genres' label)."""
    genres = set()
    for lst in movies["genre_list"]:
        genres.update(lst)
    return sorted(genres)


def genre_counts(movies):
    """How many movies belong to each genre (a movie can have several)."""
    exploded = movies["genre_list"].explode().dropna()
    return exploded.value_counts().rename_axis("genre").reset_index(name="movie_count")


def add_weighted_rating(table, min_votes=None):
    """Add an IMDb-style 'weighted_rating' column.

    weighted = v/(v+m) * R + m/(v+m) * C
        R = movie's average rating, v = its number of ratings
        C = average rating of all movies, m = minimum votes needed (a "prior")
    It pulls movies with few ratings toward the overall mean, so a movie with
    two 5-star ratings does not beat a movie with 300 ratings averaging 4.4.
    """
    rated = table[table["rating_count"] > 0]
    c = rated["avg_rating"].mean()
    if min_votes is None:
        min_votes = rated["rating_count"].quantile(0.75)   # a sensible default
    v = table["rating_count"]
    r = table["avg_rating"].fillna(c)
    table = table.copy()
    table["weighted_rating"] = (v / (v + min_votes)) * r + (min_votes / (v + min_votes)) * c
    return table


# ----------------------------------------------------------------------------
# 4. FEATURES FOR K-MEANS
# ----------------------------------------------------------------------------
def prepare_movie_features(table, genres, min_ratings=10):
    """Build the numeric feature table used for clustering.

    Only movies with at least `min_ratings` ratings are used, because an
    average computed from 1-2 ratings is not reliable.

    Features:
        avg_rating        average rating (0.5 - 5)
        log_rating_count  log(1 + number of ratings)  -> popularity (log reduces skew)
        year              release year
        one 0/1 column per genre
    """
    use = table[(table["rating_count"] >= min_ratings) & table["year"].notna()].copy()
    if use.empty:
        raise ValueError("No movies left after filtering. Lower min_ratings.")

    features = pd.DataFrame(index=use["movieId"].values)
    features["avg_rating"] = use["avg_rating"].values
    features["log_rating_count"] = np.log1p(use["rating_count"].values)
    features["year"] = use["year"].values
    for g in genres:
        features[g] = use["genre_list"].apply(lambda lst, g=g: 1 if g in lst else 0).values
    features.index.name = "movieId"
    return features


# ----------------------------------------------------------------------------
# 5. TRANSACTIONS FOR APRIORI
# ----------------------------------------------------------------------------
def prepare_transactions(ratings, table, like_threshold=4.0, min_movie_ratings=50, min_liked=2):
    """Turn real ratings into Apriori "baskets".

    RULE: a movie is counted in a user's basket only if that user rated it
    >= `like_threshold` (default 4.0 = "the user liked it").
    Each USER is one transaction; the items are the movies they liked.

    To keep Apriori fast and meaningful we only use movies that have at least
    `min_movie_ratings` ratings, and users who liked at least `min_liked` of them.

    These are REAL transactions from the dataset - nothing is simulated.

    Returns
    -------
    basket : DataFrame of True/False, rows = users, columns = movieId
    """
    popular_ids = table.loc[table["rating_count"] >= min_movie_ratings, "movieId"]
    liked = ratings[(ratings["rating"] >= like_threshold) & ratings["movieId"].isin(popular_ids)]
    basket = pd.crosstab(liked["userId"], liked["movieId"]).astype(bool)
    basket = basket[basket.sum(axis=1) >= min_liked]
    return basket

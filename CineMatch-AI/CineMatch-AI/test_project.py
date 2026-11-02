"""
test_project.py
---------------
Simple tests for the backend. Run with:   python test_project.py

Each test prints PASS or FAIL. The Apriori test also re-calculates one rule by
hand from the raw ratings to prove the numbers are correct.
"""

import os
import sys
import traceback

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import association_rules as ar
import clustering as cl
import data_processing as dp
import recommendation as rec

# ---- shared setup (real data) -------------------------------------------------
movies_raw, ratings_raw, tags_raw, links_raw = dp.load_data()
movies, ratings, tags = dp.clean_data(movies_raw, ratings_raw, tags_raw)
table = dp.add_weighted_rating(dp.build_movie_table(movies, ratings))
genres = dp.get_genre_list(movies)
features = dp.prepare_movie_features(table, genres, 10)
X = cl.scale_features(features)
K = 6
labels, model, score = cl.perform_clustering(X, K)
clustered = cl.add_cluster_labels(table, labels)
basket = dp.prepare_transactions(ratings, table, 4.0, 50)
itemsets = ar.generate_frequent_itemsets(basket, 0.12, 3)
rules = ar.generate_association_rules(itemsets, len(basket), 0.6, 1.0)
id_to_title = dict(zip(table["movieId"], table["title"]))
TOY_STORY = int(table.loc[table["title"] == "Toy Story (1995)", "movieId"].iloc[0])


# ---- tests --------------------------------------------------------------------
def test_dataset_loading():
    assert len(movies_raw) > 9000 and len(ratings_raw) > 100000
    assert {"movieId", "title", "genres"} <= set(movies_raw.columns)
    assert {"userId", "movieId", "rating", "timestamp"} <= set(ratings_raw.columns)


def test_missing_file_error():
    try:
        dp.load_data("this_folder_does_not_exist")
    except FileNotFoundError:
        return
    raise AssertionError("expected FileNotFoundError")


def test_preprocessing():
    assert not movies.duplicated("movieId").any()
    assert not ratings.duplicated(["userId", "movieId"]).any()
    assert ratings["rating"].between(0.5, 5).all()
    assert table["rating_count"].sum() == len(ratings)
    # average rating of Toy Story must equal a manual calculation
    manual = ratings.loc[ratings["movieId"] == TOY_STORY, "rating"].mean()
    got = table.loc[table["movieId"] == TOY_STORY, "avg_rating"].iloc[0]
    assert abs(manual - got) < 0.001
    assert len(genres) == 19 and "Comedy" in genres
    assert table.loc[table["movieId"] == TOY_STORY, "year"].iloc[0] == 1995


def test_kmeans():
    assert len(labels) == len(X) == len(features)
    assert labels.nunique() == K
    assert -1 <= score <= 1
    elbow = cl.find_optimal_clusters(X, 2, 5)
    assert list(elbow["k"]) == [2, 3, 4, 5]
    assert elbow["inertia"].is_monotonic_decreasing          # inertia must fall as k grows
    assert 2 <= cl.suggest_k(elbow, k_floor=2) <= 5
    summary = cl.summarize_clusters(clustered, genres)
    assert summary["movies"].sum() == len(clustered)


def test_apriori_numbers_are_correct():
    assert len(rules) > 0
    row = rules.iloc[0]
    a, c = list(row["antecedents"]), list(row["consequents"])
    both = basket[a + c].all(axis=1).mean()                  # support, calculated by hand
    conf = basket[a + c].all(axis=1).sum() / basket[a].all(axis=1).sum()
    lift = conf / basket[c].all(axis=1).mean()
    assert abs(both - row["support"]) < 1e-9
    assert abs(conf - row["confidence"]) < 1e-9
    assert abs(lift - row["lift"]) < 1e-9
    assert (rules["confidence"] >= 0.6).all() and (rules["lift"] >= 1.0).all()


def test_transactions_follow_rule():
    # every True in the basket must correspond to a real rating >= 4.0
    liked = ratings[ratings["rating"] >= 4.0].set_index(["userId", "movieId"]).index
    sample = basket.stack()
    sample = sample[sample].index[:2000]
    assert all((u, m) in liked for u, m in sample)


def test_search():
    res = rec.search_movies(table, "toy story")
    assert len(res) >= 1 and res["title"].str.contains("Toy Story", case=False).all()
    assert rec.search_movies(table, "zzzzqqqq").empty
    assert rec.search_movies(table, "").empty


def test_genre_filter():
    res = rec.filter_movies(table, genres=["Comedy", "Romance"])
    assert len(res) > 0 and res["genre_list"].apply(lambda g: "Comedy" in g and "Romance" in g).all()
    res = rec.recommend_by_genre(table, ["Horror"], n=10, min_ratings=30)
    assert len(res) == 10 and res["genres"].str.contains("Horror").all() and (res["rating_count"] >= 30).all()
    assert rec.recommend_by_genre(table, []).empty


def test_top_rated_threshold():
    res = rec.get_top_rated_movies(table, 10, min_ratings=50)
    assert (res["rating_count"] >= 50).all()
    assert res["avg_rating"].is_monotonic_decreasing


def test_similar_recommendations():
    res, _ = rec.recommend_similar_movies(TOY_STORY, clustered, X, 10)
    assert len(res) == 10 and TOY_STORY not in res["movieId"].values
    toy_cluster = clustered.loc[clustered["movieId"] == TOY_STORY, "cluster"].iloc[0]
    assert (res["cluster"] == toy_cluster).all()
    assert res["distance"].is_monotonic_increasing
    # movie without enough ratings -> friendly message, no crash
    rare = int(table.loc[table["rating_count"] == 1, "movieId"].iloc[0])
    empty, msg = rec.recommend_similar_movies(rare, clustered, X, 5)
    assert empty.empty and len(msg) > 0


def test_association_recommendations():
    movie_id = int(list(rules.iloc[0]["antecedents"])[0])
    res, _ = rec.recommend_associated_movies(movie_id, rules, table, 5)
    assert len(res) > 0
    allowed = {c for s, cons in zip(rules["antecedents"], rules["consequents"]) if movie_id in s for c in cons}
    assert set(res["movieId"]) <= allowed                    # recommendations really come from rules
    empty, msg = rec.recommend_associated_movies(-1, rules, table, 5)
    assert empty.empty


def test_personalized():
    res = rec.recommend_personalized(table, ["Comedy", "Romance"], 3.5, 30, (1980, 2018), 10)
    assert len(res) == 10 and (res["avg_rating"] >= 3.5).all()
    assert rec.recommend_personalized(table, []).empty


# ---- runner -------------------------------------------------------------------
if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL  {t.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    sys.exit(1 if failed else 0)

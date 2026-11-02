"""
app.py  -  CineMatch AI (Streamlit front-end)
---------------------------------------------
Run with:   streamlit run app.py

This file only builds the user interface. All the real work (cleaning data,
K-Means, Apriori, recommendations, charts) is done in the other modules.
"""

import html
import os
import re

import pandas as pd
import streamlit as st

import association_rules as ar
import clustering as cl
import data_processing as dp
import posters
import recommendation as rec
import visualizations as viz

st.set_page_config(page_title="CineMatch AI", page_icon="🎬", layout="wide")

# ============================================================================
# STYLE (dark cinematic theme)
# ============================================================================
st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1300px;}
h1, h2, h3 {letter-spacing: -0.01em;}
.hero {padding: 2.2rem 2rem; border-radius: 18px; margin-bottom: 1.2rem;
       background: linear-gradient(120deg, #1a1a22 0%, #2a1a1d 60%, #3a2a08 100%);
       border: 1px solid #33333d;}
.hero h1 {margin: 0; font-size: 2.4rem;}
.hero p {color: #b9b9c4; margin: .5rem 0 0 0; font-size: 1.05rem;}
.accent {color: #F5B301;}
.grid {display: grid; grid-template-columns: repeat(auto-fill, minmax(165px, 1fr)); gap: 16px; margin: .4rem 0 1.2rem 0;}
.card {background: #17171d; border: 1px solid #2a2a33; border-radius: 12px; overflow: hidden;
       transition: transform .15s, border-color .15s;}
.card:hover {transform: translateY(-3px); border-color: #F5B301;}
.poster {height: 215px; position: relative; display: flex; align-items: center; justify-content: center;
         background-size: cover; background-position: center;}
.poster .initials {font-size: 2.6rem; font-weight: 800; color: rgba(255,255,255,.85);
                   text-shadow: 0 2px 10px rgba(0,0,0,.5);}
.poster .year {position: absolute; top: 8px; right: 8px; background: rgba(0,0,0,.6); color: #fff;
               font-size: .72rem; padding: 2px 8px; border-radius: 10px;}
.info {padding: 10px 12px 12px 12px;}
.info .t {font-weight: 700; font-size: .92rem; line-height: 1.25; height: 2.3em; overflow: hidden; color: #f2f2f5;}
.info .g {color: #8d8d99; font-size: .72rem; margin-top: 4px; height: 1.9em; overflow: hidden;}
.info .r {color: #F5B301; font-size: .82rem; margin-top: 6px; font-weight: 600;}
.info .r span {color: #8d8d99; font-weight: 400;}
.info .x {color: #7BD88F; font-size: .74rem; margin-top: 4px;}
.note {background: #1b1b22; border-left: 4px solid #F5B301; padding: .8rem 1rem; border-radius: 6px;
       color: #c9c9d2; font-size: .93rem; margin: .5rem 0 1rem 0;}
</style>
""", unsafe_allow_html=True)


# ============================================================================
# CACHED BACKEND CALLS  (the heavy work runs once, then results are reused)
# ============================================================================
@st.cache_data(show_spinner="Loading and cleaning the MovieLens data...")
def load_everything():
    movies, ratings, tags, links = dp.load_data()
    movies, ratings, tags = dp.clean_data(movies, ratings, tags)
    table = dp.add_weighted_rating(dp.build_movie_table(movies, ratings))

    # Make every title unique so it can be used in drop-down menus
    dup = table["title"].duplicated(keep=False)
    table["label"] = table["title"].where(~dup, table["title"] + " [id " + table["movieId"].astype(str) + "]")

    genres = dp.get_genre_list(movies)
    features = dp.prepare_movie_features(table, genres, min_ratings=10)
    X_scaled = cl.scale_features(features)
    return {
        "table": table, "ratings": ratings, "tags": tags, "links": links,
        "genres": genres, "genre_df": dp.genre_counts(movies),
        "X_scaled": X_scaled, "id_to_title": dict(zip(table["movieId"], table["title"])),
    }


@st.cache_data(show_spinner="Running the Elbow Method (K-Means for k = 2 ... 10)...")
def compute_elbow(_X_scaled):
    elbow = cl.find_optimal_clusters(_X_scaled, 2, 10)
    return elbow, cl.suggest_k(elbow)


@st.cache_data(show_spinner="Running K-Means clustering...")
def run_kmeans(k):
    data = load_everything()
    labels, model, score = cl.perform_clustering(data["X_scaled"], k)
    clustered = cl.add_cluster_labels(data["table"], labels)
    summary = cl.summarize_clusters(clustered, data["genres"])
    pca_df, explained = cl.pca_projection(data["X_scaled"])
    return {"clustered": clustered, "summary": summary, "score": score,
            "inertia": float(model.inertia_), "pca": pca_df, "explained": explained}


@st.cache_data(show_spinner="Running Apriori on user transactions...")
def run_apriori(like_threshold, min_movie_ratings, min_support, min_conf, min_lift):
    data = load_everything()
    basket = dp.prepare_transactions(data["ratings"], data["table"], like_threshold, min_movie_ratings)
    itemsets = ar.generate_frequent_itemsets(basket, min_support, max_len=3)
    rules = ar.generate_association_rules(itemsets, len(basket), min_conf, min_lift)
    return {
        "n_users": int(basket.shape[0]), "n_movies": int(basket.shape[1]),
        "itemsets": itemsets,
        "itemsets_readable": ar.itemsets_to_readable(itemsets, data["id_to_title"]),
        "rules": rules,
        "rules_readable": ar.rules_to_readable(rules, data["id_to_title"]),
    }


@st.cache_data(show_spinner=False)
def cached_poster(tmdb_id, api_key):
    return posters.fetch_tmdb_poster(tmdb_id, api_key)


# ============================================================================
# UI HELPERS
# ============================================================================
def render_cards(df, extra=None):
    """Draw movie cards. `extra` = {column: "label"} to show one more line on each card.

    Rows without a title are skipped so no broken/empty card is ever shown.
    """
    if df is None or df.empty:
        st.info("No movies to show.")
        return
    data = load_everything()
    links = data["links"]
    api_key = posters.get_api_key()
    tmdb_lookup = {}
    if api_key and not links.empty and "tmdbId" in links.columns:
        tmdb_lookup = dict(zip(links["movieId"], links["tmdbId"]))

    cards = []
    for _, row in df.iterrows():
        title = row.get("title")
        if not isinstance(title, str) or not title.strip():
            continue
        clean = re.sub(r"\s*\(\d{4}\)\s*$", "", title)
        initials = "".join(w[0] for w in re.findall(r"[A-Za-z0-9]+", clean)[:2]).upper() or "🎬"
        c1, c2 = posters.genre_colors(row.get("genres", ""))
        poster_url = None
        if api_key and row.get("movieId") in tmdb_lookup:
            poster_url = cached_poster(tmdb_lookup[row["movieId"]], api_key)
        if poster_url:
            style = f"background-image:url('{html.escape(poster_url)}')"
            inner = ""
        else:
            style = f"background:linear-gradient(160deg,{c1},{c2})"
            inner = f'<span class="initials">{html.escape(initials)}</span>'
        year = row.get("year")
        year_html = f'<span class="year">{int(year)}</span>' if pd.notna(year) else ""
        avg, cnt = row.get("avg_rating"), row.get("rating_count")
        rating_html = (f'★ {avg:.2f} <span>· {int(cnt):,} ratings</span>'
                       if pd.notna(avg) else '<span>No ratings yet</span>')
        extra_html = ""
        if extra:
            parts = [f"{lbl}: {row[col]:.2f}" if isinstance(row.get(col), float) else f"{lbl}: {row.get(col)}"
                     for col, lbl in extra.items() if col in row and pd.notna(row[col])]
            if parts:
                extra_html = f'<div class="x">{html.escape(" · ".join(parts))}</div>'
        genres = html.escape(str(row.get("genres", "")).replace("|", " · "))
        cards.append(
            f'<div class="card"><div class="poster" style="{style}">{inner}{year_html}</div>'
            f'<div class="info"><div class="t">{html.escape(clean)}</div><div class="g">{genres}</div>'
            f'<div class="r">{rating_html}</div>{extra_html}</div></div>'
        )
    if cards:
        st.markdown(f'<div class="grid">{"".join(cards)}</div>', unsafe_allow_html=True)
    else:
        st.info("No movies to show.")


def pick_movie(label, options_df, key, default_title=None):
    """Drop-down of movie titles. Returns the selected movieId or None."""
    if options_df.empty:
        st.warning("No movies available for selection.")
        return None
    labels = options_df.sort_values("rating_count", ascending=False)["label"].tolist()
    index = labels.index(default_title) if default_title in labels else 0
    chosen = st.selectbox(label, labels, index=index, key=key)
    row = options_df[options_df["label"] == chosen]
    return int(row["movieId"].iloc[0]) if not row.empty else None


def explain(text):
    st.markdown(f'<div class="note">{text}</div>', unsafe_allow_html=True)


# ============================================================================
# LOAD DATA (with friendly error messages)
# ============================================================================
try:
    D = load_everything()
except FileNotFoundError as err:
    st.error("Dataset not found.")
    st.code(str(err))
    st.stop()
except Exception as err:  # unexpected problem: show it instead of a crash page
    st.error(f"Could not load the data: {err}")
    st.stop()

TABLE, RATINGS, GENRES = D["table"], D["ratings"], D["genres"]
ELBOW, SUGGESTED_K = compute_elbow(D["X_scaled"])

# Default Apriori settings used by Home / Recommendations / Insights pages
AP_DEFAULT = dict(like_threshold=4.0, min_movie_ratings=50, min_support=0.12, min_conf=0.6, min_lift=1.0)

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================
st.sidebar.markdown("## 🎬 CineMatch <span class='accent'>AI</span>", unsafe_allow_html=True)
page = st.sidebar.radio("Navigate", [
    "🏠 Home", "🔎 Explore Movies", "🎯 Recommendations", "🧩 K-Means Analysis",
    "🔗 Apriori Analysis", "📊 Data Insights", "ℹ️ About Project"], label_visibility="collapsed")
st.sidebar.caption("Movie recommendations using K-Means clustering and Apriori association rules "
                   "on the MovieLens latest-small dataset.")
st.sidebar.caption("Posters: " + ("TMDB (API key found)" if posters.get_api_key()
                                  else "placeholders (no TMDB API key set)"))


# ============================================================================
# PAGE 1: HOME
# ============================================================================
if page == "🏠 Home":
    st.markdown('<div class="hero"><h1>Welcome to Cine<span class="accent">Match</span> AI</h1>'
                '<p>Discover movies using real data mining: K-Means clustering finds similar movies and '
                'Apriori finds movies that people like together.</p></div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Movies", f"{len(TABLE):,}")
    c2.metric("Ratings", f"{len(RATINGS):,}")
    c3.metric("Genres", len(GENRES))
    c4.metric("Average rating", f"{RATINGS['rating'].mean():.2f} / 5")

    st.subheader("🔥 Popular movies (most ratings)")
    render_cards(rec.get_most_rated_movies(TABLE, 6))

    st.subheader("🏆 Top rated (at least 50 ratings)")
    render_cards(rec.get_top_rated_movies(TABLE, 6, min_ratings=50))

    st.subheader("🆕 Newest releases in the dataset")
    st.caption("MovieLens has no 'date added' column, so this uses the release year from the title "
               "(movies with at least 10 ratings).")
    newest = TABLE[TABLE["rating_count"] >= 10].sort_values(["year", "rating_count"], ascending=False).head(6)
    render_cards(rec._show(newest))

    st.subheader("✨ Recommendation highlights")
    ap = run_apriori(**AP_DEFAULT)
    hl = rec.highlights(TABLE, ap["rules"], D["id_to_title"])
    h1, h2, h3 = st.columns(3)
    h1.markdown(f"**Most popular**\n\n{hl['popular']}")
    h2.markdown(f"**Highest rated (50+ ratings)**\n\n{hl['top_rated']}")
    if hl["rule"]:
        h3.markdown(f"**Strongest association rule**\n\nUsers who like *{hl['rule'][0]}* also like "
                    f"*{hl['rule'][1]}* (lift {hl['rule'][2]:.2f})")
    else:
        h3.markdown("**Strongest association rule**\n\nNo rule found with the default settings.")
    st.info("Go to **Recommendations** in the sidebar to get movie suggestions.")


# ============================================================================
# PAGE 2: EXPLORE
# ============================================================================
elif page == "🔎 Explore Movies":
    st.title("🔎 Explore Movies")
    f1, f2 = st.columns([2, 2])
    query = f1.text_input("Search by title", placeholder="e.g. matrix, toy story ...")
    sel_genres = f2.multiselect("Genres (movie must have all selected)", GENRES)
    f3, f4, f5 = st.columns(3)
    min_rating = f3.slider("Minimum average rating", 0.0, 5.0, 0.0, 0.5)
    min_count = f4.slider("Minimum number of ratings", 0, 200, 10, 5)
    y_min, y_max = int(TABLE["year"].min()), int(TABLE["year"].max())
    years = f5.slider("Release year", y_min, y_max, (y_min, y_max))

    results = rec.filter_movies(TABLE, query, sel_genres, min_rating, min_count, years)
    st.write(f"**{len(results):,}** movies match your filters.")
    if results.empty:
        st.warning("No movies found. Try removing a filter or lowering the minimum ratings.")
    else:
        show_n = st.slider("How many cards to show", 6, 60, 12, 6)
        render_cards(rec._show(results.head(show_n)))

        st.subheader("Movie details")
        chosen = st.selectbox("Select a movie to see its details", results.head(200)["label"].tolist())
        row = results[results["label"] == chosen].iloc[0]
        d1, d2 = st.columns([1, 2])
        with d1:
            render_cards(rec._show(results[results["label"] == chosen]))
        with d2:
            st.markdown(f"### {row['title']}")
            st.write(f"**Genres:** {row['genres'].replace('|', ', ')}")
            st.write(f"**Average rating:** {row['avg_rating']:.2f} / 5  \n"
                     f"**Number of ratings:** {int(row['rating_count']):,}  \n"
                     f"**Weighted rating:** {row['weighted_rating']:.2f}")
            tags = D["tags"]
            if tags is not None and len(tags) > 0:
                top_tags = tags[tags["movieId"] == row["movieId"]]["tag"].str.lower().value_counts().head(8)
                if len(top_tags):
                    st.write("**User tags:** " + ", ".join(top_tags.index))
            st.caption("Weighted rating = average rating pulled toward the global average when a movie "
                       "has few ratings (so 2 five-star votes cannot beat 300 votes averaging 4.4).")


# ============================================================================
# PAGE 3: RECOMMENDATIONS
# ============================================================================
elif page == "🎯 Recommendations":
    st.title("🎯 Recommendations")
    tabs = st.tabs(["Similar movies (K-Means)", "Liked together (Apriori)", "By genre", "Personalized", "Top rated"])

    with tabs[0]:
        explain("Movies are grouped by K-Means using rating, popularity, year and genres. "
                "We recommend the closest movies from the <b>same cluster</b>.")
        km = run_kmeans(SUGGESTED_K)
        pool = km["clustered"]
        mid = pick_movie("Select a movie you like", pool, "sim_movie", "Toy Story (1995)")
        n = st.slider("Number of recommendations", 3, 20, 10, key="sim_n")
        if mid is not None:
            res, msg = rec.recommend_similar_movies(mid, km["clustered"], D["X_scaled"], n)
            if res.empty:
                st.warning(msg)
            else:
                st.caption(f"{msg} (K-Means with k = {SUGGESTED_K})")
                render_cards(res, {"distance": "distance"})

    with tabs[1]:
        explain("Apriori looked at real users: a movie counts as 'liked' if the user rated it "
                f"{AP_DEFAULT['like_threshold']} or higher. Rules show which movies are liked together.")
        ap = run_apriori(**AP_DEFAULT)
        in_rules = set()
        if not ap["rules"].empty:
            for s in ap["rules"]["antecedents"]:
                in_rules.update(s)
        pool = TABLE[TABLE["movieId"].isin(in_rules)]
        if pool.empty:
            st.warning("No association rules were found with the default settings.")
        else:
            st.caption(f"{len(pool)} movies appear in at least one rule. Change thresholds on the Apriori page.")
            mid = pick_movie("Select a movie", pool, "assoc_movie")
            n = st.slider("Number of recommendations", 3, 20, 10, key="assoc_n")
            res, msg = rec.recommend_associated_movies(mid, ap["rules"], TABLE, n)
            if res.empty:
                st.warning(msg)
            else:
                st.caption(msg)
                render_cards(res, {"lift": "lift", "confidence": "confidence"})

    with tabs[2]:
        explain("Pick one or more genres. Movies are ranked by weighted rating (a fair score that "
                "accounts for the number of ratings).")
        g_sel = st.multiselect("Preferred genres", GENRES, default=["Sci-Fi"], key="rg_genres")
        g1, g2, g3 = st.columns(3)
        g_all = g1.checkbox("Movie must contain ALL selected genres", key="rg_all")
        g_min = g2.slider("Minimum ratings", 5, 200, 30, key="rg_min")
        g_n = g3.slider("How many", 3, 30, 10, key="rg_n")
        if g_sel:
            res = rec.recommend_by_genre(TABLE, g_sel, g_n, g_min, g_all)
            if res.empty:
                st.warning("No movies match. Try fewer genres or a lower minimum number of ratings.")
            else:
                render_cards(res)
        else:
            st.info("Select at least one genre.")

    with tabs[3]:
        explain("Tell us what you like. Score = 60% genre match + 40% quality (weighted rating).")
        p_gen = st.multiselect("Your favourite genres", GENRES, default=["Comedy", "Romance"], key="pp_genres")
        p1, p2 = st.columns(2)
        p_rating = p1.slider("Only movies rated at least", 1.0, 5.0, 3.5, 0.1, key="pp_rating")
        p_min = p2.slider("Minimum number of ratings", 5, 200, 30, key="pp_min")
        y_min, y_max = int(TABLE["year"].min()), int(TABLE["year"].max())
        p_years = st.slider("Release years", y_min, y_max, (1980, y_max), key="pp_years")
        if p_gen:
            res = rec.recommend_personalized(TABLE, p_gen, p_rating, p_min, p_years, 12)
            if res.empty:
                st.warning("No movies match these preferences. Try lowering the minimum rating.")
            else:
                render_cards(res, {"score": "score"})
        else:
            st.info("Select at least one genre.")

    with tabs[4]:
        t_min = st.slider("Minimum number of ratings", 10, 300, 50, key="top_min")
        t_gen = st.selectbox("Genre (optional)", ["All"] + GENRES, key="top_gen")
        res = rec.get_top_rated_movies(TABLE, 12, t_min, None if t_gen == "All" else t_gen)
        if res.empty:
            st.warning("No movies meet this threshold.")
        else:
            render_cards(res)


# ============================================================================
# PAGE 4: K-MEANS
# ============================================================================
elif page == "🧩 K-Means Analysis":
    st.title("🧩 K-Means Clustering")
    explain("<b>K-Means</b> puts movies into <i>k</i> groups so that movies in the same group are similar. "
            "Each movie is described by: average rating, number of ratings (log), release year and 19 genre "
            "columns (0/1). Features are standardised with StandardScaler. Only movies with at least "
            "10 ratings are used.")

    st.subheader("1. Choosing k: Elbow Method and silhouette score")
    k = st.slider("Number of clusters (k)", 2, 10, SUGGESTED_K,
                  help=f"Suggested k = {SUGGESTED_K} (best silhouette among k>=4, smallest k within 0.005 of the best)")
    e1, e2 = st.columns(2)
    e1.plotly_chart(viz.elbow_chart(ELBOW, k), width="stretch")
    e2.plotly_chart(viz.silhouette_chart(ELBOW, k), width="stretch")
    st.caption("The elbow curve falls smoothly without a sharp bend, and silhouette scores are low. "
               "This is common for movie data with many 0/1 genre columns: groups overlap. "
               "So k is a practical choice, not a perfect one. See the table below for exact values.")
    with st.expander("Show Elbow / silhouette values"):
        st.dataframe(ELBOW.round(3), hide_index=True, width="stretch")

    km = run_kmeans(k)
    st.subheader(f"2. Result for k = {k}")
    m1, m2, m3 = st.columns(3)
    m1.metric("Silhouette score", f"{km['score']:.3f}")
    m2.metric("Inertia", f"{km['inertia']:,.0f}")
    m3.metric("Movies clustered", f"{len(km['clustered']):,}")

    c1, c2 = st.columns(2)
    c1.plotly_chart(viz.cluster_distribution_chart(km["summary"]), width="stretch")
    c2.plotly_chart(viz.cluster_stats_chart(km["summary"]), width="stretch")

    st.plotly_chart(viz.cluster_scatter_chart(km["pca"], km["clustered"], km["explained"]), width="stretch")
    st.caption("The 22 features are squeezed into 2 dimensions with PCA only for drawing. "
               "Clusters overlap in 2-D because some information is lost in this projection.")

    st.subheader("3. Cluster-wise statistics")
    st.dataframe(km["summary"].drop(columns=["name"]), hide_index=True, width="stretch")

    st.subheader("4. Movies in a cluster")
    options = {row["name"]: row["cluster"] for _, row in km["summary"].iterrows()}
    chosen = st.selectbox("Select a cluster", list(options.keys()))
    members = km["clustered"][km["clustered"]["cluster"] == options[chosen]]
    st.write(f"{len(members):,} movies. Showing the 12 most-rated.")
    render_cards(rec._show(members.sort_values("rating_count", ascending=False).head(12)))
    with st.expander("Show full table of this cluster"):
        st.dataframe(rec._show(members.sort_values("rating_count", ascending=False)),
                     hide_index=True, width="stretch")


# ============================================================================
# PAGE 5: APRIORI
# ============================================================================
elif page == "🔗 Apriori Analysis":
    st.title("🔗 Apriori Association Rules")
    explain("<b>Apriori</b> finds groups of movies that many users like together. "
            "Each <b>user</b> is one transaction. A movie is in the user's basket only if the user "
            "rated it at or above the 'liked' threshold. <b>Support</b> = share of users who liked all movies "
            "in the rule. <b>Confidence</b> = of users who liked the 'if' movies, the share who also liked the "
            "'then' movie. <b>Lift</b> &gt; 1 = liked together more often than by chance.")
    st.caption("Transactions are REAL user ratings from the dataset (nothing simulated). "
               "Only movies with enough ratings are used, otherwise Apriori finds nothing reliable.")

    a1, a2, a3 = st.columns(3)
    like_thr = a1.slider("A rating counts as 'liked' if it is at least", 3.0, 5.0, 4.0, 0.5)
    min_mr = a2.slider("Use only movies with at least this many ratings", 20, 150, 50, 10)
    min_sup = a3.slider("Minimum support", 0.05, 0.40, 0.12, 0.01)
    b1, b2 = st.columns(2)
    min_conf = b1.slider("Minimum confidence", 0.3, 0.95, 0.6, 0.05)
    min_lift = b2.slider("Minimum lift", 1.0, 3.0, 1.0, 0.1)

    try:
        ap = run_apriori(like_thr, min_mr, min_sup, min_conf, min_lift)
    except ValueError as err:
        st.error(f"Could not run Apriori: {err}")
        st.stop()
    except MemoryError:
        st.error("Too many itemsets. Increase the minimum support.")
        st.stop()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Transactions (users)", ap["n_users"])
    m2.metric("Distinct movies", ap["n_movies"])
    m3.metric("Frequent itemsets", len(ap["itemsets"]))
    m4.metric("Rules", len(ap["rules"]))

    if ap["itemsets"].empty:
        st.warning("No frequent itemsets with this support. Lower the minimum support.")
        st.stop()

    st.subheader("1. Frequent itemsets")
    st.plotly_chart(viz.frequent_itemsets_chart(ap["itemsets_readable"]), width="stretch")
    with st.expander("Show all frequent itemsets"):
        st.dataframe(ap["itemsets_readable"], hide_index=True, width="stretch")

    st.subheader("2. Association rules")
    if ap["rules"].empty:
        st.warning("No rules satisfy these confidence / lift thresholds. Lower them or lower the support.")
    else:
        rr = ap["rules_readable"]
        st.dataframe(rr, hide_index=True, width="stretch", height=380)
        v1, v2 = st.columns(2)
        v1.plotly_chart(viz.rules_scatter_chart(rr), width="stretch")
        v2.plotly_chart(viz.top_rules_chart(rr), width="stretch")

        st.subheader("3. Explore a movie")
        in_rules = set()
        for s in ap["rules"]["antecedents"]:
            in_rules.update(s)
        pool = TABLE[TABLE["movieId"].isin(in_rules)]
        mid = pick_movie("Movies that appear in rules", pool, "ap_movie")
        if mid is not None:
            res, msg = rec.recommend_associated_movies(mid, ap["rules"], TABLE, 10)
            if res.empty:
                st.warning(msg)
            else:
                st.caption("Users who liked this movie also liked:")
                render_cards(res, {"lift": "lift", "confidence": "confidence"})


# ============================================================================
# PAGE 6: INSIGHTS
# ============================================================================
elif page == "📊 Data Insights":
    st.title("📊 Data Insights")
    i1, i2 = st.columns(2)
    i1.plotly_chart(viz.genre_distribution_chart(D["genre_df"]), width="stretch")
    i2.plotly_chart(viz.rating_distribution_chart(RATINGS), width="stretch")

    st.plotly_chart(viz.ratings_per_movie_chart(TABLE), width="stretch")

    min_c = st.slider("Minimum ratings for the Top 10 chart", 10, 300, 50)
    t1, t2 = st.columns(2)
    t1.plotly_chart(viz.top_movies_chart(rec.get_top_rated_movies(TABLE, 10, min_c),
                                         f"Top 10 rated (at least {min_c} ratings)"), width="stretch")
    t2.plotly_chart(viz.most_rated_chart(rec.get_most_rated_movies(TABLE, 10)), width="stretch")

    st.subheader("Cluster analysis")
    km = run_kmeans(SUGGESTED_K)
    st.caption(f"K-Means with k = {SUGGESTED_K} (silhouette {km['score']:.3f}).")
    k1, k2 = st.columns(2)
    k1.plotly_chart(viz.cluster_distribution_chart(km["summary"]), width="stretch")
    k2.plotly_chart(viz.cluster_stats_chart(km["summary"]), width="stretch")
    st.dataframe(km["summary"].drop(columns=["name"]), hide_index=True, width="stretch")

    st.subheader("Movie association insights")
    ap = run_apriori(**AP_DEFAULT)
    if ap["rules"].empty:
        st.info("No association rules with the default settings.")
    else:
        st.caption(f"Default Apriori settings: liked = rating >= {AP_DEFAULT['like_threshold']}, "
                   f"support >= {AP_DEFAULT['min_support']}, confidence >= {AP_DEFAULT['min_conf']}.")
        st.plotly_chart(viz.top_rules_chart(ap["rules_readable"]), width="stretch")


# ============================================================================
# PAGE 7: ABOUT
# ============================================================================
else:
    st.title("ℹ️ About CineMatch AI")
    st.markdown("""
**CineMatch AI** is a movie recommendation system built as a BCA Data Mining project. It uses two data mining
techniques on the MovieLens *latest-small* dataset (GroupLens Research, University of Minnesota).

### Objectives
- Clean and prepare real movie-rating data.
- Group similar movies with **K-Means clustering**.
- Find movies that users like together with **Apriori association rules**.
- Turn these results into working recommendations in a simple web app.

### Technologies
Python, Pandas, NumPy, Scikit-learn, mlxtend, Plotly, Streamlit.

### Algorithms
- **K-Means** (Scikit-learn) with StandardScaler, Elbow Method (inertia) and silhouette score.
- **Apriori** (mlxtend) with support, confidence and lift.
- **Weighted rating** (IMDb-style) so movies with very few ratings do not rank unfairly high.

### Limitations
- The dataset is small (about 610 users, 9.7k movies) and ends in 2018, so recent movies are missing.
- K-Means silhouette scores are low: movie groups overlap, so clusters are approximate.
- Apriori uses only popular movies and users' "liked" ratings (>= 4.0), so rules cover popular titles.
- Association is not causation: a rule says movies are liked together, not why.
- No poster images come with the dataset; placeholders are used unless a TMDB key is configured.
- Recommendations are not personalised per user account (no login).

### Future scope
Collaborative filtering, content features from tags, the larger MovieLens datasets, user accounts,
and a proper evaluation of recommendation quality.

*Dataset citation: F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context.
ACM TiiS 5, 4, Article 19.*
""")
    for name, caption in (("architecture.png", "System architecture"), ("workflow.png", "Workflow")):
        path = os.path.join("assets", name)
        if os.path.exists(path):
            st.image(path, caption=caption)

"""
visualizations.py
-----------------
All Plotly charts. Every function receives real calculated data and returns a
Plotly figure; app.py only has to call st.plotly_chart(fig).
"""

import plotly.express as px
import plotly.graph_objects as go

ACCENT = "#F5B301"      # gold
ACCENT_2 = "#E5484D"    # red
BG = "rgba(0,0,0,0)"
TEXT = "#E8E8EC"
GRID = "#2A2A32"
PALETTE = ["#F5B301", "#E5484D", "#4CC9F0", "#7BD88F", "#B388FF",
           "#FF8A5B", "#4DD0B8", "#F06292", "#9CCC65", "#64B5F6"]


def _style(fig, title=None, height=420):
    """Apply the same dark look to every chart."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color=TEXT)) if title else None,
        paper_bgcolor=BG, plot_bgcolor=BG, font=dict(color=TEXT, family="Inter, sans-serif"),
        height=height, margin=dict(l=20, r=20, t=60 if title else 20, b=20),
        legend=dict(bgcolor=BG),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig


# ---------------------------------------------------------------- data insights
def genre_distribution_chart(genre_df):
    df = genre_df.sort_values("movie_count")
    fig = px.bar(df, x="movie_count", y="genre", orientation="h",
                 color="movie_count", color_continuous_scale=["#5a3d00", ACCENT])
    fig.update_layout(coloraxis_showscale=False, xaxis_title="Number of movies", yaxis_title="")
    return _style(fig, "Movies per genre", height=520)


def rating_distribution_chart(ratings):
    counts = ratings["rating"].value_counts().sort_index().reset_index()
    counts.columns = ["rating", "count"]
    fig = px.bar(counts, x="rating", y="count", color_discrete_sequence=[ACCENT])
    fig.update_layout(xaxis_title="Rating given by users", yaxis_title="Number of ratings")
    fig.update_xaxes(dtick=0.5)
    return _style(fig, "Distribution of all ratings")


def top_movies_chart(top_df, title="Top rated movies"):
    df = top_df.sort_values("avg_rating")
    fig = px.bar(df, x="avg_rating", y="title", orientation="h", color="avg_rating",
                 color_continuous_scale=["#5a3d00", ACCENT], hover_data=["rating_count", "genres"])
    fig.update_layout(coloraxis_showscale=False, xaxis_title="Average rating", yaxis_title="",
                      xaxis_range=[min(3.5, df["avg_rating"].min() - 0.1), 5])
    return _style(fig, title, height=460)


def most_rated_chart(df):
    df = df.sort_values("rating_count")
    fig = px.bar(df, x="rating_count", y="title", orientation="h", color_discrete_sequence=[ACCENT_2],
                 hover_data=["avg_rating"])
    fig.update_layout(xaxis_title="Number of ratings", yaxis_title="")
    return _style(fig, "Most rated movies", height=460)


def ratings_per_movie_chart(table):
    rated = table[table["rating_count"] > 0]
    fig = px.histogram(rated, x="rating_count", nbins=60, log_y=True, color_discrete_sequence=[ACCENT])
    fig.update_layout(xaxis_title="Number of ratings a movie received",
                      yaxis_title="Number of movies (log scale)")
    return _style(fig, "Most movies have very few ratings")


# ---------------------------------------------------------------------- K-Means
def elbow_chart(elbow_df, chosen_k=None):
    fig = go.Figure(go.Scatter(x=elbow_df["k"], y=elbow_df["inertia"], mode="lines+markers",
                               line=dict(color=ACCENT, width=3), marker=dict(size=9),
                               hovertemplate="k=%{x}<br>inertia=%{y:,.0f}<extra></extra>"))
    if chosen_k is not None:
        fig.add_vline(x=chosen_k, line_dash="dash", line_color=ACCENT_2,
                      annotation_text=f"selected k = {chosen_k}")
    fig.update_layout(xaxis_title="Number of clusters (k)", yaxis_title="Inertia")
    fig.update_xaxes(dtick=1)
    return _style(fig, "Elbow Method")


def silhouette_chart(elbow_df, chosen_k=None):
    fig = go.Figure(go.Scatter(x=elbow_df["k"], y=elbow_df["silhouette"], mode="lines+markers",
                               line=dict(color="#4CC9F0", width=3), marker=dict(size=9),
                               hovertemplate="k=%{x}<br>silhouette=%{y:.3f}<extra></extra>"))
    if chosen_k is not None:
        fig.add_vline(x=chosen_k, line_dash="dash", line_color=ACCENT_2,
                      annotation_text=f"selected k = {chosen_k}")
    fig.update_layout(xaxis_title="Number of clusters (k)", yaxis_title="Silhouette score")
    fig.update_xaxes(dtick=1)
    return _style(fig, "Silhouette score for each k")


def cluster_distribution_chart(summary):
    df = summary.copy()
    df["label"] = df["cluster"].astype(str)
    fig = px.bar(df, x="label", y="movies", color="label", color_discrete_sequence=PALETTE,
                 hover_data=["top_genres", "avg_rating"])
    fig.update_layout(showlegend=False, xaxis_title="Cluster", yaxis_title="Number of movies")
    return _style(fig, "Movies in each cluster")


def cluster_scatter_chart(pca_df, clustered, explained):
    df = pca_df.join(clustered.set_index("movieId")[["title", "genres", "avg_rating", "cluster"]])
    df["cluster"] = df["cluster"].astype(str)
    fig = px.scatter(df, x="PC1", y="PC2", color="cluster", color_discrete_sequence=PALETTE,
                     hover_name="title", hover_data={"genres": True, "avg_rating": True, "PC1": False, "PC2": False},
                     opacity=0.75)
    fig.update_traces(marker=dict(size=6))
    fig.update_layout(xaxis_title=f"PC1 ({explained[0] * 100:.0f}% of variance)",
                      yaxis_title=f"PC2 ({explained[1] * 100:.0f}% of variance)")
    return _style(fig, "Movie clusters (2-D view using PCA)", height=560)


def cluster_stats_chart(summary):
    df = summary.copy()
    df["cluster"] = df["cluster"].astype(str)
    fig = px.bar(df, x="cluster", y="avg_rating", color="cluster", color_discrete_sequence=PALETTE,
                 hover_data=["top_genres", "movies"])
    fig.update_layout(showlegend=False, xaxis_title="Cluster", yaxis_title="Average rating",
                      yaxis_range=[0, 5])
    return _style(fig, "Average rating of each cluster")


# ---------------------------------------------------------------------- Apriori
def frequent_itemsets_chart(readable_itemsets, top_n=15):
    df = readable_itemsets.head(top_n).copy()
    df["movies"] = df["movies"].str.slice(0, 70)
    df = df.sort_values("support")
    fig = px.bar(df, x="support", y="movies", orientation="h", color="length",
                 color_continuous_scale=["#5a3d00", ACCENT_2], labels={"length": "Items"})
    fig.update_layout(xaxis_title="Support (share of users)", yaxis_title="")
    return _style(fig, f"Top {len(df)} frequent itemsets", height=520)


def rules_scatter_chart(rules_readable):
    fig = px.scatter(rules_readable, x="support", y="confidence", color="lift", size="lift",
                     color_continuous_scale=["#4CC9F0", ACCENT, ACCENT_2],
                     hover_data=["if_user_likes", "then_also_likes", "lift"])
    fig.update_layout(xaxis_title="Support", yaxis_title="Confidence")
    return _style(fig, "Association rules: support vs confidence (colour = lift)", height=480)


def top_rules_chart(rules_readable, top_n=10):
    df = rules_readable.head(top_n).copy()
    df["rule"] = (df["if_user_likes"].str.slice(0, 35) + "  →  " + df["then_also_likes"].str.slice(0, 35))
    df = df.sort_values("lift")
    fig = px.bar(df, x="lift", y="rule", orientation="h", color="confidence",
                 color_continuous_scale=["#5a3d00", ACCENT], hover_data=["support", "confidence"])
    fig.update_layout(xaxis_title="Lift", yaxis_title="")
    return _style(fig, f"Top {len(df)} rules by lift", height=480)

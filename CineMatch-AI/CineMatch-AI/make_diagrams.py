"""
make_diagrams.py
Creates assets/architecture.png and assets/workflow.png (used on the About page and README).

Run:  python make_diagrams.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
BG, BOX, EDGE, TEXT, GOLD = "#0e0e12", "#1d1d26", "#F5B301", "#E8E8EC", "#F5B301"


def box(ax, x, y, w, h, title, sub="", edge=EDGE):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                                fc=BOX, ec=edge, lw=1.6))
    ax.text(x + w / 2, y + h / 2 + (0.18 if sub else 0), title, ha="center", va="center",
            color=TEXT, fontsize=11, fontweight="bold")
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.28, sub, ha="center", va="center", color="#a9a9b6", fontsize=8.5)


def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=GOLD, lw=1.8))


def canvas(w, h, title):
    fig, ax = plt.subplots(figsize=(w, h), facecolor=BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    ax.text(w / 2, h - 0.35, title, ha="center", color=TEXT, fontsize=15, fontweight="bold")
    return fig, ax


def architecture():
    fig, ax = canvas(13, 7.2, "CineMatch AI - System Architecture")
    box(ax, 4.7, 5.7, 3.6, 1.0, "User (browser)", "chooses page, movie, genres, settings")
    box(ax, 4.2, 3.9, 4.6, 1.1, "app.py  (Streamlit front-end)", "7 pages, movie cards, caching, messages")
    box(ax, 0.3, 3.9, 3.0, 1.1, "visualizations.py", "Plotly charts")
    box(ax, 9.7, 3.9, 3.0, 1.1, "posters.py", "placeholders / optional TMDB")
    box(ax, 0.3, 1.5, 2.8, 1.3, "data_processing.py", "load, clean, features,\ntransactions")
    box(ax, 3.4, 1.5, 2.8, 1.3, "clustering.py", "StandardScaler, K-Means,\nelbow, silhouette, PCA")
    box(ax, 6.5, 1.5, 2.9, 1.3, "association_rules.py", "Apriori (mlxtend), support,\nconfidence, lift")
    box(ax, 9.7, 1.5, 3.0, 1.3, "recommendation.py", "similar, associated, genre,\npersonalized, top rated")
    box(ax, 0.3, 0.1, 2.8, 0.8, "data/*.csv", "MovieLens latest-small")
    arrow(ax, 6.5, 5.7, 6.5, 5.0)
    arrow(ax, 6.5, 5.0, 6.5, 5.7)
    arrow(ax, 4.2, 4.45, 3.3, 4.45)
    arrow(ax, 8.8, 4.45, 9.7, 4.45)
    # app.py -> backend modules (bus line drawn just below app.py)
    ax.plot([1.7, 11.2], [3.45, 3.45], color=GOLD, lw=1.8)
    ax.plot([6.5, 6.5], [3.9, 3.45], color=GOLD, lw=1.8)
    for x in (1.7, 4.8, 7.95, 11.2):
        arrow(ax, x, 3.45, x, 2.8)
    arrow(ax, 1.7, 1.5, 1.7, 0.9)
    fig.savefig(os.path.join(OUT, "architecture.png"), dpi=150, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


def workflow():
    fig, ax = canvas(18.6, 6.0, "CineMatch AI - Workflow")
    w, h = 2.2, 1.3
    xs = [0.2, 2.85, 5.5, 8.15, 10.8, 13.45, 16.1]
    mid = 2.35
    box(ax, xs[0], mid - h / 2, w, h, "1. Load CSVs", "movies, ratings,\ntags, links")
    box(ax, xs[1], mid - h / 2, w, h, "2. Clean", "duplicates, missing,\nyear, genres")
    box(ax, xs[2], mid - h / 2, w, h, "3. Movie table", "avg rating,\nrating count")
    top, bot = 3.9, 0.8
    box(ax, xs[3], top - h / 2, w, h, "4a. Features", "rating, popularity,\nyear, 19 genres")
    box(ax, xs[3], bot - h / 2, w, h, "4b. Transactions", "user baskets: movies\nrated >= 4.0")
    box(ax, xs[4], top - h / 2, w, h, "5a. K-Means", "scale, elbow, silhouette,\nlabel clusters")
    box(ax, xs[4], bot - h / 2, w, h, "5b. Apriori", "itemsets and rules:\nsupport/confidence/lift")
    box(ax, xs[5], mid - h / 2, w, h, "6. Recommender", "similar, associated,\ngenre, personalized")
    box(ax, xs[6], mid - h / 2, w, h, "7. Streamlit app", "cards, Plotly charts,\n7 pages")
    arrow(ax, xs[0] + w, mid, xs[1], mid)
    arrow(ax, xs[1] + w, mid, xs[2], mid)
    arrow(ax, xs[2] + w, mid + 0.2, xs[3], top - 0.2)
    arrow(ax, xs[2] + w, mid - 0.2, xs[3], bot + 0.2)
    arrow(ax, xs[3] + w, top, xs[4], top)
    arrow(ax, xs[3] + w, bot, xs[4], bot)
    arrow(ax, xs[4] + w, top - 0.2, xs[5], mid + 0.2)
    arrow(ax, xs[4] + w, bot + 0.2, xs[5], mid - 0.2)
    arrow(ax, xs[5] + w, mid, xs[6], mid)
    fig.savefig(os.path.join(OUT, "workflow.png"), dpi=150, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    architecture()
    workflow()
    print("Saved assets/architecture.png and assets/workflow.png")

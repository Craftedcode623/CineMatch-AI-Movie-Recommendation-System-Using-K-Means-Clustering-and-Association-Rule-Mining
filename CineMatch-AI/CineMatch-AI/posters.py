"""
posters.py
----------
Movie poster handling.

Posters are NOT part of the MovieLens data. Two modes:

1. No API key (default)  -> a genre-coloured placeholder card is drawn with CSS.
                            No internet connection and no external service needed.
2. TMDB API key set      -> the poster is requested from The Movie Database (TMDB)
                            using the tmdbId stored in links.csv.

To switch on mode 2, get a free key from https://www.themoviedb.org/settings/api and set it:
    Windows PowerShell:   $env:TMDB_API_KEY = "your_key_here"
    or create .streamlit/secrets.toml containing:   TMDB_API_KEY = "your_key_here"

NOTE: Mode 2 depends on an external service and a personal key. It has not been
verified in the development environment (no key was available), so the app must
and does work fully without it.
"""

import os

import requests

TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w342"

# One colour pair per genre (used for placeholder posters)
GENRE_COLORS = {
    "Action": ("#7f1d1d", "#ef4444"), "Adventure": ("#78350f", "#f59e0b"),
    "Animation": ("#134e4a", "#2dd4bf"), "Children": ("#831843", "#f472b6"),
    "Comedy": ("#713f12", "#facc15"), "Crime": ("#1f2937", "#9ca3af"),
    "Documentary": ("#1e3a5f", "#60a5fa"), "Drama": ("#3b0764", "#a78bfa"),
    "Fantasy": ("#4c1d95", "#c084fc"), "Film-Noir": ("#111827", "#6b7280"),
    "Horror": ("#450a0a", "#dc2626"), "IMAX": ("#0c4a6e", "#38bdf8"),
    "Musical": ("#701a75", "#e879f9"), "Mystery": ("#1e1b4b", "#818cf8"),
    "Romance": ("#881337", "#fb7185"), "Sci-Fi": ("#082f49", "#22d3ee"),
    "Thriller": ("#27272a", "#f97316"), "War": ("#292524", "#a8a29e"),
    "Western": ("#78350f", "#d97706"),
}
DEFAULT_COLORS = ("#27272a", "#a1a1aa")


def get_api_key():
    """Return the TMDB key from the environment, or None."""
    key = os.environ.get("TMDB_API_KEY", "").strip()
    return key or None


def genre_colors(genres_text):
    """Colours of the first genre of a movie ('Action|Comedy' -> Action colours)."""
    first = str(genres_text).split("|")[0]
    return GENRE_COLORS.get(first, DEFAULT_COLORS)


def fetch_tmdb_poster(tmdb_id, api_key, timeout=5):
    """Return a poster URL for a TMDB movie id, or None if it cannot be fetched."""
    if not api_key or tmdb_id is None:
        return None
    try:
        resp = requests.get(
            f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}",
            params={"api_key": api_key}, timeout=timeout,
        )
        if resp.status_code != 200:
            return None
        path = resp.json().get("poster_path")
        return f"{TMDB_IMAGE_BASE}{path}" if path else None
    except (requests.RequestException, ValueError, TypeError):
        return None

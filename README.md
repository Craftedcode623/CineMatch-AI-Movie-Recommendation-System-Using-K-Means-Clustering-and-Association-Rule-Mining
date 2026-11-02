# CineMatch-AI-Movie-Recommendation-System-Using-K-Means-Clustering-and-Association-Rule-Mining
Movie recommender using K-Means and Apriori on MovieLens data (Python, Streamlit).
---

## 1. Introduction
CineMatch AI is a web app that recommends movies. It applies two data mining techniques to real movie ratings:

- **K-Means clustering** groups similar movies (by rating, popularity, release year and genres).
- **Apriori association rules** find movies that many users like *together*.

## 2. Problem statement
Thousands of movies are available, and users find it hard to choose. This project uses data mining to discover
(a) which movies are similar to each other and (b) which movies are liked together, and turns that into recommendations.

## 3. Objectives
1. Load, clean and prepare the MovieLens data.
2. Group movies using K-Means (with Elbow Method and silhouette score).
3. Discover movie associations using Apriori (support, confidence, lift).
4. Build a recommendation engine from these results.
5. Present everything in an interactive dashboard.

## 4. Features
| Page | What it does |
|---|---|
| Home | Totals (movies, ratings, genres, average rating), popular / top-rated / newest movies, highlights |
| Explore Movies | Title search, genre, rating, year filters, movie cards, details and user tags |
| Recommendations | Similar movies (K-Means), liked-together (Apriori), by genre, personalized, top rated |
| K-Means Analysis | Elbow graph, silhouette graph, choose k, cluster distribution, 2-D cluster plot, cluster-wise stats and movies |
| Apriori Analysis | Adjustable settings, frequent itemsets, rules table, support/confidence/lift charts, explore one movie |
| Data Insights | Genre and rating distribution, top-rated, most-rated, ratings per movie, cluster and rule insights |
| About Project | Description, objectives, technologies, algorithms, limitations, future scope, diagrams |

## 5. Technologies
Python · Pandas · NumPy · Scikit-learn · mlxtend · Plotly · Streamlit · Requests (optional posters) · Matplotlib (diagrams only)

## 6. Algorithms (simple explanation)

### K-Means
1. Each movie becomes a row of numbers: average rating, log(number of ratings), release year, and 19 genre columns (1 if the movie has the genre, else 0).
2. **StandardScaler** puts all numbers on the same scale (otherwise "year ≈ 2000" would swamp the 0/1 genre columns).
3. K-Means puts the movies into *k* groups so that movies in a group are close to each other.
4. **Elbow Method**: run K-Means for k = 2…10 and plot **inertia** (total squared distance to cluster centres). Look for where the curve bends.
5. **Silhouette score** (−1 to 1) tells how well separated the clusters are.
6. *Similar movies* = movies from the same cluster, ranked by distance to the selected movie.

Only movies with **at least 10 ratings** are clustered (an average from 1–2 ratings is unreliable). That leaves **2,269 movies**.

### Apriori
- Each **user** is a "shopping basket"; the items are movies the user **liked**.
- **Rule for "liked": a rating of 4.0 or higher** (adjustable in the app).
- Only movies with at least **50 ratings** (450 movies) are used, otherwise almost every itemset is rare.
- **Support** = share of users whose basket contains all the movies of the rule.
- **Confidence** = among users who liked the "if" movies, the share who also liked the "then" movie.
- **Lift** = confidence ÷ how popular the "then" movie is anyway. Lift > 1 means "liked together more than by chance".
- **These are real transactions from the ratings. Nothing is simulated.**

### Weighted rating (used for ranking by genre / personalized)
`weighted = v/(v+m) × R + m/(v+m) × C` (R = movie average, v = its number of ratings, C = global average, m = threshold).
A movie with 2 five-star ratings can no longer beat one with 300 ratings averaging 4.4.

## 7. Dataset
- **MovieLens Latest Small** (GroupLens Research, University of Minnesota): https://grouplens.org/datasets/movielens/
- 9,742 movies · 100,836 ratings · 610 users · 3,683 tags · ratings from 1996 to 2018 · 19 genres (34 movies have no genre).
- Files in `data/`: `movies.csv`, `ratings.csv`, `tags.csv` (optional), `links.csv` (optional, only needed for TMDB posters).
- The copy in this project was fetched from a public GitHub mirror because the GroupLens server was not reachable while building. It has the same shape and content as `ml-latest-small`
  (same row counts and columns). For an official copy, run `python download_data.py` or download the zip from the link above and copy the CSV files into `data/`.
- Citation: F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context.* ACM TiiS 5(4), Article 19.
- The dataset has **no "date added" column**, so "newest movies" on the Home page uses the release year from the movie title.

## 8. Project structure
```
CineMatch-AI/
├── app.py               Streamlit interface (7 pages). Only UI code lives here.
├── data_processing.py   load_data, clean_data, build_movie_table, prepare_movie_features, prepare_transactions
├── clustering.py        scale_features, find_optimal_clusters, suggest_k, perform_clustering, summarize_clusters, pca_projection
├── association_rules.py generate_frequent_itemsets, generate_association_rules, readable tables, rules_for_movie
├── recommendation.py    recommend_similar_movies, recommend_associated_movies, recommend_by_genre,
│                        recommend_personalized, get_top_rated_movies, search_movies, filter_movies
├── visualizations.py    All Plotly charts (dark theme)
├── posters.py           Placeholder posters; optional TMDB posters if an API key is set
├── test_project.py      12 backend tests (run: python test_project.py)
├── download_data.py     Downloads the dataset from GroupLens into data/
├── make_diagrams.py     Re-creates the two diagrams in assets/
├── data/                movies.csv, ratings.csv, tags.csv, links.csv
├── assets/              architecture.png, workflow.png
├── .streamlit/config.toml  Dark theme colours
├── requirements.txt     Python packages
├── README.md            This file
└── .gitignore
```

### Architecture
<img width="1541" height="861" alt="image" src="https://github.com/user-attachments/assets/c1e6ad4f-b6fd-46fb-b251-704759d70f76" />


### Workflow
![Workflow](assets/workflow.png)

## 9. Installation and execution (Windows, VS Code terminal)

**1. Check Python (3.10 or newer; 3.11–3.13 recommended)**
```powershell
python --version
```
If this fails, install Python from https://www.python.org/downloads/ and tick **"Add Python to PATH"**.

**2. Put the project in a folder and open it in VS Code**
```powershell
cd C:\Users\<YourName>\Documents\CineMatch-AI
code .
```
Open the terminal in VS Code (Terminal → New Terminal).

**3. Create a virtual environment**
```powershell
python -m venv venv
```

**4. Activate it**
```powershell
venv\Scripts\activate
```
If PowerShell blocks this ("running scripts is disabled"), run once:
`Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`, then activate again. You should see `(venv)` at the start of the line.

**5. Install packages**
```powershell
pip install -r requirements.txt
```

**6. Dataset**
`data/movies.csv` and `data/ratings.csv` must exist. They are included. If missing: `python download_data.py`.

**7. (Optional) test the backend**
```powershell
python test_project.py
```

**8. Run the app**
```powershell
streamlit run app.py
```
The browser opens at http://localhost:8501. The first load takes a few seconds while the data is processed (results are then cached).

### Optional: real posters with TMDB
Posters are not part of MovieLens. By default the app shows coloured genre placeholders. To use real posters, create a free API key at https://www.themoviedb.org/settings/api and run:
```powershell
$env:TMDB_API_KEY = "your_key_here"
streamlit run app.py
```
> This code path has **not been tested** (no API key was available while building). The app works fully without it.

## 10. What was actually tested
| Test | Result |
|---|---|
| `python test_project.py` (12 tests: loading, missing-file error, cleaning, K-Means, Apriori values re-calculated by hand, transactions rule, search, genre filter, top-rated threshold, similar / association / personalized recommendations) | 12/12 passed |
| Same tests in a clean virtual environment with pandas 2.3.3, mlxtend 0.23.4 | 12/12 passed |
| Streamlit `AppTest` (headless): all 7 pages load, k and support sliders, no-result search | no exceptions |
| Environment used | Linux, Python 3.13, pandas 3.0.5 / 2.3.3, scikit-learn 1.9.1, plotly 7.1.0, streamlit 1.66.1 |

**Not tested:** the app was not opened in a real browser, so the visual look (card layout, colours) has not been checked by eye. It was also not run on Windows. TMDB posters were not tested.
Please open every page once and take your screenshots.

### Real results with the default settings
- K-Means: suggested **k = 6**, silhouette **0.170**, inertia ≈ 35,393. Clusters include Action/Adventure/Thriller (551 movies), Drama/Comedy/Romance (1,165), Children/Animation/Adventure (209), Mystery/Thriller/Drama (155), Horror/Thriller/Sci-Fi (169) and a small Documentary cluster (20).
- Example: movies similar to *Toy Story (1995)* → Monsters, Inc., Toy Story 2, Finding Nemo, Shrek …
- Apriori (support ≥ 0.12, confidence ≥ 0.6, lift ≥ 1): **598 users × 450 movies → 389 frequent itemsets → 319 rules**, lift between 1.32 and 4.16. The strongest rules are the *Lord of the Rings* trilogy movies, followed by *Star Wars* — plausible.

## 11. Screenshots checklist
- [ ] Home page (metrics + movie cards)
- [ ] Explore Movies with a search and filters
- [ ] Movie details
- [ ] Recommendations → Similar movies
- [ ] Recommendations → Liked together (Apriori)
- [ ] K-Means: Elbow and silhouette charts
- [ ] K-Means: cluster scatter plot and cluster statistics
- [ ] Apriori: frequent itemsets chart
- [ ] Apriori: rules table and lift chart
- [ ] Data Insights page
- [ ] About page with diagrams
- [ ] Terminal showing `python test_project.py` output

## 12. Limitations
- Small dataset (610 users) that ends in 2018; recent movies are missing.
- K-Means silhouette scores are low (0.13–0.22 for k = 2…10) and the elbow curve has no sharp bend: movie groups overlap, so the chosen k is a practical choice, not a perfect one.
- Apriori covers only popular movies (≥ 50 ratings) and "liked" ratings (≥ 4.0). Rare movies get no association recommendations; for example *Toy Story* has no rule with default settings.
- Association ≠ causation: rules say movies are liked together, not why.
- Recommendations are not personalised per account (no login) and were not evaluated with accuracy metrics.
- Posters need an optional external API key.

## 13. Future scope
Collaborative filtering (user–user / item–item), using tags as content features, the larger MovieLens datasets, user accounts and saved watch-lists, evaluating recommendations (precision@k), real posters and trailers through TMDB.

## 14. How to explain this project to your teacher (simple English)
> "My project recommends movies using two data mining methods on real MovieLens ratings.
> **K-Means** is a clustering method. I describe every movie with numbers (rating, popularity, year, genres) and K-Means puts similar movies into the same group. If you pick a movie, I recommend other movies from its group.
> **Apriori** is association rule mining, like 'people who buy bread also buy butter'. Every user is a basket, and the movies they rated 4 or more are the items. A rule like *'users who liked A also liked B'* is judged with three numbers: support (how common), confidence (how often it is true) and lift (is it more than chance).
> I cleaned the data with Pandas, built the models with Scikit-learn and mlxtend, and showed results in a Streamlit app with Plotly charts."

**Likely questions**
- *Why scale the features?* So large numbers like year do not dominate small 0/1 genre columns.
- *How did you choose k?* Elbow Method and silhouette score. The curve is smooth, so I chose the smallest k with a near-best silhouette (k = 6) and let the user change it.
- *Why is the silhouette score low?* Genre columns are binary and movies overlap in many genres, so clear separation is hard. I report it honestly.
- *What is lift?* Confidence divided by the popularity of the recommended movie. Above 1 means a real association.
- *Are the transactions real?* Yes: users' ratings ≥ 4.0. Nothing is simulated.
- *Why only movies with 50+ ratings in Apriori?* Rare movies have too little data for reliable support.

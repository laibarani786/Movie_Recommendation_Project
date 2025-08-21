# app.py
# 🎬 Movie Recommendation System (TMDB 5000)
# Uses content-based filtering on overview + genres + keywords + cast + director

import ast
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

# ---------------------------
# Helpers
# ---------------------------
def safe_load_list(cell):
    """TMDB columns (genres/keywords/cast/crew) are JSON-like strings. Parse safely to list."""
    if pd.isna(cell) or cell == "" or cell == "[]":
        return []
    try:
        # Handle strings like: '[{"id": 28, "name": "Action"}, ...]'
        data = ast.literal_eval(cell)
        if isinstance(data, list):
            return data
        return []
    except Exception:
        return []

def names_from_list(obj_list, key="name", top=None):
    """Extract 'name' from list[dict]. Optionally take only top N."""
    if not isinstance(obj_list, list):
        return []
    items = obj_list[: top if top else len(obj_list)]
    return [str(x.get(key, "")).strip().replace(" ", "") for x in items if isinstance(x, dict)]

def get_director(crew_list):
    """Return director name from crew list."""
    if not isinstance(crew_list, list):
        return ""
    for c in crew_list:
        if isinstance(c, dict) and c.get("job") == "Director":
            return str(c.get("name", "")).strip().replace(" ", "")
    return ""

def clean_text(s):
    """Lowercase + remove extra spaces for overview/text fields."""
    if pd.isna(s):
        return ""
    return " ".join(str(s).lower().split())

def get_year(date_str):
    try:
        if pd.isna(date_str) or str(date_str).strip() == "":
            return None
        return int(str(date_str)[:4])
    except Exception:
        return None

# ---------------------------
# Data Load + Process (cached)
# ---------------------------
@st.cache_data(show_spinner=True)
def load_and_prepare():
    movies_raw = pd.read_csv("tmdb_5000_movies.csv")
    credits_raw = pd.read_csv("tmdb_5000_credits.csv")

    # Ensure correct dtypes
    for col in ["genres", "keywords", "overview"]:
        if col not in movies_raw.columns:
            movies_raw[col] = ""

    # Parse JSON-like columns
    movies_raw["genres_list"] = movies_raw["genres"].apply(safe_load_list).apply(lambda x: names_from_list(x, "name"))
    movies_raw["keywords_list"] = movies_raw["keywords"].apply(safe_load_list).apply(lambda x: names_from_list(x, "name"))
    movies_raw["overview_clean"] = movies_raw["overview"].fillna("").apply(clean_text)

    # Credits: parse cast (top 5) and crew (director)
    credits_raw["cast_list"] = credits_raw["cast"].apply(safe_load_list).apply(lambda x: names_from_list(x, "name", top=5))
    credits_raw["crew_list"] = credits_raw["crew"].apply(safe_load_list)
    credits_raw["director"] = credits_raw["crew_list"].apply(get_director)

    # Merge on id / movie_id
    credits_small = credits_raw[["movie_id", "cast_list", "director"]]
    df = movies_raw.merge(credits_small, left_on="id", right_on="movie_id", how="left")

    # Build combined tokens (no spaces in names to treat multiwords as single tokens)
    df["director"] = df["director"].fillna("").apply(lambda x: x.replace(" ", ""))
    df["cast_list"] = df["cast_list"].apply(lambda lst: [x.replace(" ", "") for x in (lst if isinstance(lst, list) else [])])
    df["genres_list"] = df["genres_list"].apply(lambda lst: [x.replace(" ", "") for x in (lst if isinstance(lst, list) else [])])
    df["keywords_list"] = df["keywords_list"].apply(lambda lst: [x.replace(" ", "") for x in (lst if isinstance(lst, list) else [])])

    df["tokens"] = (
        df["overview_clean"].fillna("") + " "
        + df["genres_list"].apply(lambda x: " ".join(x)).fillna("") + " "
        + df["keywords_list"].apply(lambda x: " ".join(x)).fillna("") + " "
        + df["cast_list"].apply(lambda x: " ".join(x)).fillna("") + " "
        + df["director"].fillna("")
    ).str.strip()

    # Additional helpful columns
    df["year"] = df["release_date"].apply(get_year)
    df["title_display"] = df.apply(
        lambda r: f"{r['title']} ({r['year']})" if pd.notna(r["title"]) and r["year"] else str(r.get("title", "")),
        axis=1
    )

    # Filter empties
    df = df[~df["title"].isna()].reset_index(drop=True)

    return df

@st.cache_resource(show_spinner=True)
def build_vectorizer(tokens_series):
    cv = CountVectorizer(stop_words="english", max_features=20000)
    vectors = cv.fit_transform(tokens_series.values)
    sim = cosine_similarity(vectors)
    return cv, sim

# ---------------------------
# UI
# ---------------------------
st.title("🎬 Movie Recommendation System")
st.caption("Content-based recommendations using TMDB 5000 (overview + genres + keywords + cast + director)")
with st.expander("ℹ️ How it works", expanded=False):
    st.write(
        "- We combine overview, genres, keywords, top cast, and director into a single text.\n"
        "- We vectorize that text and compute **cosine similarity**.\n"
        "- Selecting any movie finds the most similar ones."
    )

# Load data
with st.spinner("Loading and preparing data..."):
    df = load_and_prepare()

# Build similarity (can take a moment on first run)
with st.spinner("Building similarity index..."):
    _, similarity = build_vectorizer(df["tokens"])

# Sidebar filters
st.sidebar.header("Filters")
min_votes = st.sidebar.slider("Minimum vote count", 0, int(df["vote_count"].max() if df["vote_count"].notna().any() else 5000), 100)
min_rating = st.sidebar.slider("Minimum rating (vote_average)", 0.0, 10.0, 6.5, 0.1)
year_from, year_to = st.sidebar.select_slider(
    "Year range",
    options=sorted([y for y in df["year"].dropna().unique()]),
    value=(min([y for y in df["year"].dropna().unique()]), max([y for y in df["year"].dropna().unique()])) if df["year"].notna().any() else (1990, 2017)
)

# Movie picker
movie_titles = df["title_display"].fillna(df["title"]).tolist()
selected = st.selectbox("Pick a movie to get recommendations:", movie_titles)

top_k = st.slider("How many recommendations?", 5, 20, 10)

def recommend(selected_title_display, k=10):
    # Map back to row index
    # Prefer exact match on title_display; fallback to title match
    idx_series = df.index[df["title_display"] == selected_title_display]
    if len(idx_series) == 0:
        # fallback by removing year suffix
        base_title = selected_title_display.split(" (")[0].strip()
        idx_series = df.index[df["title"].astype(str).str.lower() == base_title.lower()]
    if len(idx_series) == 0:
        return pd.DataFrame(columns=["title", "year", "vote_average", "vote_count", "popularity"])
    idx = idx_series[0]

    distances = list(enumerate(similarity[idx]))
    distances = sorted(distances, key=lambda x: x[1], reverse=True)

    # Exclude the same movie at index 0
    rec_indices = [i for i, _ in distances[1: 1 + max(k * 5, k + 5)]]  # take a larger pool before filtering
    rec_df = df.iloc[rec_indices][["title", "year", "vote_average", "vote_count", "popularity"]].copy()

    # Apply sidebar filters
    rec_df = rec_df[
        (rec_df["vote_count"].fillna(0) >= min_votes) &
        (rec_df["vote_average"].fillna(0) >= min_rating) &
        (rec_df["year"].fillna(0).between(year_from, year_to))
    ]

    # Take top-k after filters, keep the original order by similarity
    rec_df = rec_df.head(k)
    return rec_df

if st.button("🎯 Recommend"):
    results = recommend(selected, top_k)
    if results.empty:
        st.warning("No matching recommendations with current filters. Try lowering minimum votes/rating or widening year range.")
    else:
        st.subheader("Recommended Movies")
        for i, row in results.reset_index(drop=True).iterrows():
            col1, col2, col3, col4 = st.columns([4, 1.5, 1.5, 2])
            with col1:
                st.markdown(f"**{row['title']}**" + (f" ({int(row['year'])})" if pd.notna(row['year']) else ""))
            with col2:
                st.metric("⭐ Rating", f"{row['vote_average']:.1f}" if pd.notna(row['vote_average']) else "N/A")
            with col3:
                st.metric("🗳 Votes", int(row['vote_count']) if pd.notna(row['vote_count']) else 0)
            with col4:
                st.metric("🔥 Popularity", f"{row['popularity']:.1f}" if pd.notna(row['popularity']) else "N/A")
        st.caption("Tip: Adjust filters in the left sidebar to refine results.")

st.divider()
st.markdown(
    "Built with **Streamlit** + **scikit-learn**. Data: TMDB 5000 Movies & Credits (Kaggle)."
)

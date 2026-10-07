"""
data_loader.py
----------------
All data loading and metric-calculation functions for the
Movie Ratings Analysis Dashboard.
"""

import pandas as pd


# ------------------------------------------------------------
# Phase 2 — full loader from raw IMDb TSVs (used only in
# explore.ipynb / when regenerating the cleaned CSVs; the
# deployed app uses load_cleaned_data() instead, below)
# ------------------------------------------------------------
def load_movie_data(min_votes=1000):
    basics = pd.read_csv(
        "data/title.basics.tsv.gz", sep="\t", na_values="\\N", low_memory=False
    )
    ratings = pd.read_csv("data/title.ratings.tsv.gz", sep="\t", na_values="\\N")
    crew = pd.read_csv("data/title.crew.tsv.gz", sep="\t", na_values="\\N")

    movies = basics[basics["titleType"] == "movie"].copy()
    df = movies.merge(ratings, on="tconst", how="inner")
    df = df.merge(crew[["tconst", "directors"]], on="tconst", how="left")
    df = df.dropna(subset=["startYear", "genres", "runtimeMinutes"])
    df["startYear"] = df["startYear"].astype(int)
    df["runtimeMinutes"] = df["runtimeMinutes"].astype(int)
    df = df[df["numVotes"] >= min_votes]
    df["decade"] = (df["startYear"] // 10) * 10

    df_exploded = df.copy()
    df_exploded["genres"] = df_exploded["genres"].str.split(",")
    df_exploded = df_exploded.explode("genres")

    return df, df_exploded


# ------------------------------------------------------------
# Phase 7 — lightweight loader used by the deployed app.
# Reads the small, pre-cleaned CSVs committed to the repo
# instead of the large raw IMDb files.
# ------------------------------------------------------------
def load_cleaned_data():
    df = pd.read_csv("data/movies_cleaned.csv")
    df_exploded = pd.read_csv("data/movies_exploded.csv")
    return df, df_exploded


# ------------------------------------------------------------
# Phase 4 — core metrics
# ------------------------------------------------------------
def genre_avg_rating(df_exploded):
    return df_exploded.groupby("genres")["averageRating"].mean().sort_values(ascending=False)


def decade_avg_rating(df):
    return df.groupby("decade")["averageRating"].mean().sort_index()


def genre_decade_matrix(df_exploded, min_count=5):
    matrix = df_exploded.pivot_table(
        index="genres", columns="decade", values="averageRating", aggfunc="mean"
    )
    counts = df_exploded.pivot_table(
        index="genres", columns="decade", values="averageRating", aggfunc="count"
    )
    return matrix.where(counts >= min_count)


def director_consistency(df, min_films=5):
    stats = df.groupby("directors")["averageRating"].agg(["mean", "std", "count"])
    return stats[stats["count"] >= min_films]


def top_movies(df, min_votes=5000, n=20):
    filtered = df[df["numVotes"] >= min_votes].sort_values("averageRating", ascending=False)
    return filtered[
        ["primaryTitle", "startYear", "averageRating", "numVotes", "runtimeMinutes"]
    ].head(n)


# ------------------------------------------------------------
# Additional feature — yearly release trend
# ------------------------------------------------------------
def yearly_movie_count(df):
    return df.groupby("startYear").size()

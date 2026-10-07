"""
app.py
--------
Movie Ratings Analysis Dashboard — Streamlit app.

Run with:
    streamlit run app.py

Expects data/movies_cleaned.csv and data/movies_exploded.csv to already
exist (produced once from the raw IMDb files via data_loader.load_movie_data()).
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from data_loader import (
    load_cleaned_data,
    genre_decade_matrix,
    director_consistency,
    top_movies,
    yearly_movie_count,
)

sns.set_style("darkgrid")

# ------------------------------------------------------------
# Page setup
# ------------------------------------------------------------
st.set_page_config(page_title="Movie Ratings Dashboard", layout="wide")
st.title("🎬 Movie Ratings Analysis Dashboard")


# ------------------------------------------------------------
# Load data (cached)
# ------------------------------------------------------------
@st.cache_data
def get_data():
    return load_cleaned_data()


df, df_exploded = get_data()


# ------------------------------------------------------------
# Sidebar filters
# ------------------------------------------------------------
st.sidebar.header("Filters")

min_decade = int(df["decade"].min())
max_decade = int(df["decade"].max())
decade_range = st.sidebar.slider(
    "Decade range",
    min_value=min_decade,
    max_value=max_decade,
    value=(min_decade, max_decade),
    step=10,
)

all_genres = sorted(df_exploded["genres"].unique())
selected_genres = st.sidebar.multiselect(
    "Genres", options=all_genres, default=all_genres
)

min_votes_filter = st.sidebar.slider(
    "Minimum number of votes",
    min_value=1000,
    max_value=50000,
    value=1000,
    step=1000,
)

filtered_df = df[
    (df["decade"] >= decade_range[0])
    & (df["decade"] <= decade_range[1])
    & (df["numVotes"] >= min_votes_filter)
]

filtered_exploded = df_exploded[
    (df_exploded["decade"] >= decade_range[0])
    & (df_exploded["decade"] <= decade_range[1])
    & (df_exploded["numVotes"] >= min_votes_filter)
    & (df_exploded["genres"].isin(selected_genres))
]

if filtered_df.empty:
    st.sidebar.warning("No movies match the current filters. Try widening your selection.")
    st.warning("No movies match the current filters. Try widening your selection.")
    st.stop()

# --- Additional feature: search bar ---
st.sidebar.markdown("---")
search_query = st.sidebar.text_input("Search for a movie by title")
if search_query:
    search_result = filtered_df[
        filtered_df["primaryTitle"].str.contains(search_query, case=False, na=False)
    ]
    st.subheader(f"Search results for '{search_query}'")
    if search_result.empty:
        st.info("No matching titles in the current filtered set.")
    else:
        st.dataframe(
            search_result[
                ["primaryTitle", "startYear", "averageRating", "numVotes", "runtimeMinutes"]
            ],
            use_container_width=True,
        )
    st.markdown("---")

# --- Additional feature: row count + download + surprise me ---
st.sidebar.markdown(f"**{len(filtered_df):,} movies match current filters**")

csv_data = filtered_df.to_csv(index=False).encode("utf-8")
st.sidebar.download_button(
    label="Download filtered data as CSV",
    data=csv_data,
    file_name="filtered_movies.csv",
    mime="text/csv",
)

if st.sidebar.button("Surprise Me"):
    random_movie = filtered_df.sample(1).iloc[0]
    st.sidebar.success(
        f"**{random_movie['primaryTitle']}** ({int(random_movie['startYear'])})\n\n"
        f"Rating: {random_movie['averageRating']} ({int(random_movie['numVotes'])} votes)\n\n"
        f"Runtime: {int(random_movie['runtimeMinutes'])} min"
    )


# ------------------------------------------------------------
# Tabs
# ------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    [
        "Genre Distribution",
        "Genre x Decade",
        "Runtime vs Rating",
        "Director Consistency",
        "Top Movies",
        "Compare Genres",
        "Yearly Trend",
    ]
)

# --- Tab 1: Rating Distribution by Genre ---
with tab1:
    st.header("Rating Distribution by Genre")
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.violinplot(data=filtered_exploded, x="genres", y="averageRating", ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_xlabel("Genre")
    ax.set_ylabel("Average Rating")
    st.pyplot(fig)

# --- Tab 2: Genre x Decade heatmap ---
with tab2:
    st.header("Average Rating by Genre and Decade")
    matrix = genre_decade_matrix(filtered_exploded)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(
        matrix, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax,
        cbar_kws={"label": "Avg Rating"},
    )
    ax.set_xlabel("Decade")
    ax.set_ylabel("Genre")
    st.pyplot(fig)

# --- Tab 3: Runtime vs Rating ---
with tab3:
    st.header("Runtime vs Rating")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.scatterplot(data=filtered_df, x="runtimeMinutes", y="averageRating", alpha=0.2, ax=ax)
    sns.regplot(
        data=filtered_df, x="runtimeMinutes", y="averageRating",
        scatter=False, color="red", ax=ax,
    )
    ax.set_xlabel("Runtime (minutes)")
    ax.set_ylabel("Average Rating")
    st.pyplot(fig)

    corr = filtered_df["runtimeMinutes"].corr(filtered_df["averageRating"])
    st.caption(f"Correlation coefficient: {corr:.3f}")

# --- Tab 4: Director Consistency ---
with tab4:
    st.header("Director Consistency")
    stats = director_consistency(filtered_df, min_films=5)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Most Consistent")
        most_consistent = stats.sort_values("std").head(10)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=most_consistent["std"], y=most_consistent.index, ax=ax, color="steelblue")
        ax.set_xlabel("Rating Std Dev")
        st.pyplot(fig)

    with col2:
        st.subheader("Most Polarizing")
        most_polarizing = stats.sort_values("std", ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(x=most_polarizing["std"], y=most_polarizing.index, ax=ax, color="indianred")
        ax.set_xlabel("Rating Std Dev")
        st.pyplot(fig)

    st.caption(
        "Director names show as IMDb IDs (e.g. nm0000229) since name resolution "
        "isn't wired in — see the optional Top Actors add-on to fix this."
    )

# --- Tab 5: Top Rated Movies ---
with tab5:
    st.header("Top Rated Movies")
    min_votes_top = st.slider("Minimum votes for this list", 1000, 50000, 5000, step=1000)
    top = top_movies(filtered_df, min_votes=min_votes_top, n=20)
    st.dataframe(top, use_container_width=True)

# --- Tab 6: Compare Genres ---
with tab6:
    st.header("Compare Two Genres")
    col1, col2 = st.columns(2)
    with col1:
        genre_a = st.selectbox("Genre A", all_genres, index=0, key="genre_a")
    with col2:
        default_b_index = 1 if len(all_genres) > 1 else 0
        genre_b = st.selectbox("Genre B", all_genres, index=default_b_index, key="genre_b")

    compare_df = filtered_exploded[filtered_exploded["genres"].isin([genre_a, genre_b])]

    if compare_df.empty:
        st.info("No data for the selected genres under the current filters.")
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.violinplot(data=compare_df, x="genres", y="averageRating", ax=ax)
        ax.set_xlabel("Genre")
        ax.set_ylabel("Average Rating")
        st.pyplot(fig)

# --- Tab 7: Yearly Trend ---
with tab7:
    st.header("Number of Movies Released Per Year")
    yearly_counts = yearly_movie_count(filtered_df)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(yearly_counts.index, yearly_counts.values)
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Movies")
    st.pyplot(fig)

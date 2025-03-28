import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import re
from fuzzywuzzy import process


st.set_page_config(page_title="Movie Recommender", layout="wide")
st.sidebar.title("🎬 Movie Recommender System")

# Load datasets with caching for efficiency
@st.cache_data
def load_data():
    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")
    
    # Merge movies with average ratings per movie
    average_ratings = ratings.groupby("movieId")["rating"].mean().reset_index()
    movies = movies.merge(average_ratings, on="movieId", how="left")
    
    # Remove years from movie titles for better matching
    movies['clean_title'] = movies['title'].apply(lambda x: re.sub(r' \(\d{4}\)$', '', x))
    
    return movies.fillna("")

movies = load_data()

# TF-IDF Vectorization on genres
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(movies['genres'])

# Recommendation function
def recommend_movies(movie_title, num_recommendations=5):
    # Use fuzzy matching to find the best match
    best_match, score = process.extractOne(movie_title, movies['clean_title'])
    if score < 60:  # If confidence is too low
        return ["Movie not found! Try another one."]
    
    # Find index of the best matched movie
    idx = movies[movies['clean_title'] == best_match].index[0]
    
    # Compute cosine similarity for the selected movie only
    sim_scores = list(enumerate(linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations + 1]
    
    # Get recommended movie indices
    movie_indices = [i[0] for i in sim_scores]
    
    return movies['title'].iloc[movie_indices].tolist()

# Streamlit UI

st.sidebar.markdown("### Find movies similar to your favorite ones!")

# Sidebar input
movie_name = st.sidebar.text_input("Enter a movie name", placeholder="e.g. Toy Story")
if st.sidebar.button("Recommend 🎥"):
    recommendations = recommend_movies(movie_name)
    
    # Display recommendations in a modern UI
    st.markdown("## Recommended Movies")
    if recommendations[0] == "Movie not found! Try another one.":
        st.error(recommendations[0])
    else:
        cols = st.columns(2)
        for i, movie in enumerate(recommendations):
            with cols[i % 2]:
                st.success(f"🎞️ {movie}")

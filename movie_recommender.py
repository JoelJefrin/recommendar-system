import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import re
from fuzzywuzzy import process


def load_data():
    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")

    
    average_ratings = ratings.groupby("movieId")["rating"].mean().reset_index()
    movies = movies.merge(average_ratings, on="movieId", how="left")

    
    movies['clean_title'] = movies['title'].apply(lambda x: re.sub(r' \(\d{4}\)$', '', x))

    return movies.fillna("")


movies = load_data()


vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(movies['genres'])


def recommend_movies(movie_title, num_recommendations=5):
    
    best_match_tuple = process.extractOne(movie_title, movies['clean_title'].tolist())

    if best_match_tuple is None:
        return ["Movie not found! Try another one."]

    best_match, score = best_match_tuple

    if score < 60:
        return ["Movie not found! Try another one."]

    
    idx = movies[movies['clean_title'] == best_match].index[0]

    
    sim_scores = list(enumerate(linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations + 1]

    
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices].tolist()


def main():
    print("🎬 Welcome to the Movie Recommender System!")
    movie_name = input("Enter a movie name: ")

    recommendations = recommend_movies(movie_name)

    print("\n📽️ Recommended Movies:")
    if recommendations[0] == "Movie not found! Try another one.":
        print("❌ " + recommendations[0])
    else:
        for idx, rec in enumerate(recommendations, 1):
            print(f"{idx}. {rec}")

if __name__ == "__main__":
    main()

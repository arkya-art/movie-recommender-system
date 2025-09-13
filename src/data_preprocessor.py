import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

def process_movie_metadata(movies, ratings):
    movies['genres'] = movies['genres'].str.replace('|', ' ')
    movies['genres'] = movies['genres'].replace('(no genres listed)', 'no_genres')

    movies['release_year'] = movies['title'].str.extract(r'\((\d{4})\)').fillna(0).astype(int)

    movie_avg_ratings = ratings.groupby('movieId')['rating'].mean().reset_index()
    movie_avg_ratings.columns = ['movieId', 'avg_rating']

    movies = movies.merge(movie_avg_ratings, on='movieId', how='left')

    global_mean = ratings['rating'].mean()
    movies['avg_rating'] = movies['avg_rating'].fillna(global_mean)

    all_genres = []
    for genre_list in movies['genres'].str.split():
        all_genres.extend(genre_list)

    from collections import OrderedDict
    unique_genres = list(OrderedDict.fromkeys(all_genres))

    genre_matrix = np.zeros((len(movies), len(unique_genres)))

    for i, genre_list in enumerate(movies['genres'].str.split()):
        for genre in genre_list:
            if genre in unique_genres:
                j = unique_genres.index(genre)
                genre_matrix[i, j] = 1

    genre_df = pd.DataFrame(genre_matrix, columns=unique_genres)

    movie_features = pd.concat([
        movies[['movieId', 'release_year', 'avg_rating']].reset_index(drop=True),
        genre_df.reset_index(drop=True)
    ], axis=1)

    return movie_features, unique_genres

def process_user_metadata(ratings, movies, unique_genres):
    user_genre_ratings = ratings.merge(movies[['movieId', 'genres']], on='movieId')

    user_genre_ratings['genres'] = user_genre_ratings['genres'].str.split()
    user_genre_ratings = user_genre_ratings.explode('genres')

    user_avg_genre_ratings = user_genre_ratings.groupby(['userId', 'genres'])['rating'].mean().reset_index()

    user_genre_matrix = user_avg_genre_ratings.pivot(index='userId', columns='genres', values='rating').reset_index()

    user_avg_ratings = ratings.groupby('userId')['rating'].mean().to_dict()

    for genre in unique_genres:
        if genre not in user_genre_matrix.columns:
            user_genre_matrix[genre] = user_genre_matrix['userId'].map(user_avg_ratings).fillna(ratings['rating'].mean())

    user_genre_matrix['user_avg_rating'] = user_genre_matrix['userId'].map(user_avg_ratings)

    return user_genre_matrix

def main():
    
    
    print("Loading MovieLens data...")
    movies = pd.read_csv('/home/arkya/Documents/mtech_ai/subjects/ml_with_bigdata/movie-recommender/data/raw/movies.csv')
    ratings = pd.read_csv('/home/arkya/Documents/mtech_ai/subjects/ml_with_bigdata/movie-recommender/data/raw/ratings.csv')
    
   
    print("Processing movie metadata...")
    movie_features, unique_genres = process_movie_metadata(movies, ratings)
    movie_features.to_csv('/home/arkya/Documents/mtech_ai/subjects/ml_with_bigdata/movie-recommender/data/processed/movie_features.csv', index=False)
    
   
    print("Processing user metadata...")
    user_features = process_user_metadata(ratings, movies, unique_genres)
    user_features.to_csv('/home/arkya/Documents/mtech_ai/subjects/ml_with_bigdata/movie-recommender/data/processed/user_features.csv', index=False)
    
    print("Data preprocessing complete!")

if __name__ == "__main__":
    main()

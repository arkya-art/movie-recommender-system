from flask import Flask, jsonify, abort, request
from flask_cors import CORS
from kafka import KafkaProducer
import tensorflow as tf
import numpy as np
import pandas as pd
import joblib
import os
import json
import csv


producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

app = Flask(__name__)
CORS(app)


BASE_DIR         = '/home/arkya/Documents/mtech_ai/subjects/ml_with_bigdata/movie-recommender'
MODEL_PATH       = os.path.join(BASE_DIR, 'model', 'ncf_model.keras')
USER2IDX_PATH    = os.path.join(BASE_DIR, 'model', 'user2idx.pkl')
MOVIE2IDX_PATH   = os.path.join(BASE_DIR, 'model', 'movie2idx.pkl')
RAW_MOVIES_CSV   = os.path.join(BASE_DIR, 'data', 'raw', 'movies.csv')
RAW_RATINGS_CSV  = os.path.join(BASE_DIR, 'data', 'raw', 'ratings.csv')
LIKES_CSV_PATH   = os.path.join(BASE_DIR, 'data', 'raw', 'likes.csv')


if not os.path.isfile(LIKES_CSV_PATH):
    with open(LIKES_CSV_PATH, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['userId', 'movieId', 'timestamp'])


model       = tf.keras.models.load_model(MODEL_PATH)
user2idx    = joblib.load(USER2IDX_PATH)
movie2idx   = joblib.load(MOVIE2IDX_PATH)
movies_df   = pd.read_csv(RAW_MOVIES_CSV)
ratings_df  = pd.read_csv(RAW_RATINGS_CSV)


idx2movie = {idx: mid for mid, idx in movie2idx.items()}
n_movies  = len(idx2movie)


def popular_top_n(n=10):
    top = (
        ratings_df
        .groupby('movieId')['rating']
        .mean()
        .reset_index()
        .sort_values('rating', ascending=False)
        .head(n)
    )
    out = []
    for _, r in top.iterrows():
        mid   = int(r.movieId)
        score = float(r.rating)
        meta  = movies_df[movies_df['movieId'] == mid].iloc[0]
        out.append({
            'id':               mid,
            'title':            meta['title'],
            'genres':           meta['genres'],
            'predicted_rating': round(score, 2)
        })
    return out


@app.route('/predict/<user_id>', methods=['GET'])
def predict(user_id):
    
    try:
        if user_id.lower() != 'anonymous':
            uid = int(user_id)
            if uid in user2idx:
                user_idx = user2idx[uid]
            else:
                raise KeyError
        else:
            raise KeyError
    except:
        return jsonify(popular_top_n())

    
    movie_indices = np.arange(n_movies)
    movie_ids     = np.array([idx2movie[i] for i in movie_indices])
    user_indices  = np.full(n_movies, user_idx)
    preds = model.predict([user_indices, movie_indices], verbose=0).flatten()

    
    seen = set(ratings_df.loc[ratings_df['userId'] == uid, 'movieId'])

   
    try:
        likes_df = pd.read_csv(LIKES_CSV_PATH)
        new_likes = set(
            likes_df[likes_df['userId'] == uid]['movieId']
            .astype(int)
            .tolist()
        )
        print(f"[predict] user {uid} new_likes →", new_likes)
    except Exception:
        new_likes = set()
    seen |= new_likes

   
    mask = np.isin(movie_ids, list(seen), invert=True)
    movie_ids = movie_ids[mask]
    preds     = preds[mask]

  
    top_idx = np.argsort(preds)[::-1][:10]
    out     = []
    for i in top_idx:
        mid   = int(movie_ids[i])
        score = float(preds[i])
        meta  = movies_df[movies_df['movieId'] == mid].iloc[0]
        out.append({
            'id':               mid,
            'title':            meta['title'],
            'genres':           meta['genres'],
            'predicted_rating': round(score, 2)
        })

    return jsonify(out)

@app.route('/debug', methods=['GET'])
def debug():
    return jsonify({
        "status": "active",
        "endpoints": {
            "/predict/<user_id>": "Get top‑10 movie recs for a user (or use 'anonymous')",
            "/click":             "Record user clicks"
        },
        "example": "/predict/1 or /predict/anonymous"
    })

@app.route('/click', methods=['POST'])
def click():
    """
    Receives {"userId":..., "movieId":..., "timestamp":...},
    publishes to Kafka, and appends to likes.csv.
    """
    data = request.get_json()
    if not data or 'userId' not in data or 'movieId' not in data:
        return {"error": "userId & movieId required"}, 400

   
    producer.send('movie-clicks', data)

    
    with open(LIKES_CSV_PATH, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            data['userId'],
            data['movieId'],
            data.get('timestamp', '')
        ])

    return {"status": "received"}, 200


if __name__ == '__main__':
    print("Starting NCF Flask server on port 5002…")
    app.run(host='0.0.0.0', port=5002, debug=True)

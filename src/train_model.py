import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.layers import (
    Input, Embedding, Flatten, Concatenate,
    Dense, Dropout
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split


BASE_DIR       = '/home/arkya/Documents/mtech_ai/subjects/ml_with_bigdata/movie-recommender'
RAW_DIR        = os.path.join(BASE_DIR, 'data', 'raw')
MODEL_DIR      = os.path.join(BASE_DIR, 'model')
os.makedirs(MODEL_DIR, exist_ok=True)

RATINGS_CSV    = os.path.join(RAW_DIR, 'ratings.csv')
MOVIES_CSV     = os.path.join(RAW_DIR, 'movies.csv')
MODEL_OUT      = os.path.join(MODEL_DIR, 'ncf_model.keras')
USER2IDX_OUT   = os.path.join(MODEL_DIR, 'user2idx.pkl')
MOVIE2IDX_OUT  = os.path.join(MODEL_DIR, 'movie2idx.pkl')
HIST_OUT       = os.path.join(MODEL_DIR, 'ncf_training_history.png')


ratings = pd.read_csv(RATINGS_CSV)

unique_users  = ratings['userId'].unique()
unique_movies = ratings['movieId'].unique()
user2idx      = {uid:i for i,uid in enumerate(sorted(unique_users))}
movie2idx     = {mid:i for i,mid in enumerate(sorted(unique_movies))}

ratings['user_idx']  = ratings['userId'].map(user2idx)
ratings['movie_idx'] = ratings['movieId'].map(movie2idx)

X_user  = ratings['user_idx'].values
X_movie = ratings['movie_idx'].values
y       = ratings['rating'].values.astype(np.float32)


u_tr, u_te, m_tr, m_te, y_tr, y_te = train_test_split(
    X_user, X_movie, y,
    test_size=0.2,
    random_state=42
)


n_users  = len(user2idx)
n_movies = len(movie2idx)
emb_dim  = 50


user_in  = Input(shape=(1,), name='user_input')
movie_in = Input(shape=(1,), name='movie_input')


user_emb = Embedding(input_dim=n_users, output_dim=emb_dim,
                     embeddings_initializer='he_normal',
                     name='user_embedding')(user_in)
movie_emb = Embedding(input_dim=n_movies, output_dim=emb_dim,
                      embeddings_initializer='he_normal',
                      name='movie_embedding')(movie_in)


u_vec = Flatten()(user_emb)
m_vec = Flatten()(movie_emb)


x = Concatenate()([u_vec, m_vec])
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(32, activation='relu')(x)
x = Dropout(0.2)(x)


out = Dense(1, activation='linear', name='prediction')(x)

model = Model(inputs=[user_in, movie_in], outputs=out)
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mean_squared_error',
    metrics=['mean_absolute_error']
)

model.summary()


history = model.fit(
    x=[u_tr, m_tr],
    y=y_tr,
    batch_size=256,
    epochs=20,
    validation_split=0.1,
    verbose=1
)


model.save(MODEL_OUT)
joblib.dump(user2idx, USER2IDX_OUT)
joblib.dump(movie2idx, MOVIE2IDX_OUT)


plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('MSE'); plt.xlabel('epoch'); plt.legend()
plt.subplot(1,2,2)
plt.plot(history.history['mean_absolute_error'], label='train MAE')
plt.plot(history.history['val_mean_absolute_error'], label='val MAE')
plt.title('MAE'); plt.xlabel('epoch'); plt.legend()
plt.tight_layout()
plt.savefig(HIST_OUT)
plt.close()


test_loss, test_mae = model.evaluate([u_te, m_te], y_te, verbose=0)
print(f"Test MSE: {test_loss:.4f}")
print(f"Test MAE: {test_mae:.4f}")
print(f"Test RMSE: {np.sqrt(test_loss):.4f}")

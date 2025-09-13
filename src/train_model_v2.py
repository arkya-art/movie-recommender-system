import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os
import matplotlib.pyplot as plt

def prepare_training_data(ratings, movie_features, user_features):
    user_cols = [col for col in user_features.columns if col != 'userId']
    movie_cols = [col for col in movie_features.columns if col != 'movieId']

    merged_data = ratings.merge(movie_features, on='movieId')
    merged_data = merged_data.merge(user_features, on='userId', suffixes=('_movie', '_user'))

    movie_cols_merged = [col if col in ['release_year', 'avg_rating'] else col + '_movie'
                        for col in movie_cols]
    user_cols_merged = [col if col == 'user_avg_rating' else col + '_user'
                       for col in user_cols]

    X_movie = merged_data[movie_cols_merged].values
    X_user = merged_data[user_cols_merged].values
    y = merged_data['rating'].values

    user_scaler = StandardScaler()
    movie_scaler = StandardScaler()

    X_user = user_scaler.fit_transform(X_user)
    X_movie = movie_scaler.fit_transform(X_movie)

    X_user_train, X_user_test, X_movie_train, X_movie_test, y_train, y_test = train_test_split(
        X_user, X_movie, y, test_size=0.2, random_state=42
    )

    return (X_user_train, X_movie_train, y_train,
            X_user_test, X_movie_test, y_test,
            user_scaler, movie_scaler, user_cols_merged, movie_cols_merged)

def build_neural_cf_model(n_users, n_movies, n_user_features, n_movie_features, embedding_size=50):
    user_input = Input(shape=(n_user_features,), name='user_input')
    user_embedding = Dense(64, activation='relu',
                         kernel_initializer='he_normal',
                         kernel_regularizer=tf.keras.regularizers.l2(0.01))(user_input)
    user_embedding = BatchNormalization()(user_embedding)
    user_embedding = Dropout(0.3)(user_embedding)
    user_embedding = Dense(embedding_size, activation='relu', name='user_embedding',
                         kernel_initializer='he_normal',
                         kernel_regularizer=tf.keras.regularizers.l2(0.01))(user_embedding)
    user_embedding = BatchNormalization()(user_embedding)

    movie_input = Input(shape=(n_movie_features,), name='movie_input')
    movie_embedding = Dense(64, activation='relu',
                          kernel_initializer='he_normal',
                          kernel_regularizer=tf.keras.regularizers.l2(0.01))(movie_input)
    movie_embedding = BatchNormalization()(movie_embedding)
    movie_embedding = Dropout(0.3)(movie_embedding)
    movie_embedding = Dense(embedding_size, activation='relu', name='movie_embedding',
                          kernel_initializer='he_normal',
                          kernel_regularizer=tf.keras.regularizers.l2(0.01))(movie_embedding)
    movie_embedding = BatchNormalization()(movie_embedding)

    concatenated = Concatenate()([user_embedding, movie_embedding])

    x = Dense(32, activation='relu',
             kernel_initializer='he_normal',
             kernel_regularizer=tf.keras.regularizers.l2(0.01))(concatenated)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(16, activation='relu',
             kernel_initializer='he_normal',
             kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dense(1, activation='linear', name='prediction')(x)

    model = Model(inputs=[user_input, movie_input], outputs=x)
    model.compile(
        loss='mean_squared_error',
        optimizer=Adam(learning_rate=0.0005, clipnorm=1.0),
        metrics=['mean_absolute_error']
    )
    return model

def plot_training_history(history):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['mean_absolute_error'], label='Training MAE')
    plt.plot(history.history['val_mean_absolute_error'], label='Validation MAE')
    plt.title('Mean Absolute Error')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()

    plt.tight_layout()
    plt.savefig('/home/arkya/Documents/mtech_ai/subjects/ml_with_bigdata/movie-recommender/model/training_history.png')
    plt.close()

def main():
    
   
    print("Loading data...")
    ratings = pd.read_csv('/home/arkya/Documents/mtech_ai/subjects/ml_with_bigdata/movie-recommender/data/raw/ratings.csv')
    movie_features = pd.read_csv('/home/arkya/Documents/mtech_ai/subjects/ml_with_bigdata/movie-recommender/data/processed/movie_features.csv')
    user_features = pd.read_csv('/home/arkya/Documents/mtech_ai/subjects/ml_with_bigdata/movie-recommender/data/processed/user_features.csv')

   
    print("Preparing training data...")
    (X_user_train, X_movie_train, y_train, X_user_test, X_movie_test, y_test, user_scaler, movie_scaler, user_cols, movie_cols) = prepare_training_data(ratings, movie_features, user_features)

   
    X_user_train = np.nan_to_num(X_user_train)
    X_movie_train = np.nan_to_num(X_movie_train)
    y_train = np.nan_to_num(y_train)

  
    print("Building and training model...")
    n_users = len(user_features)
    n_movies = len(movie_features)
    n_user_features = X_user_train.shape[1]
    n_movie_features = X_movie_train.shape[1]

    model = build_neural_cf_model(n_users, n_movies, n_user_features, n_movie_features)
    
    history = model.fit(
        [X_user_train, X_movie_train],
        y_train,
        batch_size=64,
        epochs=20,
        validation_split=0.1,
        verbose=1
    )

    
    print("Saving model and scalers...")
    model.save('/home/arkya/Documents/mtech_ai/subjects/ml_with_bigdata/movie-recommender/model/neural_cf_model.keras')
    joblib.dump(user_scaler, '/home/arkya/Documents/mtech_ai/subjects/ml_with_bigdata/movie-recommender/model/user_scaler.pkl')
    joblib.dump(movie_scaler, '/home/arkya/Documents/mtech_ai/subjects/ml_with_bigdata/movie-recommender/model/movie_scaler.pkl')



    plot_training_history(history)

   
    print("Evaluating model...")
    X_user_test = np.nan_to_num(X_user_test)
    X_movie_test = np.nan_to_num(X_movie_test)
    y_test = np.nan_to_num(y_test)
    
    test_loss, test_mae = model.evaluate([X_user_test, X_movie_test], y_test, verbose=0)
    rmse = np.sqrt(test_loss)
    
    print(f"Test Loss (MSE): {test_loss:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

    print("Training complete! Model and artifacts saved in 'model' directory.")

if __name__ == "__main__":
    main()
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from category_encoders.count import CountEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from data_cleaning import data_for_content_based
from scipy.sparse import save_npz
from sklearn.compose import ColumnTransformer


CLEANED_DATA_PATH = 'data/cleaned_music.csv'
TRANSFORMER_PATH = 'models/column_transformer.joblib'
TRANSFORMED_DATA_PATH = 'data/transformed_music_data.npz'

frequency_encode_cols = ['year']
ohe_cols = ['artist', 'time_signature', 'key']
tfidf_cols = 'tags'
standard_scale_cols = ['duration_ms','loudness','tempo']
min_max_scale_cols = ['danceability', 'energy','speechiness', 'acousticness','instrumentalness', 'liveness', 'valence']

def train_transformer(data,transformer_path=TRANSFORMER_PATH):
    """
    Trains and saves a column transformer for preprocessing input data.

    This function constructs a scikit-learn ColumnTransformer that applies various preprocessing
    techniques to specified columns of the input DataFrame, including frequency encoding,
    one-hot encoding, TF-IDF vectorization, standard scaling, and min-max scaling. The transformer
    is then fitted to the provided data and saved to disk as a joblib file for later use.

    Args:
      data (pd.DataFrame): The input DataFrame containing features to be transformed.

    Saves:
      models/column_transformer.joblib: The fitted ColumnTransformer object.
    """
    # Define the column transformer with different preprocessing steps for different columns
    transformer = ColumnTransformer(transformers = [
    ("frequency_encode",CountEncoder(normalize=True,return_df=True),frequency_encode_cols),
    ("ohe",OneHotEncoder(handle_unknown='ignore'),ohe_cols),
    ("tfidf",TfidfVectorizer(max_features=85),tfidf_cols),
    ("standard_scale",StandardScaler(),standard_scale_cols),
    ("min_max_scale",MinMaxScaler(),min_max_scale_cols)
    ],remainder='passthrough',n_jobs=-1,force_int_remainder_cols=False)

    # Fit the transformer on the input data
    transformer.fit(data)

    # Save the fitted transformer to disk
    joblib.dump(transformer, transformer_path)

def transform_data(data, transformer):
    """
    Transforms the input data using a pre-trained column transformer.

    This function loads a column transformer from a specified joblib file and applies it to the provided data,
    returning the transformed result.

    Args:
      data (array-like or DataFrame): The input data to be transformed.
      transformer: Placeholder for the transformer object (not used, as the transformer is loaded within the function).

    Returns:
      array-like: The transformed data after applying the column transformer.
    """
    # Load the pre-trained column transformer from disk
    transformer = joblib.load('models/column_transformer.joblib')
    # Transform the input data using the loaded transformer
    transformed_data = transformer.transform(data)
    return transformed_data

def save_transformed_data(transformed_data, path):
    """
    Saves the transformed data to disk in a sparse matrix format.

    Args:
      transformed_data (sparse matrix): The transformed data to be saved.
      path (str): The path to save the transformed data to.
    """
    # Save the transformed data to disk in .npz format
    save_npz(path, transformed_data)

def calculate_similarity_scores(input_vector, data):
    """
    Calculates the cosine similarity scores between an input vector and a data matrix.

    Args:
      input_vector (array-like): The input vector to calculate similarity scores for.
      data (array-like): The data matrix to calculate similarity scores against.

    Returns:
      array-like: The cosine similarity scores between the input vector and the data matrix.
    """
    # Calculate the cosine similarity scores
    similarity_score = cosine_similarity(input_vector, data)
    return similarity_score

def recommend_songs(song_name, song_df, transformed_df, k=10):
  """
  Recommends k songs based on the similarity to a given song.

  Args:
    song_name: The name of the song to get recommendations for.
    song_df: The original dataframe containing song information.
    transformed_df: The transformed dataframe used for similarity calculation.
    k: The number of recommendations to return.

  Returns:
    A pandas DataFrame containing the top k recommended songs.
  """

  # Convert the song name to lowercase for case-insensitive matching
  song_name = song_name.lower()
  # Find the song in the dataframe
  song_input = song_df[song_df['name'] == song_name]
  # Get the index of the song
  song_index = song_input.index[0]

  # Check if the song is in the dataset
  if song_input.empty:
    print(f"Song '{song_name}' not found in the dataset.")


  # Get the vector of the song
  input_vector = transformed_df[song_index].reshape(1, -1)
  # Calculate the similarity scores between the song and all other songs
  similarity_scores = calculate_similarity_scores(input_vector,transformed_df)


  # Get the indices of the top k most similar songs
  top_k_indices = similarity_scores.argsort()[0][-(k+1):][::-1]


  # Exclude the input song itself from the recommendations
  recommended_indices = [i for i in top_k_indices if song_df.iloc[i]['name'] != song_name]

  # Return the top k recommended songs
  top_k_songs =  song_df.iloc[recommended_indices[:k]]
  # Select the name, artist, and spotify_preview_url columns
  top_songs =  top_k_songs[['name','artist','spotify_preview_url']].reset_index(drop=True)
  return top_songs

def main():
    """
    Main function to train the transformer, transform the data, and recommend songs.
    """
    # Load the cleaned music data
    data = pd.read_csv(CLEANED_DATA_PATH)
    # Prepare the data for content-based filtering
    content_based_data = data_for_content_based(data)
    # Train the column transformer
    train_transformer(content_based_data, TRANSFORMER_PATH)
    # Load the trained transformer
    transformer = joblib.load(TRANSFORMER_PATH)
    # Transform the data using the trained transformer
    transformed_data = transform_data(content_based_data, transformer)
    # Save the transformed data
    save_transformed_data(transformed_data, TRANSFORMED_DATA_PATH)
    # Recommend songs based on the song "Hips Don't Lie"
    song_list = recommend_songs("Hips Don't Lie", data, transformed_data, k=10)
    return song_list

if __name__ == "__main__":
    songs = main()
    print(songs)
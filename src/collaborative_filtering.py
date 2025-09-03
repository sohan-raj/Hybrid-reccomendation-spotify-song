import pandas as pd
import dask.dataframe as dd
from scipy.sparse import save_npz,load_npz
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

# File paths for saving intermediate and final data
track_id_save_path = 'data/track_ids.npy'
filtered_data_save_path = 'data/collab_filtered_data.csv'
interaction_matrix_save_path = 'data/interaction_matrix.npz'

song_data_path = 'data/cleaned_music.csv'
user_data_path = 'data/user.csv'

def filter_songs_data(song_data : pd.DataFrame, track_ids: list, save_df_path : str) -> pd.DataFrame:
    """
    Filter the song data to only include songs with track IDs present in the user data.

    Args:
        song_data (pd.DataFrame): DataFrame containing song metadata.
        track_ids (list): List of track IDs to filter the song data.
        save_df_path (str): Path to save the filtered DataFrame as CSV.

    Returns:
        pd.DataFrame: Filtered DataFrame containing only the songs with track IDs present in the user data.
    """
    filtered_data = song_data[song_data['track_id'].isin(track_ids)]
    filtered_data.reset_index(drop=True, inplace=True)
    save_pandas_df_to_csv(filtered_data, save_df_path)
    return filtered_data

def save_pandas_df_to_csv(df: pd.DataFrame, path: str) -> None:
    """
    Save a Pandas DataFrame to a CSV file.

    Args:
        df (pd.DataFrame): DataFrame to save.
        path (str): Path to save the CSV file.
    """
    df.to_csv(path, index=False)

def save_sparse_matrix(matrix, path: str) -> None:
    """
    Save a sparse matrix to a file in .npz format.

    Args:
        matrix (sparse matrix): Sparse matrix to save.
        path (str): Path to save the sparse matrix.
    """
    save_npz(path, matrix)

def create_interaction_matrix(
    data: dd.DataFrame,
    filtered_song_data: pd.DataFrame,
    track_id_save_path: str,
    interaction_matrix_save_path: str
) -> None:
    """
    Create and save a user-item interaction matrix from user listening data,
    aligned with filtered song metadata.
    """
    # Filter only valid track_ids
    df = data[data["track_id"].isin(filtered_song_data["track_id"].unique())]

    # Convert to Pandas for categorical operations
    df = df.compute()
    df["playcount"] = df["playcount"].astype(np.float64)

    # Convert columns to categorical
    df["user_id"] = df["user_id"].astype("category")
    df["track_id"] = df["track_id"].astype("category")

    # Encode user_id and track_id
    df["user_id_code"] = df["user_id"].cat.codes
    df["track_id_code"] = df["track_id"].cat.codes

    # Save aligned track_ids
    track_ids = df["track_id"].cat.categories.values
    np.save(track_id_save_path, track_ids, allow_pickle=True)

    # Group by user and track, sum playcounts
    grouped = df.groupby(["user_id_code", "track_id_code"])["playcount"].sum().reset_index()

    # Build sparse matrix
    interaction_matrix_sparse = csr_matrix(
        (grouped["playcount"], (grouped["track_id_code"], grouped["user_id_code"])),
        shape=(len(df["track_id"].cat.categories), len(df["user_id"].cat.categories))
    )

    save_sparse_matrix(interaction_matrix_sparse, interaction_matrix_save_path)

# def create_interaction_matrix(data: dd.DataFrame, track_id_save_path: str,interaction_matrix_save_path: str) -> pd.DataFrame:
#     """
#     Create and save a user-item interaction matrix from user listening data.

#     Args:
#         data (dd.DataFrame): Dask DataFrame containing user interaction data.
#         track_id_save_path (str): Path to save the track IDs as .npy file.
#         interaction_matrix_save_path (str): Path to save the interaction matrix as .npz file.

#     Returns:
#         None
#     """
#     # Copy and preprocess the data
#     df = data.copy()
#     df['playcount'] = df['playcount'].astype(np.float64)

#     # Categorize user_id and track_id for efficient encoding
#     df = df.categorize(columns=['user_id', 'track_id'])

#     # Create mappings for user and track IDs
#     user_mapping = df['user_id'].cat.codes
#     track_mapping = df['track_id'].cat.codes

#     # Get unique track IDs and save them
#     track_ids = df['track_id'].cat.categories.values
#     np.save(track_id_save_path, track_ids, allow_pickle=True)

#     # Assign encoded user and track IDs
#     df = df.assign(user_id = user_mapping, track_id = track_mapping)

#     # Group by user and track, summing playcounts
#     interaction_matrix = df.groupby(['user_id', 'track_id'])['playcount'].sum().reset_index()
#     interaction_matrix = interaction_matrix.compute()

#     # Prepare indices and data for sparse matrix
#     row_indices = interaction_matrix['track_id']
#     col_indices = interaction_matrix['user_id']
#     data_values = interaction_matrix['playcount']

#     n_tracks = row_indices.nunique()
#     n_users = col_indices.nunique()

#     # Create and save the sparse interaction matrix
#     interaction_matrix_sparse = csr_matrix((data_values, (row_indices, col_indices)), shape=(n_tracks, n_users))
#     save_sparse_matrix(interaction_matrix_sparse, interaction_matrix_save_path)

def collaborative_recommendations(song_name: str, artist_name: str, song_df: pd.DataFrame, track_ids: np.ndarray, interaction_matrix: csr_matrix, k: int = 10) -> pd.DataFrame:
    """
    Generate collaborative filtering recommendations for a given song.

    Args:
        song_name (str): Name of the song to generate recommendations for.
        artist_name (str): Name of the artist of the song.
        song_df (pd.DataFrame): DataFrame containing song metadata.
        track_ids (np.ndarray): Array of track IDs.
        interaction_matrix (csr_matrix): User-item interaction matrix.
        k (int): Number of recommendations to return.

    Returns:
        pd.DataFrame: DataFrame containing the top k recommended songs.
    """
    # Lowercase for case-insensitive matching
    song_name = song_name.lower()
    artist_name = artist_name.lower()

    # Find the song row in the DataFrame
    song_row = song_df[(song_df['name'] == song_name)]
    if song_row.empty:
        print(f"Song '{song_name}' by '{artist_name}' not found in the dataset.")
    
    # Get the track ID and its index
    input_track_id = song_row['track_id'].values.item()
    input_index = np.where(track_ids == input_track_id)[0][0]

    # Get the interaction vector for the song (may be sparse or 1D)
    input_row = interaction_matrix[input_index]
    # Ensure input_row is 2D: (1, n_users)
    if hasattr(input_row, "toarray"):
        input_vector = input_row.toarray().reshape(1, -1)
    else:
        input_vector = np.asarray(input_row).reshape(1, -1)

    # Compute cosine similarity with all other songs; flatten to 1D
    similarity_scores = cosine_similarity(input_vector, interaction_matrix).ravel()

    # Get indices of top (k+1) similar songs to account for the input song itself,
    # then exclude the input song and take first k results.
    top_indices = np.argsort(similarity_scores)[-(k+1):][::-1]
    # remove the input index if present
    top_indices = [idx for idx in top_indices if idx != input_index]
    recommendation_indices = np.array(top_indices)[:k]
    # Get track IDs for recommendations
    recommendation_track_ids = track_ids[recommendation_indices]
    # Filter song DataFrame for recommended songs
    filtered_songs = song_df[song_df['track_id'].isin(recommendation_track_ids)]

    # Return relevant columns for recommendations
    return filtered_songs[['name', 'artist', 'spotify_preview_url']].reset_index(drop=True)

def main():
    """
    Main function to run the collaborative filtering pipeline:
    - Loads user and song data
    - Filters songs to those present in user data
    - Creates and saves the interaction matrix
    """
    # Load user listening data
    user_data = dd.read_csv(user_data_path)

    # Get unique track IDs from user data
    unique_track_ids = user_data['track_id'].unique().compute().tolist()
    # Load song metadata
    song_data = pd.read_csv(song_data_path)
    # Filter songs to those present in user data
    filtered_song_data = pd.read_csv(filtered_data_save_path)
    track_ids = np.load(track_id_save_path, allow_pickle=True)
    interaction_matrix = load_npz(interaction_matrix_save_path)
    recommendations = collaborative_recommendations("hips don't lie", 'shakira', filtered_song_data, track_ids, interaction_matrix, k=6)
    print(recommendations)

if __name__ == "__main__":
    main()




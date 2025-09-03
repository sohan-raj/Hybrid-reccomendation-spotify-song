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
    return interaction_matrix_sparse

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
    if song_row.empty:
        # song not found -- return empty dataframe with expected columns
        return pd.DataFrame(columns=['name', 'artist', 'spotify_preview_url'])

    input_track_id = song_row['track_id'].values.item()

    # Ensure track_ids is an array (handles 0-d numpy scalars or lists)
    track_ids_arr = np.atleast_1d(np.array(track_ids))
    matches = np.flatnonzero(track_ids_arr == input_track_id)
    if matches.size == 0:
        # track id not present in provided track_ids mapping
        print(f"Track id '{input_track_id}' not found in track_ids mapping.")
        return pd.DataFrame(columns=['name', 'artist', 'spotify_preview_url'])

    input_index = int(matches[0])

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
    # Ensure track_ids is a numpy array for advanced indexing (handles lists)
    track_ids_arr = np.atleast_1d(np.array(track_ids))
    # Get track IDs for recommendations
    recommendation_track_ids = track_ids_arr[np.asarray(recommendation_indices, dtype=int)]
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
    song_data = pd.read_csv(song_data_path)
    df = user_data[user_data["track_id"].isin(song_data["track_id"].unique())]

    # Convert to Pandas for categorical operations
    df = df.compute()
    # Get unique track IDs from user data
    unique_track_ids = user_data['track_id'].unique().compute().tolist()

    # Filter songs to those present in user data
    filtered_song_data = filter_songs_data(song_data,   unique_track_ids, filtered_data_save_path)
    # Create and save interaction matrix
    interaction_matrix_sparse = create_interaction_matrix(user_data, filtered_song_data, track_id_save_path, interaction_matrix_save_path)


    recommendations = collaborative_recommendations("hips don't lie", 'shakira', filtered_song_data, unique_track_ids, interaction_matrix_sparse, k=6)
    print(recommendations)

if __name__ == "__main__":
    main()




import pandas as pd
import dask.dataframe as dd
from scipy.sparse import save_npz,load_npz
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix


track_id_save_path = 'data/track_ids.npy'
filtered_data_save_path = 'data/collab_filtered_data.csv'
interaction_matrix_save_path = 'data/interaction_matrix.npz'

song_data_path = 'data/cleaned_music.csv'
user_data_path = 'data/user.csv'

def filter_songs_data(song_data : pd.DataFrame, track_ids: list, save_df_path : str) -> pd.DataFrame:
    filtered_data = song_data[song_data['track_id'].isin(track_ids)]
    filtered_data.reset_index(drop=True, inplace=True)
    save_pandas_df_to_csv(filtered_data, save_df_path)
    return filtered_data

def save_pandas_df_to_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)

def save_sparse_matrix(matrix, path: str) -> None:
    save_npz(path, matrix)

def create_interaction_matrix(data: dd.DataFrame, track_id_save_path: str,interaction_matrix_save_path: str) -> pd.DataFrame:

    df = data.copy()
    df['playcount'] = df['playcount'].astype(np.float64)

    df = df.categorize(columns=['user_id', 'track_id'])

    user_mapping = df['user_id'].cat.codes
    track_mapping = df['track_id'].cat.codes

    track_ids = df['track_id'].cat.categories.values

    np.save(track_id_save_path, track_ids, allow_pickle=True)

    df = df.assign(user_id = user_mapping, track_id = track_mapping)

    interaction_matrix = df.groupby(['user_id', 'track_id'])['playcount'].sum().reset_index()

    interaction_matrix = interaction_matrix.compute()

    row_indices = interaction_matrix['track_id']
    col_indices = interaction_matrix['user_id']
    data_values = interaction_matrix['playcount']

    n_tracks = row_indices.nunique()
    n_users = col_indices.nunique()

    interaction_matrix_sparse = csr_matrix((data_values, (row_indices, col_indices)), shape=(n_tracks, n_users))

    save_sparse_matrix(interaction_matrix_sparse, interaction_matrix_save_path)

def collaborative_recommendations(song_name: str, artist_name: str, song_df: pd.DataFrame, track_ids: np.ndarray, interaction_matrix: csr_matrix, k: int = 10) -> pd.DataFrame:
    song_name = song_name.lower()

    artist_name = artist_name.lower()

    song_row = song_df[(song_df['name'] == song_name)]
    if song_row.empty:
        print(f"Song '{song_name}' by '{artist_name}' not found in the dataset.")
    
    input_track_id = song_row['track_id'].values.item()
    input_index = np.where(track_ids == input_track_id)[0][0]

    input_array = interaction_matrix[input_index]
    similarity_scores = cosine_similarity(input_array, interaction_matrix)
    recommendation_indices = similarity_scores.ravel().argsort()[0][-(k-1):][::-1]
    recommendation_track_ids = track_ids[recommendation_indices]
    filtered_songs = song_df[song_df['track_id'].isin(recommendation_track_ids)]

    return filtered_songs[['name', 'artist', 'spotify_preview_url']].reset_index(drop=True)

def main():
    user_data = dd.read_csv(user_data_path)

    unique_track_ids = user_data['track_id'].unique().compute().tolist()
    song_data = pd.read_csv(song_data_path)
    filtered_song_data = filter_songs_data(song_data, unique_track_ids, filtered_data_save_path)
    create_interaction_matrix(user_data, track_id_save_path, interaction_matrix_save_path)
    # track_ids = np.load(track_id_save_path, allow_pickle=True)
    # interaction_matrix = load_npz(interaction_matrix_save_path)
    # recommendations = collaborative_recommendations("hips don't lie", 'sakira', filtered_song_data, track_ids, interaction_matrix, k=6)
    # print(recommendations)

if __name__ == "__main__":
    main()




import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import load_npz


class HybridRecommendation:
    """
    A class to generate hybrid music recommendations by combining content-based and collaborative filtering scores.

    This class takes a song name and artist, computes similarity scores using both content-based and collaborative
    filtering approaches, normalizes and combines these scores using specified weights, and returns the top N recommended songs.

    Args:
        song_name (str): The name of the input song.
        artist_name (str): The name of the input artist.
        number_of_recommendations (int): Number of recommendations to return.
        weight_collaborative (float): Weight for collaborative filtering score.
        weight_content_based (float): Weight for content-based filtering score.
        song_data (pd.DataFrame): DataFrame containing song metadata.
        track_ids (np.ndarray): Array of track IDs corresponding to the songs.
        interaction_matrix (np.ndarray): User-item interaction matrix for collaborative filtering.
        transformed_data (np.ndarray): Feature matrix for content-based filtering.
    """
    def __init__(self,song_name,artist_name,number_of_recommendations,weight_collaborative,weight_content_based,song_data, track_ids,interaction_matrix,transformed_data):
        self.song_name = song_name.lower()
        self.artist_name = artist_name.lower()
        self.number_of_recommendations = number_of_recommendations
        self.weight_collaborative = weight_collaborative
        self.weight_content_based = weight_content_based
        self.song_data = song_data
        self.track_ids = track_ids
        self.interaction_matrix = interaction_matrix
        self.transformed_data = transformed_data
        
        # All methods below are used internally for the recommendation process

    def calculate_content_based_scores(self,song_name,artist_name,song_data,transformed_data):
        """
        Calculate content-based similarity scores for the input song using cosine similarity.

        Returns:
            np.ndarray: Similarity scores for all songs.
        """
        song_row = song_data[(song_data['name'] == song_name) & (song_data['artist'] == artist_name)]
        if song_row.empty:
            print(f"Song '{song_name}' by '{artist_name}' not found in the dataset.")
            return None
        song_index = song_row.index[0]
        input_vector = transformed_data[song_index].reshape(1, -1)
        similarity_scores = cosine_similarity(input_vector, transformed_data)
        return similarity_scores.ravel()
    
    def calculate_collaborative_scores(self,song_name,artist_name,song_data, track_ids,interaction_matrix):
        """
        Calculate collaborative filtering similarity scores for the input song using cosine similarity.

        Returns:
            np.ndarray: Similarity scores for all songs.
        """
        song_row = song_data[(song_data['name'] == song_name) & (song_data['artist'] == artist_name)]
        if song_row.empty:
            print(f"Song '{song_name}' by '{artist_name}' not found in the dataset.")
            return None
          # Get the track ID and its index
        input_track_id = song_row['track_id'].values.item()
        input_index = np.where(track_ids == input_track_id)[0][0]

        # Get the interaction vector for the song
        input_array = interaction_matrix[input_index]
        # Compute cosine similarity with all other songs
        similarity_scores = cosine_similarity(input_array, interaction_matrix)
        return similarity_scores.ravel()
    
    def normalize_scores(self,scores):
        """
        Normalize the similarity scores to a 0-1 range.

        Returns:
            np.ndarray: Normalized scores.
        """
        min_score = np.min(scores)
        max_score = np.max(scores)
        # if max_score - min_score == 0:
        #     return np.zeros_like(scores)
        normalized_scores = (scores - min_score) / (max_score - min_score)
        return normalized_scores
    
    def weighted_combination(self,content_scores,collaborative_scores):
        """
        Combine content-based and collaborative scores using the specified weights.

        Returns:
            np.ndarray: Weighted combined scores.
        """
        combined_scores = (self.weight_content_based * content_scores) + (self.weight_collaborative * collaborative_scores)
        return combined_scores
    
    def get_recommendations(self):
        """
        Generate the top N hybrid recommendations for the input song and artist.

        Returns:
            pd.DataFrame: DataFrame of recommended songs.
        """
        # Calculate content-based similarity scores
        content_based_similarity_scores = self.calculate_content_based_scores(self.song_name,self.artist_name,self.song_data,self.transformed_data)

        # Calculate collaborative similarity scores
        collaborative_similarity_scores = self.calculate_collaborative_scores(self.song_name, self.artist_name, self.song_data, self.track_ids, self.interaction_matrix)

        # Normalize both sets of scores
        normalized_content_scores = self.normalize_scores(content_based_similarity_scores)
        normalized_collaborative_scores = self.normalize_scores(collaborative_similarity_scores)

        # Combine scores using the specified weights
        weighted_scores = self.weighted_combination(normalized_content_scores,normalized_collaborative_scores)

        # Get indices of top recommendations (excluding the input song itself)
        recommedation_indices = weighted_scores.argsort()[-(self.number_of_recommendations+1):-1][::-1]

       

        # Get track IDs for recommendations
        recommendation_track_ids = self.track_ids[recommedation_indices]

        # Get top scores
        top_score = np.sort(weighted_scores)[-self.number_of_recommendations:][::-1]


        # Create a DataFrame with track IDs and scores
        scores_df = pd.DataFrame({"track_id":recommendation_track_ids.tolist(),"score":top_score})

        # Merge with song data and return top recommendations
        tok_k_songs = (
                        self.song_data
                        .loc[self.song_data["track_id"].isin(recommendation_track_ids)]
                        .merge(scores_df, on="track_id")
                        .sort_values("score", ascending=False)
                        .drop(columns=['danceability', 'energy',
                                        'key', 'loudness', 'mode', 'speechiness', 'acousticness',
                                        'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature'])
                        .reset_index(drop=True)
        )

        return tok_k_songs
        

def main():
    """
    Example main function to run the HybridRecommendation class and print top recommendations.
    """
    song_name = "Immigrant Song"
    artist_name = "Led Zeppelin"
    number_of_recommendations = 10

    # Load required data
    filtered_data_path = 'data/collab_filtered_data.csv'
    filtered_data = pd.read_csv(filtered_data_path)
    transformed_hybrid_data_path = 'data/transformed_hybrid_data.npz'
    transformed_hybrid_data = load_npz(transformed_hybrid_data_path)
    track_ids_path = 'data/track_ids.npy'
    track_ids = np.load(track_ids_path, allow_pickle=True)
    interaction_matrix_path = 'data/interaction_matrix.npz'
    interaction_matrix = load_npz(interaction_matrix_path)

    # Instantiate and run the hybrid recommender
    recommender = HybridRecommendation(
        song_name=song_name,
        artist_name=artist_name,
        number_of_recommendations=number_of_recommendations,
        weight_collaborative=0.5,
        weight_content_based=0.5,
        song_data=filtered_data,
        track_ids=track_ids,
        interaction_matrix=interaction_matrix,
        transformed_data=transformed_hybrid_data
    )
    top_recommendations = recommender.get_recommendations()
    print(top_recommendations)

if __name__ == "__main__":
    main()




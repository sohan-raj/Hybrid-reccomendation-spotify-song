import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from scipy.sparse import load_npz

# ensure src directory is importable when running from project root
SRC_DIR = os.path.dirname(__file__)
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

try:
    from hybrid_recommendation import HybridRecommendation
except Exception:
    HybridRecommendation = None

from collaborative_filtering import collaborative_recommendations
from content_based_filtering import recommend_songs

# Paths to data files
CLEANED_DATA_PATH = 'data/cleaned_music.csv'
FILTERED_DATA_PATH = 'data/collab_filtered_data.csv'
TRANSFORMED_CONTENT_PATH = 'data/transformed_music_data.npz'
TRANSFORMED_HYBRID_PATH = 'data/transformed_hybrid_data.npz'
TRACK_IDS_PATH = 'data/track_ids.npy'
INTERACTION_MATRIX_PATH = 'data/interaction_matrix.npz'


@st.cache_data
def load_song_data(path):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


@st.cache_data
def load_transformed(path):
    if not os.path.exists(path):
        return None
    return load_npz(path)


@st.cache_data
def load_track_ids(path=TRACK_IDS_PATH):
    if not os.path.exists(path):
        return None
    return np.load(path, allow_pickle=True)


@st.cache_data
def load_interaction_matrix(path=INTERACTION_MATRIX_PATH):
    if not os.path.exists(path):
        return None
    return load_npz(path)


st.set_page_config(page_title="Music Recommendation", layout="wide")
st.title("Music Recommendation System 🎧")

# Sidebar inputs
st.sidebar.header("Input")
song_name = st.sidebar.text_input("Song Name", "Hips Don't Lie")
artist_name = st.sidebar.text_input("Artist Name", "Shakira")
number_of_recommendations = st.sidebar.slider("Number of Recommendations", 1, 20, 10)
recommendation_type = st.sidebar.selectbox("Recommendation Type", ("Hybrid", "Collaborative", "Content-Based"))

st.subheader("Selected Parameters")
st.write(f"**Song:** {song_name}")
st.write(f"**Artist:** {artist_name}")
st.write(f"**Count:** {number_of_recommendations}")
st.write(f"**Type:** {recommendation_type}")


def get_recommendations(song, artist, k, rec_type):
    """Call the appropriate recommender and return a DataFrame of results.

    This wrapper handles missing data and provides fallbacks where possible.
    """
    

    try:
        if rec_type == "Content-Based":
            song_df = load_song_data(CLEANED_DATA_PATH)
            if song_df is None:
                st.error(f"Could not find cleaned song data at {CLEANED_DATA_PATH}")
                return None
            transformed = load_transformed(TRANSFORMED_CONTENT_PATH)
            if transformed is None:
                st.error(f"Content transformed data not found at {TRANSFORMED_CONTENT_PATH}")
                return None
            return recommend_songs(song, song_df, transformed, k)

        if rec_type == "Collaborative":
            track_ids = load_track_ids()
            interaction = load_interaction_matrix()
            song_df = load_song_data(FILTERED_DATA_PATH)
            if track_ids is None or interaction is None:
                st.error("Collaborative data (track ids or interaction matrix) not found.")
                return None
            # convert sparse to csr if needed; collaborative_recommendations expects arrays / csr
            return collaborative_recommendations(song, artist, song_df, track_ids, interaction, k)

        if rec_type == "Hybrid":
            # load resources required by hybrid recommender
            song_data = load_song_data(FILTERED_DATA_PATH)
            if song_data is None:
                st.error(f"Could not find filtered song data at {FILTERED_DATA_PATH}")
                return None
            track_ids = load_track_ids()
            interaction = load_interaction_matrix()
            transformed = load_transformed(TRANSFORMED_HYBRID_PATH)
            if track_ids is None or interaction is None or transformed is None:
                st.error("Hybrid requires track ids, interaction matrix and transformed features. One or more resources are missing.")
                return None

            # instantiate and call
            try:
                recommender = HybridRecommendation(
                    song_name=song,
                    artist_name=artist,
                    number_of_recommendations=k,
                    weight_collaborative=0.6,
                    weight_content_based=0.4,
                    song_data=song_data,
                    track_ids=track_ids,
                    interaction_matrix=interaction,
                    transformed_data=transformed
                )
                return recommender.get_recommendations()
            except Exception as e:
                st.error(f"Hybrid recommender failed: {e}")
                return None

    except Exception as e:
        st.error(f"Recommendation error: {e}")
        return None


if st.button("Get Recommendation"):
    with st.spinner("Computing recommendations..."):
        results = get_recommendations(song_name.strip(), artist_name.strip(), number_of_recommendations, recommendation_type)

    if results is None or results.empty:
        st.info("No recommendations found.")
    else:
        st.success(f"Found {len(results)} recommendation(s)")
        # show table and playable previews where available
        st.table(results[['name', 'artist']].head(number_of_recommendations))
        for idx, row in results.head(number_of_recommendations).iterrows():
            st.markdown(f"**{idx+1}. {row['name'].title()} — {row['artist'].title()}**")
            if 'spotify_preview_url' in row and pd.notna(row['spotify_preview_url']):
                try:
                    st.audio(row['spotify_preview_url'], format='audio/mp3')
                except Exception:
                    st.write("Preview available but could not be played.")
            else:
                st.write("No preview available.")
            st.write("---")

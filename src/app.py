import streamlit as st
import pandas as pd
import joblib
from scipy.sparse import load_npz

from content_based_filtering import recommend_songs

# Paths
CLEANED_DATA_PATH = 'data/cleaned_music_data.csv'
TRANSFORMED_DATA_PATH = 'data/transformed_music_data.npz'

# Load data
@st.cache_data
def load_data():
    song_df = pd.read_csv(CLEANED_DATA_PATH)
    transformed_df = load_npz(TRANSFORMED_DATA_PATH)
    return song_df, transformed_df

song_df, transformed_df = load_data()

st.title("Music Recommendation System 🎵")
st.write("Get song recommendations based on your favorite track!")

song_names = song_df['name'].str.title().tolist()
selected_song = st.selectbox("Select a song:", song_names)

k = st.slider("Number of recommendations:", min_value=1, max_value=20, value=10)

if st.button("Recommend"):
    # Lowercase for matching
    recommendations = recommend_songs(selected_song, song_df, transformed_df, k)
    st.subheader("Recommended Songs:")
    st.dataframe(recommendations)
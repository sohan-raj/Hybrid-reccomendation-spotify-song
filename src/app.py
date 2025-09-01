import streamlit as st
import pandas as pd
import joblib
from scipy.sparse import load_npz
from data_cleaning import data_for_content_based
from content_based_filtering import recommend_songs  

# Paths
CLEANED_DATA_PATH = 'data/cleaned_music.csv'
TRANSFORMER_PATH = 'models/column_transformer.joblib'
TRANSFORMED_DATA_PATH = 'data/transformed_music_data.npz'

# Load data and transformer once
@st.cache_data
def load_data():
    df = pd.read_csv(CLEANED_DATA_PATH)
    df_cleaned = data_for_content_based(df)
    transformer = joblib.load(TRANSFORMER_PATH)
    transformed_data = load_npz(TRANSFORMED_DATA_PATH)
    return df, df_cleaned, transformer, transformed_data

df, df_cleaned, transformer, transformed_data = load_data()

# ---------------- Streamlit UI ----------------
st.title("🎶 Content-Based Music Recommender")
st.write("Type in a song you like, and get 10 recommended songs!")

# Input form
song_name = st.text_input("Enter a song name:", "")

if st.button("Recommend"):
    if song_name.strip() == "":
        st.warning("Please enter a song name.")
    else:
        try:
            recs = recommend_songs(song_name, df, transformed_data, k=10)
            if recs.empty:
                st.error(f"Song '{song_name}' not found in dataset.")
            else:
                st.success(f"Top 10 recommendations for '{song_name}':")
                for i, row in recs.iterrows():
                    st.markdown(f"**{row['name']}** by *{row['artist']}*")
                    if pd.notna(row['spotify_preview_url']):
                        st.audio(row['spotify_preview_url'])
        except Exception as e:
            st.error(f"Error: {e}")
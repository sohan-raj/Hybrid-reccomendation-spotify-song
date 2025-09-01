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

frequency_encode_cols = ['year','key','time_signature']
ohe_cols = ['artist']
tfidf_cols = 'tags'
standard_scale_cols = ['duration_ms','loudness','tempo']
min_max_scale_cols = ['danceability', 'energy','speechiness', 'acousticness','instrumentalness', 'liveness', 'valence']


transformer = ColumnTransformer(transformers = [
("frequency_encode",CountEncoder(normalize=True,return_df=True),frequency_encode_cols),
("ohe",OneHotEncoder(handle_unknown='ignore'),ohe_cols),
("tfidf",TfidfVectorizer(max_features=85),tfidf_cols),
("standard_scale",StandardScaler(),standard_scale_cols),
("min_max_scale",MinMaxScaler(),min_max_scale_cols)
],remainder='passthrough',n_jobs=-1)



df = pd.read_csv(CLEANED_DATA_PATH)
df = data_for_content_based(df)

transformer.fit(df)

joblib.dump(transformer, TRANSFORMER_PATH)

transformed_data = transformer.transform(df)

save_npz(TRANSFORMED_DATA_PATH, transformed_data)
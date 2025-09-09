# Hybrid-reccomendation-spotify-song

Hybrid recommendation system for Spotify songs that combines content-based filtering with collaborative filtering. It provides a Streamlit UI to search a song and get similar tracks using three modes: Content-Based, Collaborative, and Hybrid.

This project is reproducible end-to-end using DVC pipelines, with intermediate artifacts saved under `data/` and `models/`.

## Features

- Content-based recommendations using a `ColumnTransformer` with TF-IDF, One-Hot, CountEncoder, scaling, and min-max normalization.
- Collaborative recommendations using a user–item interaction matrix and cosine similarity.
- Hybrid recommendations by weighted combination of content and collaborative scores.
- Streamlit app UI to explore recommendations.
- DVC pipeline for reproducibility of all preprocessing and modeling steps.

## Repository Structure

```
.
├── data/                        # Raw and processed data (created by pipeline)
├── models/                      # Saved transformer(s)
├── notebooks/                   # Exploration notebooks
├── src/                         # Source code
│   ├── app.py                   # Streamlit app
│   ├── data_cleaning.py         # Cleans raw data -> cleaned_music.csv
│   ├── content_based_filtering.py  # Trains transformer and builds content features
│   ├── collaborative_filtering.py  # Builds interaction matrix and CF recommendations
│   ├── transformed_filtered_data.py # Transforms filtered collab data for hybrid
│   └── hybrid_recommendation.py # Hybrid recommender class
├── dvc.yaml                     # DVC pipeline definition
├── dvc.lock                     # DVC lock file (generated)
├── requirements.txt             # Python dependencies
├── LICENSE
└── README.md
```

## Data

Expected raw inputs:

- Songs metadata CSV: `data/Music.csv` (as referenced by `src/data_cleaning.py`)
- User interactions CSV: `data/user.csv` (as referenced in `src/collaborative_filtering.py`)

Important: The `dvc.yaml` currently references `data/music.csv` and `data/User.csv` (different capitalization). Please align your file names to match the actual files on your system. The Python scripts expect `data/Music.csv` and `data/user.csv`.

## DVC Pipeline

Defined in `dvc.yaml`:

1. data_cleaning
   - Command: `python src/data_cleaning.py`
   - Deps: `src/data_cleaning.py`, `data/music.csv`
   - Outs: `data/cleaned_music.csv`

2. content_based_filtering
   - Command: `python src/content_based_filtering.py`
   - Deps: `src/content_based_filtering.py`, `data/cleaned_music.csv`, `src/data_cleaning.py`
   - Outs: `data/transformed_music_data.npz`, `models/column_transformer.joblib`

3. collaborative_filtering
   - Command: `python src/collaborative_filtering.py`
   - Deps: `src/collaborative_filtering.py`, `data/cleaned_music.csv`, `data/User.csv`
   - Outs: `data/collab_filtered_data.csv`, `data/track_ids.npy`, `data/interaction_matrix.npz`

4. transformed_filtered_data
   - Command: `python src/transformed_filtered_data.py`
   - Deps: `src/transformed_filtered_data.py`, `data/collab_filtered_data.csv`, `models/column_transformer.joblib`
   - Outs: `data/transformed_hybrid_data.npz`

5. hybrid_recommendation
   - Command: `python src/hybrid_recommendation.py`
   - Deps: `src/hybrid_recommendation.py`, `data/transformed_hybrid_data.npz`, `data/track_ids.npy`, `data/interaction_matrix.npz`, `data/collab_filtered_data.csv`

Outputs are written to `data/` and `models/` and consumed by the Streamlit app (`src/app.py`).

## Setup

Prerequisites:

- Python 3.9+
- Git
- DVC (installed via pip)

Create and activate a virtual environment and install requirements:

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

Optional: Configure a DVC remote to store artifacts if collaborating across machines:

```bash
dvc remote add -d origin <remote-url>
dvc push
```

## Reproduce the Pipeline

Place raw files under `data/` as described in the Data section, then run:

```bash
dvc repro
```

Alternatively, run steps manually in order:

```bash
python src/data_cleaning.py
python src/content_based_filtering.py
python src/collaborative_filtering.py
python src/transformed_filtered_data.py
```

This will generate the following key artifacts used by the app:

- `data/cleaned_music.csv`
- `models/column_transformer.joblib`
- `data/transformed_music_data.npz`
- `data/collab_filtered_data.csv`
- `data/track_ids.npy`
- `data/interaction_matrix.npz`
- `data/transformed_hybrid_data.npz`

## Run the App

Once the artifacts exist, launch the Streamlit app from the project root:

```bash
streamlit run src/app.py
```

In the UI:

- Use the song search box to select a song.
- Choose recommendation type: `Hybrid`, `Collaborative`, or `Content-Based`.
- Set number of recommendations and click "Get Recommendation".

## How It Works

- `src/content_based_filtering.py` trains a `ColumnTransformer` over features such as `artist`, `year`, `time_signature`, `key`, `tags`, and audio features (e.g., `danceability`, `energy`, `tempo`, etc.). Similarity is computed via cosine similarity over the transformed feature space.
- `src/collaborative_filtering.py` constructs a user–item interaction matrix from `data/user.csv`, aligns `track_id`s, and computes item-to-item cosine similarity.
- `src/hybrid_recommendation.py` normalizes both scores and combines them with weights to produce final ranked recommendations.
- `src/app.py` orchestrates loading the artifacts and serving results via Streamlit.

## Notebooks

Exploration and prototyping notebooks are under `notebooks/`:

- `collaborative_filtering_Spotify_songs.ipynb`
- `content_based_reccomendation_spotify_song.ipynb`

## Troubleshooting

- File name casing must match exactly on your OS. Ensure:
  - Songs CSV: `data/Music.csv` (script expectation) vs `data/music.csv` (DVC). Align one way and update the other if needed.
  - Users CSV: `data/user.csv` (script) vs `data/User.csv` (DVC). Align as well.
- If the Streamlit app shows missing artifact errors, confirm you ran `dvc repro` and that files exist at the paths used in `src/app.py`.
- For large user datasets, Dask is used in `src/collaborative_filtering.py`. Ensure sufficient memory.

## Requirements

See `requirements.txt` for the full list:

- pandas, numpy, scikit-learn, category_encoders, joblib, scipy, streamlit, dask, dvc

## License

This project is licensed under the terms of the MIT License. See `LICENSE` for details.
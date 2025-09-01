import pandas as pd


DATA_PATH  = 'data/Music.csv'

CLEANED_DATA_PATH = 'data/cleaned_music.csv'

def clean_data(data):
    """
    Cleans the input DataFrame by performing the following operations:

    1. Removes duplicate rows based on the 'spotify_id' column.
    2. Drops the 'genre' and 'spotify_id' columns.
    3. Fills missing values in the 'tags' column with 'no_tags'.
    4. Converts the 'name', 'artist', and 'tags' columns to lowercase.
    5. Resets the DataFrame index.

    Args:
        data (pandas.DataFrame): The input DataFrame containing music information.

    Returns:
        pandas.DataFrame: The cleaned DataFrame.
    """

    return (
        data
        .drop_duplicates(subset=['spotify_id'])
        .drop(columns=['genre','spotify_id'])
        .fillna({'tags':'no_tags'})
        .assign(name = lambda df: df['name'].str.lower(),
                artist = lambda df: df['artist'].str.lower(),
                tags = lambda df: df['tags'].str.lower()
        )
        .reset_index(drop=True)
    )

def data_for_content_based(data):
    """
    Prepares the DataFrame for content-based recommendation by dropping unnecessary columns.

    Specifically, it removes the 'track_id', 'name', and 'spotify_preview_url' columns.

    Args:
        data (pandas.DataFrame): The cleaned DataFrame containing music information.

    Returns:
        pandas.DataFrame: The DataFrame ready for content-based filtering.
    """

    return(
        data
        .drop(columns=['track_id','name','spotify_preview_url',])
    ) 


def main(DATA_PATH, CLEANED_DATA_PATH):
    """
    Reads the raw music data, cleans it, and saves the cleaned data to a CSV file.

    Args:
        DATA_PATH (str): Path to the raw music data CSV file.
        CLEANED_DATA_PATH (str): Path where the cleaned CSV file will be saved.

    Returns:
        None
    """
    data = pd.read_csv(DATA_PATH)
    cleaned_data = clean_data(data)
    cleaned_data.to_csv(CLEANED_DATA_PATH, index=False)

if __name__ == "__main__":
    main(DATA_PATH,CLEANED_DATA_PATH)
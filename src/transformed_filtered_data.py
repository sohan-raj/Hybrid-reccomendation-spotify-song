import pandas as pd
from data_cleaning import data_for_content_based
from content_based_filtering import transform_data, save_transformed_data
import joblib

# Path to the filtered collaborative data CSV file
filtered_data_path = 'data/collab_filtered_data.csv'

# Path to save the transformed hybrid data in .npz format
save_path = 'data/transformed_hybrid_data.npz'

transformer_path = 'models/column_transformer.joblib'

def main(filtered_data_path, save_path,transformer_path=transformer_path):
    # Read the filtered collaborative data
    filtered_data = pd.read_csv(filtered_data_path)
    # Prepare the data for content-based filtering (drop unnecessary columns)
    content_based_data = data_for_content_based(filtered_data)
    # Transform the data using the pre-trained transformer
    transformer = joblib.load(transformer_path)
    transformed_data = transformer.transform(content_based_data)
    # Save the transformed data to disk
    save_transformed_data(transformed_data, save_path)

if __name__ == "__main__":
    main(filtered_data_path, save_path)
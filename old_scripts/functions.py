import os
import pandas as pd
import re
import kagglehub
import numpy as np
import seaborn as sns

# Transformer Packages
import torch
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModel
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
from sentence_transformers import SentenceTransformer

def install_data():
    """
    Function to install data for the project. Data isn't included in the repository to save space.
    This function should handle downloading, extracting, and organizing the data as needed.
    """
    output_path = os.path.abspath("./data/raw/imdb.csv")
    
    # Check if the file already exists
    if os.path.exists(output_path):
        print(f"Data already exists at: {output_path}")
        return

    print("Installing data...")
    # Downloading IMDB dataset from Kaggle using KaggleHub
    # Download latest version
    path = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
    print("Path to dataset files:", path)

    # Assuming the dataset is a CSV file in the downloaded path
    dataset_file = os.path.join(path, "IMDB Dataset.csv")  # Adjust filename if different
    
    # Load the dataset and save it to the desired location
    df = pd.read_csv(dataset_file)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Ensure the directory exists
    df.to_csv(output_path, index=False)
    print(f"Data saved to: {output_path}")
    print("Data installation complete.")

def clean_string(raw_string):
    """
    Takes in a raw_string and returns the cleaned version.
    Requires re (regular expressions) module.
    Cleans the string by removing HTML breaks, slashes, and special characters.
    """
    import re
    clean_text = re.sub(r'<br/>', '', raw_string)  # Remove breaks
    clean_text = re.sub(r'<br />', '', clean_text)  # Remove breaks (format 2)
    clean_text = re.sub(r'/', ' ', clean_text)  # Remove slashes   
    clean_text = re.sub(r'[^a-zA-Z0-9 ]', '', clean_text).lower()  # Remove special characters and lowercase
    return clean_text

def save_embedding_with_information(embedding, IMDb_subset, filename):
    # Save embedding for future:
    embedding = pd.DataFrame(embedding) # Pandas dataframe of embeddings
    embedding[['rating']] = IMDb_subset[[]].apply(pd.to_numeric)
    embedding['review'] = IMDb_subset['review']
    embedding.to_csv(f'../Data/{filename}/{filename}.csv', index = False)

def positive_to_negative_vector(Positive, Negative):
    """
    Takes a positive and an negative data point and defines the vector spanning both vectors.
    """
    posneg_vector = Positive.mean().to_frame().T-Negative.mean().to_frame().T
    posneg_vector = pd.DataFrame(posneg_vector)
    return posneg_vector

def project_matrix_to_vector(matrix, vector):
    """Compute the projection of a matrix onto the space spanned by the vector
    Args:
        vector: ndarray of dimension (D, 1), the vector spanning D dimensions that you want to project upon.
        matrix: ndarray of dimension (D, M), the matrix consisting of M vectors that you want to map to the subspace spanned by the vector.
    
    Returns:
        p: projection of matrix onto the subspac spanned by the columns of vector; size (D, 1)
    """
    m = matrix.to_numpy() # Turn into a matrix
    v = vector.to_numpy()[0] #Turn into a numpy array

    # Compute v dot v (denominator)
    v_dot_v = np.dot(v, v)

    # Compute projection of each row of m onto v
    projection = np.outer(np.dot(m, v) / v_dot_v, v)
    projection = pd.DataFrame(projection)

    return projection


def express_matrix_by_vector(matrix, vector):
    """Compute the projection of a matrix onto the space spanned by the vector
    Args:
        vector: ndarray of dimension (D, 1), the vector spanning D dimensions that you want to project upon.
        matrix: ndarray of dimension (D, M), the matrix consisting of M vectors that you want to map to the subspace spanned by the vector.
    
    Returns:
        projection: projection of matrix onto the subspac spanned by the columns of vector; size (D, 1)
        projection_in_1D_subspace: Each embedding projected onto 1 dimensional subspace spanned by input vector.
    """
    unit_vector = vector / np.linalg.norm(vector) # Find the unit vector for interpretatbility by dividing with its norm
    projection = project_matrix_to_vector(matrix, vector) # Find projections, so we can find lengths by finding relations in first dimension
    projection_in_1D_subspace = projection.iloc[:,0]/unit_vector.iloc[:,0][0] # Location in subspace

    return projection, projection_in_1D_subspace

def get_clean_words(raw_string):
    """
    Takes in a raw_string and returns the set of words appearing in the string.
    """
    import re
    try:
        clean_text = str(raw_string)  # Convert to string
        clean_text = re.sub(r'<br/><br/>', ' ', clean_text)  # Remove breaks
        clean_text = re.sub(r'/', ' ', clean_text)  # Replace forward slashes with space
        clean_text = re.sub(r'[^a-zA-Z0-9 ]', '', clean_text).lower()  # Remove special characters and lowercase
        words = clean_text.split()  # Split the text into words
        unique_words = set(words)  # Get unique words
        return unique_words
    except Exception as e:
        print(f"Error processing input: {raw_string}. Error: {e}")
        return set()  # Return an empty set in case of failure


# This script loads the fiction4 sentiment dataset from Huggingface and saves it as a CSV file.
# %%
# Import necessary libraries
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Load from hugginface and save as csv
from src.loader import CorpusLoader
loader = CorpusLoader(text_col='text', label_col='label')
loader.load_from_huggingface("chcaa/fiction4sentiment")
# Save to df
# Process
loader.df = loader.df[['sentence', 'label', 'annotator_1', 'annotator_2', 'annotator_3']]
loader.df.to_csv('fiction4.csv', index=False)

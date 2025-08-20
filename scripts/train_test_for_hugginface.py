#%% 
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.loader import CorpusLoader
from core.embedder import Embedder
from core.projection import ProjectionAnalyzer

# Load packages:
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import spearmanr, pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %%
# --- Initialize loaders ---
MultiLingMPNET = Embedder(model_name="paraphrase-multilingual-mpnet-base-v2")
# Load the fiction4 dataset from HuggingFace -- Training set (positive/negative filter
loader = CorpusLoader(text_col="text", label_col="label")
loader.load_from_huggingface("chcaa/fiction4sentiment", split="train")
loader.split_binary_train_continuous_test(positive_threshold=7, negative_threshold=3, train_size=0.6, random_state=42)

#%%
# Load the EmoBank Corpus
loaderEmoBank = CorpusLoader(path=("../../EmoBank/corpus/emobank.csv"),text_col="text", label_col="V")
loaderEmoBank.load_csv()
# Load the metadata from EmoBank
df_meta = pd.read_csv("../data/raw/meta.tsv", sep="\t")
# Add metadata to the EmoBank loader
loaderEmoBank.df = pd.merge(loaderEmoBank.df, df_meta, left_on="id", right_on="id", how="left")


#%%
# Create a DataFrame for the training set
train_text = pd.DataFrame({
    "sentence": loader.train_df['sentence'],
    "label": (loader.train_df['continuous_label'] - 5) / 4,  # Standardize labels from [1, 9] to [-1, 1]
    "binary_label": loader.train_df['label'],
    "category": loader.train_df['category'],
    "original_dataset": "Fiction4"
})

# Save the training set DataFrame to a CSV file
train_text.to_csv("../data/processed/csv/train.csv", index=False)

# %%
# Create a DataFrame for the test set
test_text = pd.DataFrame({
    "sentence": loader.df['sentence'],
    "label": (loader.test_df['label'] - 5) / 4,  # Standardize labels from [1, 9] to [-1, 1]
    "category": loader.test_df['category'],
    "original_dataset": "Fiction4"
})
# Create a DataFrame for the EmoBank Sentences
EmoBank_text = pd.DataFrame({
    "sentence": loaderEmoBank.df['sentence'],
    "label": (loaderEmoBank.df['label'] - 3) / 2,  # Standardize
    "category": loaderEmoBank.df['category'],
    "original_dataset": "EmoBank"
})

EmoBank_text=EmoBank_text[EmoBank_text['category'].isin(['letters', 'blog', 'newspaper', 'essays', 'fiction', 'travel-guides'])]

test_text = pd.concat([test_text, EmoBank_text], ignore_index=True)
test_text.to_csv("../data/processed/csv/test.csv", index=False)

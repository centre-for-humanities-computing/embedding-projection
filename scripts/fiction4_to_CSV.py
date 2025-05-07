
# %%
import json
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

# Path to your JSON file
with open('../data/raw/fiction4_data.json', 'r') as f:
    data = json.load(f)
# %% 
# Convert embeddings to DataFrame and add sentiment and review
df = pd.DataFrame({
    'sentences' : data['SENTENCE_ENGLISH'],
    'sentiment' : data['HUMAN'],
    'ANNOTATOR_1': data['ANNOTATOR_1'],
    'ANNOTATOR_2': data['ANNOTATOR_2'],
    'ANNOTATOR_3': data['ANNOTATOR_3'],
})

# Add a label based on sentiment value
# If sentiment <= 3, label is 'negative'; if sentiment >= 7, label is 'positive'; else 'neutral'
df['label'] = df['sentiment'].apply(lambda x: 'negative' if x <= 3 else ('positive' if x >= 7 else 'neutral'))

df.to_csv('../data/raw/fiction4.csv', index=False)

print("CSV with embeddings saved successfully.")

# %%

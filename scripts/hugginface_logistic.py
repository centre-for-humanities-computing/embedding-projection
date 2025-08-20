#%% 
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.loader import CorpusLoader
from core.embedder import Embedder
from core.projection import ProjectionAnalyzer

# Load packages:
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import spearmanr
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# --- Initialize loaders ---
MultiLingMPNET = Embedder(model_name="paraphrase-multilingual-mpnet-base-v2")
# %%
# --- Initialize loaders ---
# Load the fiction4 dataset from HuggingFace -- Training set (positive/negative filter
loader = CorpusLoader(text_col="text", label_col="label")
loader.load_from_huggingface("chcaa/fiction4sentiment", split="train")
loader.split_binary_train_continuous_test(positive_threshold=7, negative_threshold=3, train_size=0.6, random_state=42)
# Print the distribution of labels in the training set
print(loader.df.shape)
print(loader.train_df.shape)
print(loader.train_df['label'].value_counts())
print(loader.test_df.shape)
#%%
# Load the EmoBank Corpus
loaderEmoBank = CorpusLoader(path=("../../EmoBank/corpus/emobank.csv"),text_col="text", label_col="V")
loaderEmoBank.load_csv()
loaderEmoBank.df.head()

# %%
# --- Embed model ---
train_MPNET = MultiLingMPNET.embed(loader.train_texts, cache_path="../data/processed/embeddings/fiction4_train_MultiLingMPNET_neg3_pos7.csv")
train_MPNET["label"] = loader.train_labels

test_MPNET = MultiLingMPNET.embed(loader.test_texts, cache_path="../data/processed/embeddings/fiction4_test_MultiLingMPNET_hug.csv")
test_MPNET["label"] = loader.test_labels

all_MPNET =  MultiLingMPNET.embed(loader.texts, cache_path="../data/processed/embeddings/fiction4_MultiLingMPNET_hug.csv")
all_MPNET['label'] = loader.labels
#%% 
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import spearmanr
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EmoBank_MPNET = MultiLingMPNET.embed(loaderEmoBank.df['sentence'], cache_path="../data/processed/embeddings/EmoBank_MultiLingMPNET.csv")
EmoBank_MPNET["valence"] = loaderEmoBank.df['label']


## %
# === 1. Project embeddings ===
analyzer = ProjectionAnalyzer(matrix_concept=train_MPNET, matrix_project=EmoBank_MPNET)
analyzer.project()

# === 2. Make a plot ===
x_raw = analyzer.projected_in_1D
y = EmoBank_MPNET['valence'].values

# === 3. Rescale projections to [0.2, 9.2] ===
scaler = MinMaxScaler((1, 5))
scaled = scaler.fit_transform(x_raw.values.reshape(-1, 1)).flatten()

# === 4. Calculate Spearman correlation ===
corr, _ = spearmanr(scaled, y)
# === 5. Create the plot ===
g = sns.jointplot(x=scaled, y=y, kind="scatter", marginal_kws=dict(fill=True),
                  alpha=0.5, height=7)
g.ax_joint.set_xlim(1, 5)
g.ax_joint.set_ylim(1, 5)
g.ax_joint.set(xlabel='Subspace Projection', ylabel='Human Gold Standard - Valence')
g.ax_joint.set_title(f'Trained on fiction4 - Tested on EmoBank - Naive Vector Definition', pad=90)
g.ax_joint.text(0.05, 0.95, f'Spearman ρ = {corr:.2f}', transform=g.ax_joint.transAxes,
                fontsize=12, color='blue')
g.figure.subplots_adjust(top=0.9)


# %%

# === 1. Project embeddings ===
analyzer = ProjectionAnalyzer(matrix_concept=train_MPNET, matrix_project=EmoBank_MPNET)
analyzer.concept_vector = analyzer.define_logistic_concept_vector()
analyzer.project()

# === 2. Make a plot ===
x_raw = analyzer.projected_in_1D
y = EmoBank_MPNET['valence'].values

# === 3. Rescale projections to [0.2, 9.2] ===
scaler = MinMaxScaler((1, 5))
scaled = scaler.fit_transform(x_raw.values.reshape(-1, 1)).flatten()

# === 4. Calculate Spearman correlation ===
corr, _ = spearmanr(scaled, y)
# === 5. Create the plot ===
g = sns.jointplot(x=scaled, y=y, kind="scatter", marginal_kws=dict(fill=True),
                  alpha=0.5, height=7)
g.ax_joint.set_xlim(1, 5)
g.ax_joint.set_ylim(1, 5)
g.ax_joint.set(xlabel='Subspace Projection', ylabel='Human Gold Standard - Valence')
g.ax_joint.set_title(f'Trained on fiction4 - Tested on EmoBank - Logistic Vector Definition', pad=90)
g.ax_joint.text(0.05, 0.95, f'Spearman ρ = {corr:.2f}', transform=g.ax_joint.transAxes,
                fontsize=12, color='blue')
g.figure.subplots_adjust(top=0.9)

# %%
## %
# === 1. Project embeddings ===
analyzer = ProjectionAnalyzer(matrix_concept=train_MPNET, matrix_project=test_MPNET)
analyzer.project()

# === 2. Make a plot ===
x_raw = analyzer.projected_in_1D
y = test_MPNET['label'].values

# === 3. Rescale projections to [0.2, 9.2] ===
scaler = MinMaxScaler((1, 10))
scaled = scaler.fit_transform(x_raw.values.reshape(-1, 1)).flatten()

# === 4. Calculate Spearman correlation ===
corr, _ = spearmanr(scaled, y)
# === 5. Create the plot ===
g = sns.jointplot(x=scaled, y=y, kind="scatter", marginal_kws=dict(fill=True),
                  alpha=0.5, height=7)
g.ax_joint.set_xlim(1, 10)
g.ax_joint.set_ylim(1, 10)
g.ax_joint.set(xlabel='Subspace Projection', ylabel='Human Gold Standard - Valence')
g.ax_joint.set_title(f'Trained on fiction4 - Tested on fiction4 - Naive Vector Definition', pad=90)
g.ax_joint.text(0.05, 0.95, f'Spearman ρ = {corr:.2f}', transform=g.ax_joint.transAxes,
                fontsize=12, color='blue')
g.figure.subplots_adjust(top=0.9)
# %%
# === 1. Project embeddings ===
analyzer = ProjectionAnalyzer(matrix_concept=train_MPNET, matrix_project=test_MPNET)
analyzer.concept_vector = analyzer.define_logistic_concept_vector()
analyzer.project()

# === 2. Make a plot ===
x_raw = analyzer.projected_in_1D
y = test_MPNET['label'].values

# === 3. Rescale projections to [0.2, 9.2] ===
scaler = MinMaxScaler((1, 10))
scaled = scaler.fit_transform(x_raw.values.reshape(-1, 1)).flatten()

# === 4. Calculate Spearman correlation ===
corr, _ = spearmanr(scaled, y)
# === 5. Create the plot ===
g = sns.jointplot(x=scaled, y=y, kind="scatter", marginal_kws=dict(fill=True),
                  alpha=0.5, height=7)
g.ax_joint.set_xlim(1, 10)
g.ax_joint.set_ylim(1, 10)
g.ax_joint.set(xlabel='Subspace Projection', ylabel='Human Gold Standard - Valence')
g.ax_joint.set_title(f'Trained on fiction4 - Tested on Fiction4 - Logistic Vector Definition', pad=90)
g.ax_joint.text(0.05, 0.95, f'Spearman ρ = {corr:.2f}', transform=g.ax_joint.transAxes,
                fontsize=12, color='blue')
g.figure.subplots_adjust(top=0.9)
# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the fiction4 dataset (adjust path if needed)
df = pd.read_csv("../data/raw/fiction4.csv")
df.head()
# If coder columns are named differently, adjust here
coder_cols = ['ANNOTATOR_1', 'ANNOTATOR_2', 'ANNOTATOR_3']

# Drop rows with missing coder ratings
df_coders = df[coder_cols].dropna()

# Pairplot to visualize spread/agreement between coders
sns.pairplot(df_coders, kind='scatter', plot_kws={'alpha':0.5})
plt.suptitle('Spread between Sentiment Coders in Fiction4', y=1.02)
plt.show()

import itertools

# Plot heatmaps of agreement for each coder pair
for coder_a, coder_b in itertools.combinations(coder_cols, 2):
    agreement_matrix = pd.crosstab(df_coders[coder_a], df_coders[coder_b])
    plt.figure(figsize=(6,5))
    sns.heatmap(agreement_matrix, annot=True, fmt="d", cmap="Blues")
    plt.title(f'Agreement Heatmap: {coder_a} vs {coder_b}')
    plt.xlabel(coder_b)
    plt.ylabel(coder_a)
    plt.show()
# Percent agreement for each pair
for coder_a, coder_b in itertools.combinations(coder_cols, 2):
    agree = (df_coders[coder_a] == df_coders[coder_b]).mean()
    print(f"Percent agreement between {coder_a} and {coder_b}: {agree:.2%}")
# %%
analyzer = ProjectionAnalyzer(matrix_concept=train_MPNET, matrix_project=all_MPNET)
# analyzer.concept_vector = analyzer.define_logistic_concept_vector()
analyzer.project()

df = pd.read_csv("../data/raw/fiction4.csv")
coder_cols = ['ANNOTATOR_1', 'ANNOTATOR_2', 'ANNOTATOR_3']

for coder in coder_cols:
    y = df[coder].values 
    # Remove NaNs for fair comparison
    mask = ~np.isnan(y)
    y_valid = y[mask]
    x_valid = analyzer.projected_in_1D[mask]
    # Rescale projections to [1, 10]
    scaler = MinMaxScaler((1, 10))
    x_valid = scaler.fit_transform(x_valid.values.reshape(-1, 1)).flatten()
    # Calculate Spearman correlation    
    corr, _ = spearmanr(x_valid, y_valid)
    g = sns.jointplot(x=x_valid, y=y_valid, kind="scatter", marginal_kws=dict(fill=True),
                      alpha=0.5, height=7)
    g.ax_joint.set(xlabel='Subspace Projection', ylabel=f'Annotator: {coder}')
    g.ax_joint.set_title(f'Trained on fiction4 - Tested on entire fiction4 - {coder}', pad=90)
    g.ax_joint.text(0.05, 0.95, f'Spearman ρ = {corr:.2f}', transform=g.ax_joint.transAxes,
                    fontsize=12, color='blue')
    g.figure.subplots_adjust(top=0.9)
    plt.show()

# %%

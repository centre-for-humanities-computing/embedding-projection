# %%
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# %%
from core.loader import CorpusLoader
from core.embedder import Embedder

# --- Initialize loaders ---
train_loader = CorpusLoader(path="../data/raw/imdb.csv", text_col="review", label_col="sentiment")
test_loader = CorpusLoader(path="../data/raw/imdb.csv", text_col="review", label_col="sentiment")

# --- Load train and test slices ---
train_loader.load_csv(start=0, n_rows=500)
train_loader.clean()

test_loader.load_csv(start=500, n_rows=100)
test_loader.clean()

# --- Embed using the same model ---
MiniLM = Embedder(model_name="all-MiniLM-L6-v2")
train_embed = MiniLM.embed(train_loader.texts, cache_path="../data/processed/embeddings/imdb_train.csv")
test_embed = MiniLM.embed(test_loader.texts, cache_path="../data/processed/embeddings/imdb_test.csv")

# --- Add labels back to embeddings ---
train_embed["label"] = train_loader.labels
test_embed["label"] = test_loader.labels

from core.projection import ProjectionAnalyzer
ProjectionAnalyzer = ProjectionAnalyzer(matrix_concept=train_embed, matrix_project=test_embed)
ProjectionAnalyzer.project()

# %%
from core.loader import CorpusLoader
from core.embedder import Embedder

# --- load and clean fiction datasets ---
fiction4_loader = CorpusLoader(path="../data/raw/fiction4.csv", text_col="sentences", label_col="label")
fiction4_loader.load_csv()
fiction4_loader.clean()
fiction4_filtered_loader = fiction4_loader.filter_labels(positive_labels=["positive"], negative_labels=["negative"])

# --- Embed using two models ---
MiniLM = Embedder(model_name="all-MiniLM-L6-v2")
MPNET = Embedder(model_name="all-mpnet-base-v2")

# --- Embed the fiction4 dataset using MiniLM and MPNET ---
train_mini = MiniLM.embed(fiction4_filtered_loader.texts, cache_path="../data/processed/embeddings/fiction4_train_miniLM.csv")
train_mini["label"] = fiction4_filtered_loader.labels
train_MPNET = MPNET.embed(fiction4_filtered_loader.texts, cache_path="../data/processed/embeddings/fiction4_train_mpnet.csv")
train_MPNET["label"] = fiction4_filtered_loader.labels
test_mini = MiniLM.embed(fiction4_loader.texts, cache_path="../data/processed/embeddings/fiction4_test_miniLM.csv")
test_mini["label"] = fiction4_loader.df['sentiment']
test_MPNET = MPNET.embed(fiction4_loader.texts, cache_path="../data/processed/embeddings/fiction4_test_mpnet.csv")
test_MPNET["label"] = fiction4_loader.df['sentiment']


# %%
from core.projection import ProjectionAnalyzer
# === 1. Project embeddings ===
analyzer = ProjectionAnalyzer(matrix_concept=train_MPNET, matrix_project=test_MPNET)
analyzer.project()


# === 1. Make a plot ===
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import spearmanr
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
x_raw = analyzer.projected_in_1D
y = test_MPNET["label"].values

# === 2. Rescale projections to [0.2, 9.2] ===
scaler = MinMaxScaler((1, 9))
scaled = scaler.fit_transform(x_raw.values.reshape(-1, 1)).flatten()

# === 3. Calculate Spearman correlation ===
corr, _ = spearmanr(scaled, y)

# === 4. Create the plot ===
g = sns.jointplot(x=scaled, y=y, kind="scatter", marginal_kws=dict(fill=True),
                  alpha=0.5, height=7)
g.ax_joint.set_xlim(0, 10)
g.ax_joint.set_xticks(np.linspace(1, 9, 9))  # Custom ticks from 1 to 10
g.ax_joint.set(xlabel='Fiction4 Subspace Projection', ylabel='Human Gold Standard')
g.ax_joint.set_title(f'Scatterplot with Correlation (MPNET Rescaled)', pad=90)
g.ax_joint.text(0.05, 0.95, f'Spearman ρ = {corr:.2f}', transform=g.ax_joint.transAxes,
                fontsize=12, color='blue')
g.figure.subplots_adjust(top=0.9)
g.figure.savefig('../img/Scatterplot_fiction4_rescaled.png', dpi=300, bbox_inches='tight')
plt.show()



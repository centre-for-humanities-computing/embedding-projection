#%% 
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.loader import CorpusLoader
from core.embedder import Embedder
from core.projection import ProjectionAnalyzer
# --- Initialize loaders ---
MultiLingMPNET = Embedder(model_name="paraphrase-multilingual-mpnet-base-v2")
# %%
# --- Initialize loaders ---
# Load the fiction4 dataset from HuggingFace -- Training set (positive/negative filter)
train_loader = CorpusLoader(text_col="text", label_col="label")
train_loader.load_from_huggingface("chcaa/fiction4sentiment", split="train")
train_loader.transform_labels_to_binary(positive_threshold=7, negative_threshold=3)
print(train_loader.df['label'].value_counts())
train_loader.drop_neutral()
print(train_loader.df['label'].value_counts())

# Load the fiction4 dataset from HuggingFace -- Training set (all labels)
test_loader = CorpusLoader(text_col="text", label_col="label")
test_loader.load_from_huggingface("chcaa/fiction4sentiment", split="train")


# %%
# --- Embed model ---
train_MPNET = MultiLingMPNET.embed(train_loader.texts, cache_path="../data/processed/embeddings/fiction4_train_MultiLingMPNET_neg3_pos7.csv")
train_MPNET["label"] = train_loader.labels

test_MPNET = MultiLingMPNET.embed(test_loader.texts, cache_path="../data/processed/embeddings/fiction4_test_MultiLingMPNET_hug.csv")
test_MPNET["label"] = test_loader.labels
# %%
# Define masks for english and danish for training set
english_mask_train = train_loader.df['category'].isin(['poetry', 'prose'])
danish_mask_train = ~train_loader.df['category'].isin(['poetry', 'prose'])
HEMMING_train = train_loader.df['category'].isin(['prose'])
SYLVIA_train = train_loader.df['category'].isin(['poetry'])
HCA_train = train_loader.df['category'].isin(['fairytales'])
HYMN_train = train_loader.df['category'].isin(['hymns'])

# Define masks for english and danish for test set
english_mask_test = test_loader.df['category'].isin(['poetry', 'prose'])
danish_mask_test = ~test_loader.df['category'].isin(['poetry', 'prose'])
HEMMING_test = test_loader.df['category'].isin(['prose'])
SYLVIA__test = test_loader.df['category'].isin(['poetry'])
HCA_test = test_loader.df['category'].isin(['fairytales'])
HYMN_test = test_loader.df['category'].isin(['hymns'])

train_MPNET[english_mask_train]

#%%
from core.projection import ProjectionAnalyzer
# === 1. Project embeddings ===
analyzer = ProjectionAnalyzer(matrix_concept=train_MPNET[SYLVIA_train], matrix_project=test_MPNET[HEMMING_test])
analyzer.project()

# === 1. Make a plot ===
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import spearmanr
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
x_raw = analyzer.projected_in_1D
y = test_MPNET[HEMMING_test]['label'].values

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
plt.show()
# %%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import spearmanr

from core.projection import ProjectionAnalyzer

def plot_projection_grid(train_test_pairs, train_MPNET, test_MPNET, test_loader):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for i, (train_mask, test_mask, title) in enumerate(train_test_pairs):
        # Project embeddings
        analyzer = ProjectionAnalyzer(matrix_concept=train_MPNET[train_mask],
                                      matrix_project=test_MPNET[test_mask])
        analyzer.project()
        x_raw = analyzer.projected_in_1D
        y = test_MPNET[test_mask]['label'].values

        # Rescale projections
        scaler = MinMaxScaler((1, 9))
        scaled = scaler.fit_transform(x_raw.values.reshape(-1, 1)).flatten()

        # Compute Spearman correlation
        corr, _ = spearmanr(scaled, y)

        # Plot
        ax = axes[i]
        sns.scatterplot(x=scaled, y=y, alpha=0.5, ax=ax)
        ax.set_xlim(0, 10)
        ax.set_xticks(np.linspace(1, 9, 9))
        ax.set_xlabel('Fiction4 Subspace Projection')
        ax.set_ylabel('Human Gold Standard')
        ax.set_title(title + f"\nSpearman ρ = {corr:.2f}", fontsize=11)
    
    plt.tight_layout()
    plt.show()

train_test_pairs_fairytales_hymns = [
    (HCA_train, HCA_test, "Trained on Fairytales → Tested on Fairytales"),
    (HCA_train, HYMN_test, "Trained on Fairytales → Tested on Hymns"),
    (HYMN_train, HCA_test, "Trained on Hymns → Tested on Fairytales"),
    (HYMN_train, HYMN_test, "Trained on Hymns → Tested on Hymns")
]


train_test_pairs_english = [
    (HEMMING_train, HEMMING_test, "Trained on Prose → Tested on Prose"),
    (HEMMING_train, SYLVIA__test, "Trained on Prose → Tested on Poetry"),
    (SYLVIA_train, HEMMING_test, "Trained on Poetry → Tested on Prose"),
    (SYLVIA_train, SYLVIA__test, "Trained on Poetry → Tested on Poetry")
]
train_test_pairs_english_danish = [
    (english_mask_train, english_mask_test, "Trained on English → Tested on English"),
    (english_mask_train, danish_mask_test, "Trained on English → Tested on Danish"),
    (danish_mask_train, english_mask_test, "Trained on Danish → Tested on English"),
    (danish_mask_train, danish_mask_test, "Trained on Danish → Tested on Danish")
]


plot_projection_grid(train_test_pairs_english, train_MPNET, test_MPNET, test_loader)

# %%
plot_projection_grid(train_test_pairs_fairytales_hymns, train_MPNET, test_MPNET, test_loader)

# %%

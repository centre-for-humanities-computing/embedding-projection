
# --- Correlation by genre in test_MPNET ---
# This block should be placed after test_MPNET, scaled, and all dependencies are defined
if 'test_MPNET' in globals() and 'scaled' in globals():
    genres = ['poetry', 'prose', 'hymns', 'fairytales']
    print('Correlations in test_MPNET by genre:')
    for genre in genres:
        mask = test_MPNET['category'] == genre
        # Ensure mask is aligned with scaled
        if hasattr(scaled, '__len__') and len(scaled) == len(test_MPNET):
            x = scaled[mask]
        else:
            # fallback: recalculate scaled for test_MPNET
            scaler_tmp = MinMaxScaler((-1, 1))
            x = scaler_tmp.fit_transform(test_MPNET['projected_in_1D'].values.reshape(-1, 1)).flatten()[mask]
        y_genre = test_MPNET.loc[mask, 'label']
        if len(x) > 1 and len(y_genre) > 1:
            spearman_corr, spearman_p = spearmanr(x, y_genre)
            pearson_corr, pearson_p = pearsonr(x, y_genre)
            print(f"{genre.capitalize()}: Spearman ρ = {spearman_corr:.2f} (p={spearman_p:.2g}), Pearson r = {pearson_corr:.2f} (p={pearson_p:.2g})")
        else:
            print(f"{genre.capitalize()}: Not enough data for correlation.")
else:
    print('test_MPNET or scaled not defined yet. Please run the relevant cells first.')
#%%
# Load the necessary modules and load data into all_MPNET DataFrame
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

# --- Initialize loaders ---
MultiLingMPNET = Embedder(model_name="paraphrase-multilingual-mpnet-base-v2")
# Load the fiction4 dataset from HuggingFace -- Training set (positive/negative filter
loader = CorpusLoader(text_col="text", label_col="label")
loader.load_from_huggingface("chcaa/fiction4sentiment", split="train")
loader.split_binary_train_continuous_test(positive_threshold=7, negative_threshold=3, train_size=0.6, random_state=42)
# %%
# --- Embed model ---
train_MPNET = MultiLingMPNET.embed(loader.train_texts, cache_path="../data/processed/embeddings/fiction4_train_MultiLingMPNET_neg3_pos7.csv")
train_MPNET["label"] = loader.train_labels
train_MPNET['category'] = loader.train_df['category']
train_MPNET['original_dataset'] = "Fiction4"

test_MPNET = MultiLingMPNET.embed(loader.test_texts, cache_path="../data/processed/embeddings/fiction4_test_MultiLingMPNET_hug.csv")
test_MPNET["label"] = (loader.test_df['label']-5)/4  # Standardize labels from [1, 9] to [-1, 1]
test_MPNET['category'] = loader.test_df['category']
test_MPNET['original_dataset'] = "Fiction4"
# Load the EmoBank Corpus
loaderEmoBank = CorpusLoader(path=("../../EmoBank/corpus/emobank.csv"),text_col="text", label_col="V")
loaderEmoBank.load_csv()
# Load the metadata from EmoBank
df_meta = pd.read_csv("../data/raw/meta.tsv", sep="\t")
# Add metadata to the EmoBank loader
loaderEmoBank.df = pd.merge(loaderEmoBank.df, df_meta, left_on="id", right_on="id", how="left")

# Standardize loaderEmobank.df['label'] to mean 0 and std 1
EmoBank_MPNET = MultiLingMPNET.embed(loaderEmoBank.df['sentence'], cache_path="../data/processed/embeddings/EmoBank_MultiLingMPNET.csv")
EmoBank_MPNET["label"] = (loaderEmoBank.df['label'] - 3) / 2  # Standardize labels from [1, 5] to [-1, 1]
EmoBank_MPNET['category'] = loaderEmoBank.df['category']
EmoBank_MPNET['original_dataset'] = "EmoBank"
EmoBank_MPNET = EmoBank_MPNET[EmoBank_MPNET['category'].isin(['letters', 'blog', 'newspaper', 'essays', 'fiction', 'travel-guides'])]

# Append test_MPNET and EmoBank_MPNET, as they have the same columns
all_MPNET = pd.concat([test_MPNET, EmoBank_MPNET], ignore_index=True)
all_MPNET



#%%
# Run outputs:
# Create a mask for all_MPNET to select only the rows where original_dataset is "EmoBank"
EmoBank_Mask = all_MPNET['original_dataset'] == "EmoBank"
fiction4_mask = all_MPNET['original_dataset'] == "Fiction4"
english_mask = fiction4_mask & all_MPNET['category'].isin(['poetry', 'prose'])
danish_mask = fiction4_mask & ~all_MPNET['category'].isin(['poetry', 'prose'])

# === 1. Project embeddings ===
analyzer = ProjectionAnalyzer(matrix_concept=train_MPNET.iloc[:, :-2], matrix_project=test_MPNET.iloc[:, :-2])
analyzer.project()

# === 2. Make a plot ===
x_raw = analyzer.projected_in_1D
y = test_MPNET['label'].values


# === 3. Rescale projections to [-1, 1] ===
scaler = MinMaxScaler((-1, 1))
scaled = scaler.fit_transform(x_raw.values.reshape(-1, 1)).flatten()

# === 4. Calculate Spearman and Pearson correlation with p-values ===
corr_spearman, p_spearman = spearmanr(scaled, y)
corr_pearson, p_pearson = pearsonr(scaled, y)

# === 5. Create the plot ===
#g = sns.jointplot(x=scaled, y=y, kind="scatter", marginal_kws=dict(fill=True),
#                  alpha=0.5, height=7)
g = sns.jointplot(x=scaled, y=y, kind="scatter", alpha=0.5, height=7)
g.ax_joint.set_xlim(-1, 1)
g.ax_joint.set_xticks(np.linspace(-1, 1, 9))  # Custom ticks from 1 to 10
g.ax_joint.set(xlabel='Sentiment Subspace Projection', ylabel='Human Gold Standard')
g.ax_joint.set_title(f'Scatterplot with Correlation (paraphrase-multilingual-mpnet-base-v2)', pad=90)
g.ax_joint.text(0.05, 0.95, f'Spearman ρ = {corr_spearman:.2f} (p={p_spearman:.2f})\nPearson r = {corr_pearson:.2f} (p={p_pearson:.2f})', transform=g.ax_joint.transAxes,
                fontsize=12, color='blue')
# g.figure.subplots_adjust(top=0.9)
plt.show()


plt.figure(figsize=(8, 7))
sns.scatterplot(x=scaled, y=y, alpha=0.5)
plt.xlim(-1.1, 1.1)
plt.xticks(np.linspace(-1, 1, 9))
plt.xlabel('Sentiment Projection')
plt.ylim(-1.1, 1.1)
plt.ylabel('Human Gold Standard')
plt.title('Sentiment Projection (paraphrase-multilingual-mpnet-base-v2)')
plt.text(0.05, 0.95, f'Spearman ρ = {corr_spearman:.2f} (p={p_spearman:.2f})\nPearson r = {corr_pearson:.2f} (p={p_pearson:.2f})',
         transform=plt.gca().transAxes, fontsize=12, color='blue', verticalalignment='top')
plt.show()

# %%
# Make a Dataframe with the projected values and labels
my_df = pd.DataFrame({
    'projected_in_1D': analyzer.projected_in_1D,
    'text': loader.test_texts
})

# %%
# count how many of each unique labels in train_MPNET["label"]
train_MPNET["label"].value_counts().sort_index()
# %%
# === 1. Project embeddings ===
analyzer = ProjectionAnalyzer(matrix_concept=train_MPNET.iloc[:, :-2], matrix_project=EmoBank_MPNET.iloc[:, :-2])
analyzer.project()

# === 2. Make a plot ===
x_raw = analyzer.projected_in_1D
y = EmoBank_MPNET['label'].values


# === 3. Rescale projections to [-1, 1] ===
scaler = MinMaxScaler((-1, 1))
scaled = scaler.fit_transform(x_raw.values.reshape(-1, 1)).flatten()



# create a histogram of scaled and test_MPNET['label'].values, enforce labels on x-axis to be -1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1
bins = np.linspace(-1, 1, 50)  # 50 bins from -1 to 1

# Plot 1: Histogram of Sentiment Projection
plt.figure(figsize=(7, 5))
sns.histplot(scaled, bins=bins, kde=True)
plt.xticks(np.linspace(-1, 1, 9))
plt.title('Histogram of Sentiment Projection')
plt.xlabel('Projected Values')
plt.ylabel('Frequency')
plt.show()

# Plot 2: Histogram of Human Ratings
plt.figure(figsize=(7, 5))
sns.histplot(EmoBank_MPNET['label'].values, bins=50, kde=True)
plt.xticks(np.linspace(-1, 1, 9))
plt.title('Histogram of Human Ratings - EmoBank')
plt.xlabel('Human Ratings')
plt.ylabel('Frequency')
plt.show()
# %%

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
analyzer = ProjectionAnalyzer(matrix_concept=train_MPNET.iloc[:, :-2], matrix_project=all_MPNET.iloc[:, :-2])
analyzer.project()

# === 2. Make a plot ===
x_raw = analyzer.projected_in_1D
y = all_MPNET['label'].values


# === 3. Rescale projections to [-1, 1] ===
scaler = MinMaxScaler((-1, 1))
scaled = scaler.fit_transform(x_raw.values.reshape(-1, 1)).flatten()

# === 4. Calculate Spearman and Pearson correlation with p-values ===
corr_spearman, p_spearman = spearmanr(scaled, y)
corr_pearson, p_pearson = pearsonr(scaled, y)

# === 5. Create the plot ===
g = sns.jointplot(x=scaled, y=y, kind="scatter", marginal_kws=dict(fill=True),
                  alpha=0.5, height=7)
g.ax_joint.set_xlim(-1, 1)
g.ax_joint.set_xticks(np.linspace(-1, 1, 9))  # Custom ticks from 1 to 10
g.ax_joint.set(xlabel='Sentiment Subspace Projection', ylabel='Human Gold Standard')
g.ax_joint.set_title(f'Scatterplot with Correlation (paraphrase-multilingual-mpnet-base-v2)', pad=90)
g.ax_joint.text(0.05, 0.95, f'Spearman ρ = {corr_spearman:.2f} (p={p_spearman:.2f})\nPearson r = {corr_pearson:.2f} (p={p_pearson:.2f})', transform=g.ax_joint.transAxes,
                fontsize=12, color='blue')
g.figure.subplots_adjust(top=0.9)
plt.show()

# Loop through all masks and print correlations in the requested order
mask_dict = []
# 1. Fiction4 English, Danish, All
mask_dict.append(("Fiction4 All", fiction4_mask))
mask_dict.append(("Fiction4 English", english_mask))
mask_dict.append(("Fiction4 Danish", danish_mask))


# 2. Each individual genre in Fiction4 (alphabetical)
fiction4_categories = sorted(all_MPNET.loc[fiction4_mask, 'category'].unique())
for cat in fiction4_categories:
    cat_mask = fiction4_mask & (all_MPNET['category'] == cat)
    mask_dict.append((f'Fiction4 ({cat})', cat_mask))

# 3. EmoBank (all)
mask_dict.append(("EmoBank", EmoBank_Mask))

# 4. Each individual category in EmoBank (alphabetical)
emobank_categories = sorted(all_MPNET.loc[EmoBank_Mask, 'category'].unique())
for cat in emobank_categories:
    cat_mask = EmoBank_Mask & (all_MPNET['category'] == cat)
    mask_dict.append((f'EmoBank ({cat})', cat_mask))

for name, mask in mask_dict:
    x_masked = scaled[mask]
    y_masked = y[mask]
    spearman_corr, spearman_p = spearmanr(x_masked, y_masked)
    pearson_corr, pearson_p = pearsonr(x_masked, y_masked)
    print(f"{name}: Spearman ρ = {spearman_corr:.2f} (p={spearman_p:.2f}), Pearson r = {pearson_corr:.2f} (p={pearson_p:.2f})")

# %%
loaderEmoBank.df['category'].unique()
#%%
# --- Genre-specific EmoBank training and prediction with proper concept vector training ---
genres = ['letters', 'blog', 'newspaper', 'essays', 'fiction', 'travel-guides'] #   failed
results = []
for genre in genres:
    # Filter original EmoBank DataFrame for this genre
    genre_df = loaderEmoBank.df[loaderEmoBank.df['category'] == genre].copy()
    if genre_df.empty:
        print(f"No data for genre: {genre}")
        continue
    


    # Create a new loader for this genre, with correct label scale (1-5)
    genre_loader = CorpusLoader(text_col='sentence', label_col='label')
    genre_df.head()
    genre_loader.load_from_dataframe(genre_df)
    # Use extremes split for 1-5 scale (e.g., 10% min + 10% max for train, rest for test)
    genre_loader.split_extremes_train_middle_test(extreme_frac=0.1, remove_frac_from_extremes=0.5, random_state=42)
    # Embed train and test sets
    train_emb = MultiLingMPNET.embed(genre_loader.train_texts)
    test_emb = MultiLingMPNET.embed(genre_loader.test_texts)
    # Add labels for projection/correlation
    train_emb['label'] = genre_loader.train_labels
    if train_emb.shape[0] == 0 or test_emb.shape[0] == 0:
        print(f"Skipping genre {genre}: empty train or test set.")
        continue
    # Check how many postiive and negative labels are in the train set
    print(f"Genre: {genre}, Train set labels: Positive({genre_loader.train_labels.count('positive')}), Negative({genre_loader.train_labels.count('negative')})  ")

    test_emb['label'] = genre_loader.test_labels
    # Train concept vector on train set, predict on test set
    analyzer = ProjectionAnalyzer(matrix_concept=train_emb, matrix_project=test_emb)
    analyzer.project()

    x_raw = analyzer.projected_in_1D
    if x_raw is None:
        print(f"Projection failed for genre {genre}.")
        continue
    y = np.array(test_emb['label'])
    # Standardize labels from [1,5] to [-1,1]
    y = (y - 3) / 2
    # Rescale projections to [-1, 1]
    scaler = MinMaxScaler((-1, 1))
    scaled = scaler.fit_transform(x_raw.values.reshape(-1, 1)).flatten()
    # Calculate correlations
    spearman_corr, spearman_p = spearmanr(scaled, y)
    pearson_corr, pearson_p = pearsonr(scaled, y)
    results.append({
        'genre': genre,
        'spearman_corr': spearman_corr,
        'spearman_p': spearman_p,
        'pearson_corr': pearson_corr,
        'pearson_p': pearson_p
    })
    print(f"{genre}: Spearman ρ = {spearman_corr:.2f} (p={spearman_p:.2f}), Pearson r = {pearson_corr:.2f} (p={pearson_p:.2f})")

# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv("../data/processed/embeddings/hiddenstates/genrewise_projection_results.csv", index=False)
print("Saved genrewise projection correlation results.")
# %%

#%% 
# --- Import packages ---
# Import necessary libraries
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import custom modules
from core.loader import CorpusLoader
from core.embedder import Embedder
from core.projection import ProjectionAnalyzer


# %%
# --- Load Classes, Split Data and Investigate the Data ---
# --- Initialize Embedder Class:
MultiLingMPNET = Embedder(model_name="paraphrase-multilingual-mpnet-base-v2")

# --- Initialize Loader Class:
# Load the fiction4 dataset from HuggingFace -- Training set (positive/negative filter)
loader = CorpusLoader(text_col="text", label_col="label")
loader.load_from_huggingface("chcaa/fiction4sentiment", split="train")
loader.split_binary_train_continuous_test(positive_threshold=7, negative_threshold=3, train_size=0.6, random_state=42) # Define the thresholds for positive and negative labels, and split the non-neutral elements into training and test sets.

# Print the distribution of labels in the training set
print(loader.df.shape)
print(loader.train_df.shape)
print(loader.train_df['label'].value_counts())
print(loader.test_df.shape)
#%%
# --- Load the EmoBank Corpus ---
loaderEmoBank = CorpusLoader(path=("../../EmoBank/corpus/emobank.csv"),text_col="text", label_col="V")
loaderEmoBank.load_csv()
loaderEmoBank.df.head()

# %%
# --- Embed model ---
train_MPNET = MultiLingMPNET.embed(loader.train_texts, cache_path="../data/processed/embeddings/fiction4_train_MultiLingMPNET_neg3_pos7.csv")
train_MPNET["label"] = loader.train_labels

test_MPNET = MultiLingMPNET.embed(loader.test_texts, cache_path="../data/processed/embeddings/fiction4_test_MultiLingMPNET_hug.csv")
test_MPNET["label"] = loader.test_labels

#%% 
EmoBank_MPNET = MultiLingMPNET.embed(loaderEmoBank.df['sentence'], cache_path="../data/processed/embeddings/EmoBank_MultiLingMPNET.csv")
EmoBank_MPNET["valence"] = loaderEmoBank.df['label']


# %%
# Define masks for english and danish for training set
english_mask_train = loader.train_df['category'].isin(['poetry', 'prose'])
danish_mask_train = ~loader.train_df['category'].isin(['poetry', 'prose'])
HEMMING_train = loader.train_df['category'].isin(['prose'])
SYLVIA_train = loader.train_df['category'].isin(['poetry'])
HCA_train = loader.train_df['category'].isin(['fairytales'])
HYMN_train = loader.train_df['category'].isin(['hymns'])

# Define masks for english and danish for test set
english_mask_test = loader.test_df['category'].isin(['poetry', 'prose'])
danish_mask_test = ~loader.test_df['category'].isin(['poetry', 'prose'])
HEMMING_test = loader.test_df['category'].isin(['prose'])
SYLVIA__test = loader.test_df['category'].isin(['poetry'])
HCA_test = loader.test_df['category'].isin(['fairytales'])
HYMN_test = loader.test_df['category'].isin(['hymns'])


# Define high level train/test pairs
train_test_pairs_danish = [
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
train_english_test_danish = [
    (HEMMING_train, HCA_test, "Trained on Prose → Tested on Fairytales"),
    (HEMMING_train, HYMN_test, "Trained on Prose → Tested on Hymns"),
    (SYLVIA_train, HCA_test, "Trained on Poetry → Tested on Fairytales"),
    (SYLVIA_train, HYMN_test, "Trained on Poetry → Tested on Hymns")
]
train_danish_test_english = [
    (HCA_train, HEMMING_test, "Trained on Fairytales → Tested on Prose"),
    (HCA_train, SYLVIA__test, "Trained on Fairytales → Tested on Poetry"),
    (HYMN_train, HEMMING_test, "Trained on Hymns → Tested on Prose"),
    (HYMN_train, SYLVIA__test, "Trained on Hymns → Tested on Poetry")
]


#%%
### Single projection example ###
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import spearmanr
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

### Define the train/test masks ###
train_mask = english_mask_train
test_mask = english_mask_test

# === 1. Project embeddings ===
analyzer = ProjectionAnalyzer(matrix_concept=train_MPNET, matrix_project=test_MPNET[english_mask_test])
analyzer.project()

# === 2. Make a plot ===
x_raw = analyzer.projected_in_1D
y = test_MPNET[english_mask_test]['label'].values


# === 3. Rescale projections to [0.2, 9.2] ===
scaler = MinMaxScaler((1, 9))
scaled = scaler.fit_transform(x_raw.values.reshape(-1, 1)).flatten()

# === 4. Calculate Spearman correlation ===
corr, _ = spearmanr(scaled, y)

# === 5. Create the plot ===
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



## Also save a dataframe for sampling later
Projection_For_Sampling = pd.DataFrame({
    'Prediction': scaled,
    'Sentence': np.array(loader.test_texts)[english_mask_test]
})

# %%
# Plotting function for projection grid with marginals - 4x4 setup
import io
from PIL import Image

def plot_four_jointplots_together(train_test_pairs, train_MPNET, test_MPNET):
    jointplot_images = []

    for train_mask, test_mask, title in train_test_pairs:
        analyzer = ProjectionAnalyzer(matrix_concept=train_MPNET[train_mask],
                                      matrix_project=test_MPNET[test_mask])
        analyzer.project()

        x_raw = analyzer.projected_in_1D
        y = test_MPNET[test_mask]['label'].values
        scaler = MinMaxScaler((1, 9))
        scaled = scaler.fit_transform(x_raw.values.reshape(-1, 1)).flatten()
        corr, _ = spearmanr(scaled, y)

        # Create jointplot
        g = sns.jointplot(x=scaled, y=y, kind="scatter", marginal_kws=dict(fill=True),
                          alpha=0.5, height=5)
        g.ax_joint.set_xlim(0, 10)
        g.ax_joint.set_xticks(np.linspace(1, 9, 9))
        g.ax_joint.set(xlabel='Fiction4 Subspace Projection', ylabel='Human Gold Standard')
        g.figure.subplots_adjust(top=0.9)
        g.ax_joint.set_title(f"{title}\nSpearman ρ = {corr:.2f}", pad=60)

        # Save to in-memory buffer
        buf = io.BytesIO()
        g.figure.savefig(buf, format='png', dpi=150, bbox_inches='tight', pad_inches=0.1)
        buf.seek(0)
        img = Image.open(buf)
        jointplot_images.append(img)
        plt.close(g.figure)  # Prevents overlapping or duplicate figures

    # Create final grid image
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    axs = axs.flatten()

    for ax, img in zip(axs, jointplot_images):
        ax.imshow(img)
        ax.axis('off')  # Hide axis around image

    plt.subplots_adjust(wspace=0, hspace=0.1)
    plt.show()


#plot_four_jointplots_together(train_test_pairs_english, train_MPNET, test_MPNET)
#plot_four_jointplots_together(train_test_pairs_danish, train_MPNET, test_MPNET)
plot_four_jointplots_together(train_danish_test_english, train_MPNET, test_MPNET)
plot_four_jointplots_together(train_test_pairs_english_danish, train_MPNET, test_MPNET)

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
########################################
# Floored Binning
########################################

# Your data
predicted_raw = analyzer.projected_in_1D
scaler = MinMaxScaler((1, 9))
predicted = scaler.fit_transform(predicted_raw.values.reshape(-1, 1)).flatten()
projection_distance = analyzer.projection_distance

# Bin by integer (floor to get bins 1–8)
predicted_bins = np.floor(predicted).astype(int)

# Convert to ordered categorical labels
bin_order = [str(i) for i in range(1, 9)]
predicted_bins = pd.Categorical(predicted_bins.astype(str), categories=bin_order, ordered=True)

# Combine into DataFrame
df = pd.DataFrame({
    'PredictedBin': predicted_bins,
    'ProjectionDistance': projection_distance
})

# Count elements per bin
bin_counts = df['PredictedBin'].value_counts().sort_index()
bin_labels_with_counts = [f"{label} (n={bin_counts[label]})" for label in df['PredictedBin'].cat.categories]

# Plot
plt.figure(figsize=(12, 6))
ax = sns.violinplot(data=df, x='PredictedBin', y='ProjectionDistance', inner='box', palette='pastel')
plt.title("Projection Distance Distribution for Each Predicted Value (Binned to Integers)")
plt.ylabel("Projection Distance")
plt.xlabel("Predicted Value (Floored to Integer)")
ax.set_xticklabels(bin_labels_with_counts)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# --- Preidction error binned across predicted values
# Your data
predicted_raw = analyzer.projected_in_1D
scaler = MinMaxScaler((1, 9))
predicted = scaler.fit_transform(predicted_raw.values.reshape(-1, 1)).flatten()
true = test_MPNET['label']
error = predicted - true

# Floor predicted values to bin them
floored = np.floor(predicted).astype(int)

# Define bin order (1 to 8)
bin_order = [str(i) for i in range(1, 9)]
floored_categorical = pd.Categorical(floored.astype(str), categories=bin_order, ordered=True)

# Count elements per bin
bin_counts = pd.Series(floored_categorical).value_counts().sort_index()
bin_labels_with_counts = [f"{label} (n={bin_counts[label]})" for label in bin_order]

# Create DataFrame for plotting
df = pd.DataFrame({
    'PredictedBin': floored_categorical,
    'PredictionError': error
})

# Plot
plt.figure(figsize=(12, 6))
ax = sns.violinplot(data=df, x='PredictedBin', y='PredictionError', inner='box', palette='pastel')
plt.axhline(0, color='gray', linestyle='--', linewidth=1.2)  # Add horizontal line at y=0
plt.title("Prediction Error Distribution by Floored Predicted Value")
plt.ylabel("Prediction Error (Predicted − True)")
plt.xlabel("Predicted Value (Floored to Integer)")
ax.set_xticklabels(bin_labels_with_counts)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# Use true values (floored) as bins instead of predicted values
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Your data
predicted_raw = analyzer.projected_in_1D
scaler = MinMaxScaler((1, 9))
predicted = scaler.fit_transform(predicted_raw.values.reshape(-1, 1)).flatten()
true = test_MPNET['label']
error = predicted - true

# Floor true values to bin them
floored_true = np.floor(true).astype(int)

# Define bin order (1 to 8)
bin_order = [str(i) for i in range(1, 9)]
floored_categorical = pd.Categorical(floored_true.astype(str), categories=bin_order, ordered=True)

# Count elements per bin
bin_counts = pd.Series(floored_categorical).value_counts().sort_index()
bin_labels_with_counts = [f"{label} (n={bin_counts[label]})" for label in bin_order]

# Create DataFrame for plotting
df = pd.DataFrame({
    'TrueBin': floored_categorical,
    'PredictionError': error
})

# Plot
plt.figure(figsize=(12, 6))
ax = sns.violinplot(data=df, x='TrueBin', y='PredictionError', hue='TrueBin', palette='pastel', legend=False)
plt.axhline(0, color='gray', linestyle='--', linewidth=1.2)  # Add horizontal line at y=0
plt.title("Prediction Error Distribution by Floored True Value")
plt.ylabel("Prediction Error (Predicted − True)")
plt.xlabel("True Value (Floored to Integer)")
ax.set_xticklabels(bin_labels_with_counts)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# --- KDE of error in prediction

x_raw = analyzer.projected_in_1D
scaler = MinMaxScaler((1, 9))
scaled = scaler.fit_transform(x_raw.values.reshape(-1, 1)).flatten()
y = scaled-test_MPNET['label'].values
scaled = (test_MPNET['label'].values)
corr, _ = spearmanr(scaled, y)
g = sns.jointplot(x=scaled, y=y, kind="kde", marginal_kws=dict(fill=True),
                  alpha=0.5, height=7)
plt.axhline(0, color='gray', linestyle='--', linewidth=1.2)
g.ax_joint.set_xlim(0, 10)
g.ax_joint.set(xlabel='Position in 1D', ylabel='Error in Prediction (Predicted − True)')
g.ax_joint.set_title(f'KDE of Prediction Error (MPNET Rescaled)', pad=90)
g.ax_joint.text(0.05, 0.95, f'Spearman ρ = {corr:.2f}', transform=g.ax_joint.transAxes,
                fontsize=12, color='blue')
g.figure.subplots_adjust(top=0.9)
plt.show()


# --- Error in prediction regression ---
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import spearmanr
import numpy as np

# Assuming x_raw and test_MPNET['label'] are defined
x_raw = analyzer.projected_in_1D
scaler = MinMaxScaler((1, 9))
scaled = scaler.fit_transform(x_raw.values.reshape(-1, 1)).flatten()
y = scaled - test_MPNET['label'].values
scaled = test_MPNET['label'].values

# Calculate Spearman correlation
corr, _ = spearmanr(scaled, y)

# Create scatterplot with regression line using sns.jointplot
with sns.axes_style('white'):
    g = sns.jointplot(x=scaled, y=y, kind="reg", marginal_kws=dict(fill=True),
                      scatter_kws={'alpha': 0.5, 's': 10}, height=7, line_kws={"color": "red"})

# Add horizontal reference line at y=0
plt.axhline(0, color='gray', linestyle='--', linewidth=1.2)

# Customize plot
g.ax_joint.set_xlim(0, 10)
g.ax_joint.set(xlabel='Golden Standard Label', ylabel='Error in Prediction (Predicted − True)')
g.ax_joint.set_title(f'Scatterplot with Regression Line (MPNET Rescaled)', pad=90)
g.ax_joint.text(0.05, 0.95, f'Spearman ρ = {corr:.2f}', transform=g.ax_joint.transAxes,
                fontsize=12, color='blue')
g.figure.subplots_adjust(top=0.9)
plt.show()




# %%
### Single projection example - test fiction4###
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import spearmanr
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# === 1. Project embeddings ===
analyzer = ProjectionAnalyzer(matrix_concept=train_MPNET, matrix_project=test_MPNET)
analyzer.project()

# === 2. Make a plot ===
x_raw = analyzer.projected_in_1D
y = test_MPNET['label'].values

# === 3. Rescale projections to [0.2, 9.2] ===
scaler = MinMaxScaler((1, 9))
scaled = scaler.fit_transform(x_raw.values.reshape(-1, 1)).flatten()

# === 4. Calculate Spearman correlation ===
corr, _ = spearmanr(scaled, y)

# === 5. Create the plot ===
g = sns.jointplot(x=scaled, y=y, kind="scatter", marginal_kws=dict(fill=True),
                  alpha=0.5, height=7)
g.ax_joint.set_xlim(0, 10)
g.ax_joint.set_xticks(np.linspace(1, 9, 9))  # Custom ticks from 1 to 10
g.ax_joint.set(xlabel='Fiction4 Subspace Projection', ylabel='Human Gold Standard')
g.ax_joint.set_title(f'Trained on fiction4 - Tested on fiction4', pad=90)
g.ax_joint.text(0.05, 0.95, f'Spearman ρ = {corr:.2f}', transform=g.ax_joint.transAxes,
                fontsize=12, color='blue')
g.figure.subplots_adjust(top=0.9)
plt.show()

### Single projection example - test emobank ###
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import spearmanr
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
g.ax_joint.set_title(f'Trained on fiction4 - Tested on EmoBank', pad=90)
g.ax_joint.text(0.05, 0.95, f'Spearman ρ = {corr:.2f}', transform=g.ax_joint.transAxes,
                fontsize=12, color='blue')
g.figure.subplots_adjust(top=0.9)
plt.show()




# %%
df_combined = pd.DataFrame({
    'projection': scaled,
    'golden_standard':EmoBank_MPNET['valence'].values,
    'original_sentence': loaderEmoBank.df['sentence']
    })
df_combined
# %%
# --- SAMPLE DATA FOR EXPERIMENT:
import pandas as pd
import numpy as np

# Ensure 'valence' is a float
test_MPNET['valence'] = test_MPNET['valence'].astype(float)
df_sorted = test_MPNET.sort_values(by='valence').reset_index(drop=True)

# --- Step 1: Sample 50 indexes using threshold logic ---
samples = []
sampled_indices = []

while len(samples) < 50:
    threshold = np.random.uniform(2.5, 3.5)
    match = df_sorted[df_sorted['valence'] > threshold].head(1)
    if not match.empty:
        idx = match.index[0]
        if idx not in sampled_indices:
            samples.append(match[['valence', 'sentence']].iloc[0])
            sampled_indices.append(idx)

# --- Step 2: For each sampled index, run the pair sampling ---
valences = df_sorted['valence'].values
sentences = df_sorted['sentence'].values
n_pairs = 5

paired_sentences = []

for idx in sampled_indices:
    current_valence = valences[idx]
    current_sentence = sentences[idx]
    mask = np.arange(len(valences)) != idx
    other_valences = valences[mask]
    other_sentences = sentences[mask]
    chosen_indices = set()
    pairs_found = 0
    # Keep sampling until 5 pairs are found
    while pairs_found < 5:
        target = np.random.normal(loc=current_valence, scale=0.5)
        diffs = np.abs(other_valences - target)
        for i in np.argsort(diffs):
            # Only accept if valence is within [1.5, 4.5] and not already chosen
            if i not in chosen_indices and 1.5 <= other_valences[i] <= 4.5:
                chosen_indices.add(i)
                paired_sentences.append({
                    'original_sentence': current_sentence,
                    'original_valence': current_valence,
                    'paired_sentence': other_sentences[i],
                    'paired_valence': other_valences[i]
                })
                pairs_found += 1
                break  # Found a suitable pair, move to next

paired_df = pd.DataFrame(paired_sentences)
paired_df

# %%
import json

# Build the list of pairs
pairs = []
for _, row in paired_df.iterrows():
    pairs.append({
        "sentences": [
            row["original_sentence"],
            row["paired_sentence"]
        ]
    })

# Save to JSON
output = {"pairs": pairs}
with open("paired_sentences.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=4)


# %%
import pandas as pd
import numpy as np

def sample_and_pair_categories_unique(df, pred_col='Prediction', sent_col='Sentence', n_samples=50, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    # Define categories
    low_df = df[df[pred_col] <= 3].reset_index(drop=True)
    med_df = df[(df[pred_col] >= 4) & (df[pred_col] <= 6)].reset_index(drop=True)
    high_df = df[df[pred_col] >= 7].reset_index(drop=True)

    # Print unique counts for debugging
    print("Unique low sentences:", low_df[sent_col].nunique())
    print("Unique med sentences:", med_df[sent_col].nunique())
    print("Unique high sentences:", high_df[sent_col].nunique())

    # Warn if not enough unique sentences to guarantee unique pairs
    if low_df[sent_col].nunique() < n_samples + 1:
        print("Warning: Not enough unique low sentences to guarantee unique pairs.")
    if med_df[sent_col].nunique() < n_samples + 1:
        print("Warning: Not enough unique medium sentences to guarantee unique pairs.")
    if high_df[sent_col].nunique() < n_samples + 1:
        print("Warning: Not enough unique high sentences to guarantee unique pairs.")

    # Sample n_samples from each category (with replacement if not enough)
    low_samples = low_df.sample(n=n_samples, replace=len(low_df) < n_samples, random_state=random_state)
    med_samples = med_df.sample(n=n_samples, replace=len(med_df) < n_samples, random_state=random_state)
    high_samples = high_df.sample(n=n_samples, replace=len(high_df) < n_samples, random_state=random_state)

    # Track used sentences for each category
    used_low = set()
    used_med = set()
    used_high = set()

    # Helper to get prediction for a sentence
    pred_lookup = df.set_index(sent_col)[pred_col].to_dict()

    def get_unique_pair(row, pool_df, used_set):
        pool = pool_df[(pool_df[sent_col] != row[sent_col]) & (~pool_df[sent_col].isin(used_set))]
        if pool.empty:
            pool = pool_df[pool_df[sent_col] != row[sent_col]]
        chosen = pool.sample(1, random_state=random_state)[sent_col].values[0]
        used_set.add(chosen)
        return chosen

    pairs = []
    for _, row in low_samples.iterrows():
        pair_low = get_unique_pair(row, low_df, used_low)
        pair_med = get_unique_pair(row, med_df, used_med)
        pair_high = get_unique_pair(row, high_df, used_high)
        pairs.append({
            "original_sentence": row[sent_col],
            "original_category": "low",
            "original_prediction": row[pred_col],
            "pair_low": pair_low,
            "pair_low_prediction": pred_lookup.get(pair_low, np.nan),
            "pair_med": pair_med,
            "pair_med_prediction": pred_lookup.get(pair_med, np.nan),
            "pair_high": pair_high,
            "pair_high_prediction": pred_lookup.get(pair_high, np.nan)
        })
    for _, row in med_samples.iterrows():
        pair_low = get_unique_pair(row, low_df, used_low)
        pair_med = get_unique_pair(row, med_df, used_med)
        pair_high = get_unique_pair(row, high_df, used_high)
        pairs.append({
            "original_sentence": row[sent_col],
            "original_category": "medium",
            "original_prediction": row[pred_col],
            "pair_low": pair_low,
            "pair_low_prediction": pred_lookup.get(pair_low, np.nan),
            "pair_med": pair_med,
            "pair_med_prediction": pred_lookup.get(pair_med, np.nan),
            "pair_high": pair_high,
            "pair_high_prediction": pred_lookup.get(pair_high, np.nan)
        })
    for _, row in high_samples.iterrows():
        pair_low = get_unique_pair(row, low_df, used_low)
        pair_med = get_unique_pair(row, med_df, used_med)
        pair_high = get_unique_pair(row, high_df, used_high)
        pairs.append({
            "original_sentence": row[sent_col],
            "original_category": "high",
            "original_prediction": row[pred_col],
            "pair_low": pair_low,
            "pair_low_prediction": pred_lookup.get(pair_low, np.nan),
            "pair_med": pair_med,
            "pair_med_prediction": pred_lookup.get(pair_med, np.nan),
            "pair_high": pair_high,
            "pair_high_prediction": pred_lookup.get(pair_high, np.nan)
        })

    return pd.DataFrame(pairs)

# Usage:
result_df = sample_and_pair_categories_unique(Projection_For_Sampling, pred_col='Prediction', sent_col='Sentence', n_samples=30, random_state=42)
result_df
result_df.to_csv("../data/processed/csv/sentence_samples.csv", index=False, encoding="utf-8")

# %%
import json
# Filter for medium original sentences
medium_df = result_df[result_df["original_category"] == "medium"]

# Build the list of pairs for each type
pairs = []
for _, row in medium_df.iterrows():
    pairs.append({
        "sentences": [
            row["original_sentence"],
            row["pair_low"]
        ]
    })
    pairs.append({
        "sentences": [
            row["original_sentence"],
            row["pair_med"]
        ]
    })
    pairs.append({
        "sentences": [
            row["original_sentence"],
            row["pair_high"]
        ]
    })

# Save to JSON
output = {"pairs": pairs}
with open("paired_sentences_medium.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=4)
# %%

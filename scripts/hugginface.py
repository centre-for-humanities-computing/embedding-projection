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
    (HEMMING_train, HYMN_test, "Trained on Prose → TTested on Hymns"),
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
analyzer = ProjectionAnalyzer(matrix_concept=train_MPNET[danish_mask_test], matrix_project=test_MPNET[english_mask_test])
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

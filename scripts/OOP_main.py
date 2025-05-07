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
ProjectionAnalyzer = ProjectionAnalyzer(matrix_concept=train_mini, matrix_project=test_mini)
ProjectionAnalyzer.project()


# %%
from core.projection import ProjectionAnalyzer
ProjectionAnalyzer = ProjectionAnalyzer(matrix_concept=train_mini, matrix_project=test_mini)
ProjectionAnalyzer.project()

from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
# Assuming both are lists or 1D numpy arrays of length 6300
x = ProjectionAnalyzer.projected_in_1D
y = test_mini["label"].values


# Plotting
plt.scatter(x, y, alpha=0.5)
plt.xlabel('Fiction4 defined Subspace Projection')
plt.ylabel('Human Golden Standard')
plt.title('Scatterplot with Correlation (MiniLM)')

# Compute correlation
corr, _ = pearsonr(x, y)
plt.figtext(0.15, 0.85, f'Pearson r = {corr:.2f}', fontsize=12, color='blue')
# Save plot in img folder:
plt.savefig('../img/Scatterplot_fiction4_w_Person_MiniLM.png', dpi=300, bbox_inches='tight')
plt.show()

# %%
from core.projection import ProjectionAnalyzer
ProjectionAnalyzer = ProjectionAnalyzer(matrix_concept=train_MPNET, matrix_project=test_MPNET)
ProjectionAnalyzer.project()

from scipy.stats import spearmanr
import seaborn as sns
import matplotlib.pyplot as plt
# Assuming both are lists or 1D numpy arrays of length 6300
x = ProjectionAnalyzer.projected_in_1D
y = test_MPNET["label"].values

# Plotting
plt.scatter(x, y, alpha=0.5)
plt.xlabel('Fiction4 defined Subspace Projection')
plt.ylabel('Human Golden Standard')
plt.title('Scatterplot with Correlation (MPNET)')

# Compute correlation
corr, _ = pearsonr(x, y)
plt.figtext(0.15, 0.85, f'Spearman ρ = {corr:.2f}', fontsize=12, color='blue')
# Save plot in img folder:
plt.savefig('../img/Scatterplot_fiction4_w_Person.png', dpi=300, bbox_inches='tight')
plt.show()
# %%

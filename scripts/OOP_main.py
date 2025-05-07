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

# %%
from core.projection import ProjectionAnalyzer
ProjectionAnalyzer = ProjectionAnalyzer(matrix_concept=train_embed, matrix_project=test_embed)
ProjectionAnalyzer.project()

# %%

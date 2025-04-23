# %%
from utils import *
from functions import *

# %%
# Load the dataset:
IMDb_df = pd.read_csv('../data/raw/imdb.csv')
print(IMDb_df['review'] [1:5])
from functions import clean_string
# Slice the DataFrame first:
subset_df  = IMDb_df.iloc[1:2000].reset_index(drop=True)

# Clean the reviews:
subset_df['review'] = subset_df ['review'].apply(clean_string)
subset_corpus= subset_df['review'].tolist()


# %%
# Load the SentenceTransformer model:
MPNET_Model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device="cuda" if torch.cuda.is_available() else "cpu")
# Encode reviews:
MPNET_Embedding = MPNET_Model.encode(subset_corpus, show_progress_bar=True, device="cuda" if torch.cuda.is_available() else "cpu")

# %%
# Save embeddings as a CSV:
embedding = pd.DataFrame(MPNET_Embedding)
embedding['sentiment'] = subset_df['sentiment']
embedding['review'] = subset_df['review']
embedding.to_csv('../data/embeddings/MPNET_Embeddings.csv', index=False)


# %%
# Load the embeddings from the file:
embedding = pd.read_csv('../data/embeddings/MPNET_Embeddings.csv')
# Check the first 5 rows and columns of the dataframe:
embedding.iloc[-5:-1] # Check the first 5 rows of the dataframe



# %%
# Filter pandas dataframe for positive and negative sentiment:
positive = embedding[embedding['sentiment'] == "positive"]
negative = embedding[embedding['sentiment'] == "negative"]
# Investigate the shape of the dataframes:
print(positive.shape)
print(negative.shape)

# %%
from functions import positive_to_negative_vector, express_matrix_by_vector

# Define the sentiment vector by subtracting the mean of the negative from the mean of the positive:
sentiment_vector = positive_to_negative_vector(positive.iloc[:, :-2], negative.iloc[:, :-2])
projection, projection_in_1D_subspace = express_matrix_by_vector(embedding.iloc[:, :-2], sentiment_vector)

# Check mean projection_in_1D_subspace for positive and negative sentiment:
print(projection_in_1D_subspace[embedding['sentiment'] == "positive"].mean())
print(projection_in_1D_subspace[embedding['sentiment'] == "negative"].mean())


# %%
# Plot histogram of the projection_in_1D_subspace, colored by sentiment:
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# Plot histogram of projections colored by sentiment
plt.figure(figsize=(10, 6))
sns.histplot(data=embedding, x=projection_in_1D_subspace, hue='sentiment', element='step', stat='density', common_norm=False, palette='coolwarm', bins=30)
plt.title('Distribution of Reviews Along the Sentiment Vector')
plt.xlabel('Projection onto Sentiment Vector')
plt.ylabel('Density')
plt.grid(True)
plt.show()

# %%
# Plot KDE of projections colored by sentiment
plt.figure(figsize=(10, 6))
sns.kdeplot(
    x=projection_in_1D_subspace[embedding['sentiment'] == 'positive'],
    label='Positive',
    fill=True,
    alpha=0.4,
    linewidth=2,
    color='blue'
)
sns.kdeplot(
    x=projection_in_1D_subspace[embedding['sentiment'] == 'negative'],
    label='Negative',
    fill=True,
    alpha=0.4,
    linewidth=2,
    color='red'
)
plt.title('Sentiment Distributions Of Reviews Along Projection Vector')
plt.xlabel('Projection onto Sentiment Vector')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.tight_layout()
# Save plot in img folder:
plt.savefig('../img/sentiment_distributions.png', dpi=300, bbox_inches='tight')
# Show plot:
plt.show()


# %%
from functions import get_clean_words
# Get the most common words in the positive and negative reviews:
mapped_set = map(get_clean_words, embedding['review'])
my_set = set().union(*mapped_set)

# Convert set to list
words = list(my_set)
# Initialize the tokenizer, good basic tokenizer
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')


# Tokenize each word in the list
tokenized_words = [tokenizer.tokenize(word) for word in words]
# Flatten the tokenized words list
flat_tokens = [token for sublist in tokenized_words for token in sublist]

# Save each unique element:
review_tokens_corpus =  list(set(flat_tokens))


# %%
review_tokens_corpus

# %%
# Encode the words:
word_embeddings = MPNET_Model.encode(review_tokens_corpus, show_progress_bar=True, device="cuda" if torch.cuda.is_available() else "cpu")
# %% 
word_embeddings=pd.DataFrame(word_embeddings)
word_embeddings.iloc[0:5] # Check the first 5 rows and columns of the word embeddings

# %%
projected_variance, projection_in_1D_subspace = express_matrix_by_vector(matrix=word_embeddings, vector=sentiment_vector)

# Find the 10 smallest values' indices
smallest_indices = projection_in_1D_subspace.nsmallest(10).index
# Map indices to words in the corpus
smallest_words = [review_tokens_corpus[i] for i in smallest_indices]

print("Words corresponding to the 10 smallest embeddings:")
print(smallest_words)

# find the 10 largest values' indices
largest_indices = projection_in_1D_subspace.nlargest(10).index
# Map indices to words in the corpus
largest_words = [review_tokens_corpus[i] for i in largest_indices]
print("Words corresponding to the 10 largest embeddings:")
print(largest_words)


# %%

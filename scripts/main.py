# %%
from utils import *
from functions import *

# %%
# Load the dataset:
IMDb_df = pd.read_csv('../data/raw/imdb.csv')
print(IMDb_df['review'] [1:5])
from functions import clean_string
IMDb_df['review'] = IMDb_df['review'].apply(clean_string)
subset_corpus= IMDb_df['review'].tolist()


# %%
# Load best basic model for sentence embedding:
MPNET_Model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device="cuda" if torch.cuda.is_available() else "cpu")
# Encode the first 500 reviews (for proof of concept):
MPNET_Embedding = MPNET_Model.encode(subset_corpus[1:2000], show_progress_bar=True, device="cuda" if torch.cuda.is_available() else "cpu")

# %%
# Save embeddings to a file:
embedding = pd.DataFrame(MPNET_Embedding) # Pandas dataframe of embeddings
embedding['sentiment'] = IMDb_df['sentiment'][1:2000]  # Add sentiment column to the dataframe
embedding['review'] = subset_corpus[1:2000]  # Add review column to the dataframe
embedding.to_csv(f'../data/embeddings/MPNET_Embeddings.csv', index = False)

# %%
# Filter pandas dataframe for positive and negative sentiment:
positive = embedding[embedding['sentiment'] == "positive"]
negative = embedding[embedding['sentiment'] == "negative"]
# Investigate the shape of the dataframes:
print(positive.shape)
print(negative.shape)

# %%
# from functions import positive_to_negative_vector, express_matrix_by_vector
def positive_to_negative_vector(Positive, Negative):
    """
    Takes a positive and an negative data point and defines the vector spanning both vectors.
    """
    posneg_vector = Positive.mean().to_frame().T-Negative.mean().to_frame().T
    posneg_vector = pd.DataFrame(posneg_vector)
    return posneg_vector

def project_matrix_to_vector(matrix, vector):
    """Compute the projection of a matrix onto the space spanned by the vector
    Args:
        vector: ndarray of dimension (D, 1), the vector spanning D dimensions that you want to project upon.
        matrix: ndarray of dimension (D, M), the matrix consisting of M vectors that you want to map to the subspace spanned by the vector.
    
    Returns:
        p: projection of matrix onto the subspac spanned by the columns of vector; size (D, 1)
    """
    m = matrix.to_numpy() # Turn into a matrix
    v = vector.to_numpy()[0] #Turn into a numpy array

    # Compute v dot v (denominator)
    v_dot_v = np.dot(v, v)

    # Compute projection of each row of m onto v
    projection = np.outer(np.dot(m, v) / v_dot_v, v)
    projection = pd.DataFrame(projection)

    return projection


def express_matrix_by_vector(matrix, vector):
    """Compute the projection of a matrix onto the space spanned by the vector
    Args:
        vector: ndarray of dimension (D, 1), the vector spanning D dimensions that you want to project upon.
        matrix: ndarray of dimension (D, M), the matrix consisting of M vectors that you want to map to the subspace spanned by the vector.
    
    Returns:
        projection: projection of matrix onto the subspac spanned by the columns of vector; size (D, 1)
        projection_in_1D_subspace: Each embedding projected onto 1 dimensional subspace spanned by input vector.
    """
    unit_vector = vector / np.linalg.norm(vector) # Find the unit vector for interpretatbility by dividing with its norm
    projection = project_matrix_to_vector(matrix, vector) # Find projections, so we can find lengths by finding relations in first dimension
    projection_in_1D_subspace = projection.iloc[:,0]/unit_vector.iloc[:,0][0] # Location in subspace

    return projection, projection_in_1D_subspace


# %%
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
#from functions import get_clean_words
def get_clean_words(raw_string):
    """
    Takes in a raw_string and returns the set of words appearing in the string.
    """
    import re
    try:
        clean_text = str(raw_string)  # Convert to string
        clean_text = re.sub(r'<br/><br/>', ' ', clean_text)  # Remove breaks
        clean_text = re.sub(r'/', ' ', clean_text)  # Replace forward slashes with space
        clean_text = re.sub(r'[^a-zA-Z0-9 ]', '', clean_text).lower()  # Remove special characters and lowercase
        words = clean_text.split()  # Split the text into words
        unique_words = set(words)  # Get unique words
        return unique_words
    except Exception as e:
        print(f"Error processing input: {raw_string}. Error: {e}")
        return set()  # Return an empty set in case of failure

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

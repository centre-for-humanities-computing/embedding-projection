<h1 align="center">Continuous sentiment scores for literary and multilingual contexts</h1>
<p align="center">
    <a href="https://arxiv.org/abs/2508.14620">
        <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2508.14620-b31b1b.svg">
    </a>
    <a href="https://huggingface.co/datasets/chcaa/fiction4sentiment">
        <img alt="arXiv" src=https://img.shields.io/badge/Fiction4_Data-hugginface-yellow>
    </a>
    <a href="https://github.com/JULIELab/EmoBank/blob/master/corpus/emobank.csv">
        <img alt="Emobank" src="https://img.shields.io/badge/Emobank_Data-github-blue">
    </a>
</p>

This is a github repository for reproducing the results of
[Continuous sentiment scores for literary and multilingual contexts](https://arxiv.org/abs/2508.14620).

## 📋 Table of Contents
- [Overview](#-overview)
- [Installation](#️-installation)
- [Usage](#-usage)
  - [Reproducing Paper Results](#-reproducing-paper-results)
- [Sanity-Check of the Sentiment Vector](#-sanity-check-of-the-sentiment-vector)
  - [Distribution Analysis](#-distribution-analysis)
  - [Word-Level Analysis](#-word-level-analysis)


## 🔍 Overview
This repository *extracts sentiment from contextual embeddings* by using linear projection in embedding space. The repository contains the sentiment **datasets and the functions** used to embed text, define concept vectors, and project new data. 

Concretly, it is a project developing a technique for extracting information from contextual sentence embeddings ([model="paraphrase-multilingual-mpnet-base-v2"](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2)) by utilizing projection of embeddings onto a [concept vector](data/concept_vectors/vectors).


<p align="center">
    <img src="powerpoints/concept-vectors-for-sentiment.png" alt="Projection Pipeline" width="600"/>
</p>

The pipeline of the Semantic Projection algorithm is roughly visualised in the image below, and can be executed and validated by running the main.py script.
![Projection Pipeline](powerpoints/Projection_Pipeline.png)

If you are instead looking for a quick implementation of Semantic Projection, please check out the SemanticProjection package: 
https://github.com/lauritswl/SemanticProjection

## 🛠️ Installation
1. Clone the repository and navigate to it
```bash
git clone https://github.com/centre-for-humanities-computing/embedding-projection.git
cd embedding-projection
```

2. Create and activate virtual environment with uv
```bash
uv venv
source .venv/bin/activate
```

3. Install dependencies from pyproject.toml
```bash
python -m venv venv
source venv/bin/activate
pip install -e .
```
**Requirements:**
Dependencies needed to run main.py can be found in the [pyproject.toml](pyproject.toml). 

## 🚀 Usage
### 📊 Reproducing Paper Results
Run main.py to reproduce the plots and results used in the paper:

```bash
# Make sure your virtual environment is activated
source .venv/bin/activate

# Run the main script
python main.py

# The script will:
# 1. Load and preprocess the datasets
# 2. Create embeddings using MPNET
# 3. Generate concept vectors
# 4. Project test data onto concept vectors
# 5. Create plots in the plots/ directory
```

## 🧪 Sanity-Check of the Sentiment Vector
It seems that there is a rather strong correlation between average human anotator and the projection method!
This is seen in the scatterplot below, visualising the correlation between predictions and annotators for the EmoBank dataset (which is left out of training dataset):

![Human Annotator Correlation with Semantic Projection](plots/Scatterplot_Emobank_MultiLingMPNET_standardized.png)



### 📈 Distribution Analysis
The projection of binary-classified IMDB reviews onto our Sentiment Vector shows clear separation between positive and negative sentiments:

![Projection of Reviews onto Sentiment Vector](plots/IMDb_Sentiment_Distributions.png)

### 🔤 Word-Level Analysis
To validate our approach, we projected individual words from the corpus onto the Sentiment Vector. This method, inspired by [S3 - Semantic Signal Separation](https://arxiv.org/abs/2406.09556). The script for doing this is not included in the repo:


#### ⬆️ Highest Projection Score
```
pleasure    anytime     admired     admire      fabulous
classical   beloved     romantic    anthologies  lovely
```

#### ⬇️ Lowest Projection Score
```
worse       terrible    sucked      horrible    worst
bad         rotten      unacceptable stupidity   awful
```
*⚠️ Note: it seems that the vector might be correlated with the romantic literature period (H.C.Andersen), i.e. "anthologies, classical, romantic". This might be a byproduct of fairytales having a high density of positive semantics, thus being overrepresented in the training set.*

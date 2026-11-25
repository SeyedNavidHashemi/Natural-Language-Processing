# Word Embeddings and Semantic Analysis

## Overview
This project explores several fundamental concepts in Natural Language Processing (NLP), focusing on word representations, sentiment classification, and semantic similarity. The assignment covers the implementation of the Skip-Gram with Negative Sampling (SGNS) algorithm from scratch, training a Logistic Regression classifier for sentiment analysis, and evaluating word and sentence embeddings using pretrained GloVe vectors.

The main objective is to understand how distributed word representations capture semantic relationships, how simple machine learning models can be applied to text classification, and how different embedding-based approaches perform in measuring semantic similarity.

## Datasets
* **Brown Corpus (10K subset):** Used for building the vocabulary, generating target-context pairs, and training the SGNS model.
* **IMDB Reviews (2K):** Used for training and evaluating the Logistic Regression sentiment classifier.
* **GloVe 6B (50d):** Pretrained word embeddings used for comparison with SGNS and for word and sentence similarity evaluation.
* **WordSim-353:** Used to evaluate word similarity by comparing model predictions with human similarity judgments.
* **SimLex-999:** Used to evaluate semantic similarity between word pairs.
* **STS Benchmark:** Used to evaluate sentence-level semantic similarity against human annotations.

## Implemented Features

### 1. Word Embeddings and SGNS
* **SGNS from Scratch:** Implementation of the Skip-Gram with Negative Sampling algorithm using PyTorch tensor operations and mini-batch SGD.
* **Vocabulary Construction:** Tokenization, frequency-based vocabulary filtering, and generation of positive target-context pairs.
* **Negative Sampling:** Construction of the negative sampling distribution using word frequencies raised to the power of 0.75.
* **Gradient Verification:** Numerical gradient checking using the Finite Difference method to validate the manually implemented gradients.
* **Embedding Training:** Training 50-dimensional word embeddings and monitoring the Fixed-Subset Loss across epochs.
* **Nearest Neighbors:** Finding semantically related words using Cosine Similarity.
* **Analogy Evaluation:** Evaluating semantic and syntactic relationships using vector arithmetic and comparing SGNS with pretrained GloVe embeddings.
* **PCA Visualization:** Reducing embedding dimensions to two components and visualizing semantic relationships between selected words.

### 2. Sentiment Classification with Logistic Regression
* **Feature Engineering:** Extracting six handcrafted features from text, including positive and negative word counts, negation indicators, pronoun counts, exclamation marks, and log token count.
* **Logistic Regression from Scratch:** Implementing the model using the Sigmoid function, Binary Cross-Entropy Loss, and Stochastic Gradient Descent.
* **Model Evaluation:** Measuring classification performance using Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
* **Threshold Analysis:** Investigating the effect of different classification thresholds (0.5 and 0.8) on prediction performance.
* **Learning Rate Comparison:** Comparing a standard learning rate (0.01) with a significantly larger learning rate (15.0) to analyze convergence behavior and numerical stability.
* **Decision Boundary Visualization:** Visualizing the classification boundary using the first two input features.

### 3. Word Similarity Evaluation
* **Cosine Similarity:** Implementing cosine similarity to measure the semantic similarity between word vectors.
* **Nearest Neighbors:** Identifying the closest words in the embedding space.
* **Human Agreement Evaluation:** Comparing embedding-based similarity scores with human judgments using Spearman's Rank Correlation.
* **Benchmark Evaluation:** Evaluating pretrained GloVe embeddings on WordSim-353 and SimLex-999.

### 4. Sentence Similarity and Pooling Methods
* **Mean Pooling:** Constructing sentence embeddings by averaging the vectors of their constituent words.
* **Rank-Weighted Pooling:** Implementing a weighted aggregation method based on the logarithmic rank of words in the GloVe vocabulary.
* **STS Evaluation:** Comparing both pooling methods against human similarity scores using Spearman's Rank Correlation.
* **Error Analysis:** Examining selected sentence pairs to investigate the impact of out-of-vocabulary words and word weighting on similarity predictions.
* **Word Order Sensitivity:** Analyzing whether Mean and Rank-Weighted Pooling can distinguish between sentences containing the same words in different orders.

## Key Results

* **SGNS Training:** Successfully trained 50-dimensional embeddings on the Brown Corpus subset, containing 4,397 vocabulary words and 530,266 positive target-context pairs. The Fixed-Subset Loss decreased from 2.9571 to 1.9868 after 65 epochs.
* **Gradient Verification:** The numerical gradient check passed successfully, with relative errors of approximately 2.28 × 10⁻¹⁰ for the target embeddings and 6.41 × 10⁻¹⁰ for the context embeddings.
* **SGNS vs. GloVe:** Pretrained GloVe embeddings generally produced more meaningful nearest neighbors and better analogy results in the evaluated examples. For instance, in the analogy `man : he :: woman : ?`, GloVe ranked `she` first, while SGNS ranked it fourth.
* **PCA Analysis:** The first two principal components explained 43.28% of the variance in SGNS embeddings and 62.12% in GloVe embeddings.
* **Sentiment Classification:** The Logistic Regression model achieved 73.25% Accuracy and 73.45% F1-Score with a classification threshold of 0.5.
* **Threshold Trade-off:** Increasing the classification threshold to 0.8 raised Precision to 87.72%, but reduced Recall to 25%, demonstrating the trade-off between these two evaluation metrics.
* **Learning Rate Analysis:** A learning rate of 0.01 resulted in a final training loss of approximately 0.5379, while a learning rate of 15.0 produced highly unstable loss values, reaching approximately 15.8064.
* **Word Similarity:** GloVe achieved a Spearman correlation of 0.5247 on WordSim-353 (348 evaluated pairs) and 0.2652 on SimLex-999 (995 evaluated pairs).
* **Sentence Similarity:** Rank-Weighted Pooling achieved a Spearman correlation of 0.5310 on STS Benchmark, outperforming Mean Pooling, which achieved 0.4276. Both methods were evaluated on the same 1,379 sentence pairs.
* **Word Order Limitation:** Both pooling methods produced identical sentence vectors (Cosine Similarity = 1.0) for sentences with the same words in different orders, demonstrating their inability to capture word order and syntactic structure.

## Requirements
* Python 3.x
* NumPy
* Pandas
* Matplotlib
* PyTorch
* Scikit-learn
* SciPy
* Jupyter Notebook

## Usage
Run the Jupyter notebook sequentially from top to bottom.

The notebook expects the required datasets and pretrained GloVe embeddings to be available in the `datasets` directory. Make sure the dataset files are placed in their expected paths before running the notebook.

Random seeds are fixed to improve reproducibility of the experiments.

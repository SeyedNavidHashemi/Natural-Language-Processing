# BPE Tokenization and N-Gram Language Modeling

## Overview
This folder contains the implementation of a subword tokenizer using Byte Pair Encoding (BPE) and a comprehensive N-gram language model built from scratch. The project aims to address the challenges of out-of-vocabulary (OOV) words and data sparsity in language modeling through modern tokenization and statistical smoothing techniques.

## Datasets
* **WikiText-2:** Used for training and evaluating the Byte Pair Encoding (BPE) tokenizer.
* **Penn Treebank (PTB):** Used for training, validating, and testing the N-gram language models. The dataset is pre-tokenized and normalized.

## Implemented Features

### 1. Byte Pair Encoding (BPE) Tokenization
* Implementation of a frequency-based BPE tokenizer from scratch.
* Iterative merging of adjacent character pairs to build a subword vocabulary.
* Tokenization of unseen text using the trained BPE merge rules.

### 2. N-Gram Language Modeling
* **Maximum Likelihood Estimation (MLE):** Implementation of N-gram models (Unigram to 4-gram) using pure MLE.
* **Perplexity Evaluation:** Calculation of sentence and corpus-level perplexity to evaluate model performance on test data.
* **Out-Of-Vocabulary (OOV) Handling:** Dynamic replacement of unseen words with `<unk>` tokens based on the training vocabulary.

### 3. Smoothing and Backoff Techniques
To mitigate the data sparsity problem and the zero-probability issue in higher-order N-grams, the following techniques were implemented and compared on a Trigram model:
* **Laplace (Add-1) Smoothing:** A baseline additive smoothing technique.
* **Linear Interpolation:** A weighted combination of Unigram, Bigram, and Trigram probabilities. Hyperparameters (λ) were tuned on the validation set.
* **Stupid Backoff:** A heuristic approach that falls back to lower-order models with a discount factor (α) when higher-order N-grams are missing.

### 4. Text Generation
* Next-word prediction using the probability distribution of the best-performing model (Stupid Backoff).
* Implementation of **Temperature Scaling** to control the trade-off between greedy decoding (low temperature) and diverse, creative sampling (high temperature).

## Key Results
* **Data Sparsity Analysis:** Demonstrated the exponential drop in observed N-grams as N increases, proving the failure of pure MLE for 3-grams and 4-grams (resulting in infinite or extremely high perplexity).
* **Smoothing Effectiveness:** The **Stupid Backoff** method (with α = 1.2) significantly outperformed other models, reducing the test perplexity to approximately 79.35, proving its robustness in practical large-vocabulary scenarios.

## Requirements
* Python 3.x
* NumPy
* Pandas
* Matplotlib / Seaborn (for plotting perplexity comparisons)

## Usage
Run the Jupyter notebook sequentially. The code automatically downloads the required subsets of the Penn Treebank dataset directly from the source repository if they are not present in the local directory.

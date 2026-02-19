# Multi-Class Text Categorization using Machine Learning and Deep Sequential Models

## Project Overview
This project focuses on the automated categorization of Question and Answer (Q&A) text into 10 distinct topics. It explores the transition from traditional machine learning methods with statistical features to modern deep learning architectures using sequential modeling and pre-trained word embeddings.

## Key Features
- **Comprehensive Topic Categorization:** Classifies text into 10 classes including Politics, Sports, Health, Business, and more.
- **Feature Engineering Comparison:** Compares statistical methods (TF-IDF, Bag-of-Words) with advanced Word Embeddings (Word2Vec, GloVe).
- **Model Benchmarking:** Evaluates performance across multiple architectures:
  - **Machine Learning:** Logistic Regression, Multinomial Naive Bayes, Random Forest.
  - **Deep Learning:** SimpleRNN, LSTM, GRU, and Bidirectional LSTM.
- **CPU Optimization:** Implements optimized training configurations for efficient deep learning execution on CPU-based systems.

## Technology Stack
- **Language:** Python
- **Libraries:** TensorFlow, Keras, Scikit-learn, Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn, WordCloud
- **NLP:** NLTK, Gensim

## Project Structure
- `NLP-Based Multi-Class Text Categorization.ipynb`: The main notebook containing the data processing, model training, and evaluation logic.
- `train.csv` / `test.csv`: Dataset files (loaded from external source in the notebook).

## Results Summary
The project demonstrates that while traditional machine learning models provide a strong baseline, **Bidirectional LSTM** models with pre-trained **GloVe embeddings** generally achieve superior performance in capturing the semantic and sequential nuances of descriptive text.

## How to Run
1. Ensure you have the required libraries installed:
   ```bash
   pip install pandas numpy matplotlib seaborn tensorflow scikit-learn nltk gensim wordcloud
   ```
2. Open the Jupyter Notebook:
   ```bash
   jupyter notebook "NLP-Based Multi-Class Text Categorization.ipynb"
   ```
3. Run the cells sequentially to reproduce the data analysis, training, and evaluation.

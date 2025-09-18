# 🧠 TextCleaner & Classifier — NLP + Neural Network Sentiment Analyzer

A console-based Python application that classifies customer feedback as **positive** or **negative** using Natural Language Processing (NLP) and a simple neural network.

---

## 🎯 Project Goal

Demonstrate a complete machine learning pipeline for sentiment classification, including:

- Cleaning and normalizing raw text
- Converting text into numeric vectors using Bag of Words (BoW)
- Training a binary classifier using a basic feedforward neural network
- Evaluating and testing the model
- Accepting real-time text input via the command line

---

## 🧰 Tech Stack

- **Python 3.x**
- **NLTK** – for tokenization, stopword removal, and lemmatization
- **NumPy**, **Pandas** – data handling
- **Scikit-learn** – accuracy scoring, data splitting
- **TensorFlow / Keras** – neural network implementation

---

## 🧼 Preprocessing Pipeline

The following steps are applied to each input text:

1. Convert to lowercase  
2. Remove punctuation and digits  
3. Tokenize into words (`nltk.word_tokenize`)  
4. Remove English stopwords  
5. Lemmatize tokens using WordNet  

---

## 🧠 Model Architecture

A simple feedforward neural network built using Keras:

- **Input Layer:** Size of vocabulary from BoW
- **Hidden Layer:** `Dense(16, activation='relu')`
- **Output Layer:** `Dense(1, activation='sigmoid')`

**Training Settings:**
- Loss: `binary_crossentropy`
- Optimizer: `adam`
- Epochs: 20
- Batch Size: 4
- Metric: Accuracy




# 🧠 TextCleaner & Classifier — NLP + Neural Network Sentiment Analyzer

A console-based Python program that classifies customer feedback as **positive** or **negative** using Natural Language Processing (NLP) and a simple feedforward neural network. Developed as a graduation project for the Sprints.ai Bootcamp.

---

## 🎯 Project Objective

Simulate a real-world AI workflow that:
- Cleans and normalizes raw text using NLTK
- Converts text into numeric features using Bag of Words (BoW)
- Trains a binary classifier using a feedforward neural network (TensorFlow/Keras)
- Evaluates and predicts sentiment from new user input in real time via the terminal

---

## 🧰 Tech Stack

- **Python 3.x**
- **NLTK** – Tokenization, stopword removal, lemmatization
- **NumPy** & **Pandas** – Array and data handling
- **scikit-learn** – Train-test splitting, accuracy
- **TensorFlow / Keras** – Neural network model

---

## 🧼 Preprocessing Steps

Each feedback message goes through:

1. Lowercasing
2. Removing punctuation and digits
3. Tokenization (via `nltk.word_tokenize`)
4. Stopword removal (NLTK's stopword list)
5. Lemmatization (NLTK’s `WordNetLemmatizer`)

---

## 🧠 Model Architecture

A simple binary classifier with:

- **Input Layer:** Size = vocabulary length (from BoW)
- **Hidden Layer:** `Dense(16, activation='relu')`
- **Output Layer:** `Dense(1, activation='sigmoid')`

**Training Settings:**
- Loss: `binary_crossentropy`
- Optimizer: `adam`
- Epochs: 20
- Batch Size: 4




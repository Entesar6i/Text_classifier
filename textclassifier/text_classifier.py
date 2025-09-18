# === Imports ===
import nltk
import re
import numpy as np
import pandas as pd
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# === NLTK setup ===
print("🔽 Downloading NLTK resources...")
nltk.download('punkt', quiet=True)       # Tokenizer
nltk.download('stopwords', quiet=True)   # Common English stopwords
nltk.download('wordnet', quiet=True)     # For lemmatization
print("✅ NLTK setup complete.\n")

# === Text Preprocessing ===
def preprocess_text(input_text):
    """
    Clean and normalize the input text.
    Tokenizes, removes stopwords, and lemmatizes.
    """
    if not isinstance(input_text, str) or not input_text.strip():
        return []

    # Lowercase + remove punctuation/numbers
    cleaned = re.sub(r'[^a-zA-Z\s]', '', input_text.lower())

    # Tokenize the cleaned string
    words = word_tokenize(cleaned)

    # Remove common stopwords like "the", "is", etc.
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w not in stop_words]

    # Lemmatize words to their base form
    lemmatizer = WordNetLemmatizer()
    processed = [lemmatizer.lemmatize(w) for w in words]

    return processed

# === Bag-of-Words Builder ===
def create_bow_matrix(token_lists):
    """
    Converts tokenized texts into a Bag-of-Words matrix.
    Returns the matrix and the vocabulary used.
    """
    # Flatten all token lists into one big list
    all_terms = [term for doc in token_lists for term in doc]

    # Sort and deduplicate to build vocabulary
    vocabulary = sorted(set(all_terms))

    # Map each word to an index
    word_map = {term: idx for idx, term in enumerate(vocabulary)}

    # Initialize matrix: rows = docs, cols = words
    matrix = np.zeros((len(token_lists), len(vocabulary)), dtype=int)

    for i, tokens in enumerate(token_lists):
        counts = Counter(tokens)
        for term, freq in counts.items():
            if term in word_map:
                matrix[i, word_map[term]] = freq

    return matrix, vocabulary

# === Load & Process Dataset ===
print("📂 Reading feedback dataset...")
data = pd.read_csv('feedback.csv')  # CSV should have 'text' and 'label' columns
print(f"🧾 {len(data)} records loaded.\n")

print("🧹 Preprocessing text data...")
data['tokens'] = data['text'].apply(preprocess_text)

# === Build Features ===
print("🧠 Generating Bag-of-Words features...")
X, vocab = create_bow_matrix(data['tokens'].tolist())
y = data['label'].values

print(f"🗃️ Vocabulary size: {len(vocab)}")
print(f"📐 Matrix dimensions: {X.shape}\n")

# === Train/Test Split ===
print("🧪 Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# === Model Setup ===
print("🔧 Building neural network model...")
model = Sequential([
    Dense(16, activation='relu', input_shape=(X_train.shape[1],)),  # Input + hidden layer
    Dense(1, activation='sigmoid')  # Output layer (binary classification)
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# === Training ===
print("🚀 Starting training...\n")
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=4,
    validation_data=(X_test, y_test),
    verbose=1
)

# === Evaluate Model ===
print("\n📊 Evaluating performance...")
predicted = (model.predict(X_test) > 0.5).astype(int).flatten()
final_accuracy = accuracy_score(y_test, predicted)
print(f"✅ Accuracy: {final_accuracy * 100:.1f}%\n")

# === Feedback Classification Function ===
def classify_feedback(raw_text):
    """
    Takes a raw string input, preprocesses it, converts to BoW,
    runs prediction, and prints the sentiment.
    """
    if not raw_text.strip():
        print("⚠️ Empty input.")
        return

    # Preprocess and vectorize input
    tokens = preprocess_text(raw_text)
    vector = np.zeros((1, len(vocab)))
    index_map = {term: idx for idx, term in enumerate(vocab)}

    for token in tokens:
        if token in index_map:
            vector[0, index_map[token]] += 1

    # Make prediction
    probability = model.predict(vector)[0][0]
    sentiment = "Positive" if probability > 0.5 else "Negative"

    print(f"\n📝 '{raw_text}'")
    print(f"🔎 Sentiment: {sentiment} (Confidence: {probability:.2f})\n")

# === Sample Predictions ===
print("🔍 Sample predictions:")
classify_feedback("Great product!")
classify_feedback("Worst experience ever.")
classify_feedback("I love this service.")
classify_feedback("This is terrible.")

# === Interactive Prompt ===
print("=" * 50)
print("🗣️ Enter feedback to classify (or type 'quit'):")

while True:
    user_input = input("\n> ").strip()
    if user_input.lower() == 'quit':
        print("👋 Exiting. Goodbye!")
        break
    classify_feedback(user_input)

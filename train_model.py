"""
train_model.py - Email Spam Detector Model Training Script
-----------------------------------------------------------
This script trains multiple machine learning models on a spam email dataset,
compares their accuracy, and saves the best model for use in the Flask app.

Steps:
1. Load dataset
2. Preprocess text (clean, lowercase, remove stopwords, stem)
3. Convert text to TF-IDF vectors
4. Train Naive Bayes, Logistic Regression, and SVM models
5. Compare accuracy and save the best model
"""

import pandas as pd
import numpy as np
import string
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Try to import NLTK components (optional - fallback to basic cleaning if unavailable)
try:
    import nltk
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    NLTK_AVAILABLE = True
    STOP_WORDS = set(stopwords.words('english'))
    stemmer = PorterStemmer()
except Exception:
    NLTK_AVAILABLE = False
    # Basic English stopwords as fallback
    STOP_WORDS = {'i','me','my','we','our','you','your','he','she','it','they','this',
                  'that','a','an','the','and','or','but','in','on','at','to','for',
                  'is','are','was','were','be','been','have','has','had','do','does',
                  'did','will','would','could','should','may','might','can','not','no'}

# ──────────────────────────────────────────────────────────────────────────────
# Step 1: Text Preprocessing Function
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_text(text):
    """
    Clean and preprocess a single email text:
    - Convert to lowercase
    - Remove punctuation
    - Remove stopwords
    - Apply stemming (if NLTK available)
    """
    # Convert to lowercase
    text = str(text).lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Split into words (tokens)
    words = text.split()

    # Remove stopwords and apply stemming
    if NLTK_AVAILABLE:
        words = [stemmer.stem(word) for word in words if word not in STOP_WORDS]
    else:
        words = [word for word in words if word not in STOP_WORDS]

    # Join words back into a single string
    return ' '.join(words)


# ──────────────────────────────────────────────────────────────────────────────
# Step 2: Load and Prepare Dataset
# ──────────────────────────────────────────────────────────────────────────────

def load_dataset(filepath='spam.csv'):
    """Load the spam dataset from CSV file."""
    print(f"\n📂 Loading dataset from: {filepath}")

    df = pd.read_csv(filepath, encoding='latin-1')

    # Keep only the label and text columns
    df = df[['label', 'text']].copy()

    # Remove any null values
    df.dropna(inplace=True)

    # Encode labels: spam = 1, ham = 0
    df['label_encoded'] = df['label'].map({'spam': 1, 'ham': 0})

    print(f"✅ Dataset loaded: {len(df)} emails")
    print(f"   Spam emails : {df['label_encoded'].sum()}")
    print(f"   Ham emails  : {len(df) - df['label_encoded'].sum()}")

    return df


# ──────────────────────────────────────────────────────────────────────────────
# Step 3: Train and Evaluate Models
# ──────────────────────────────────────────────────────────────────────────────

def train_and_evaluate(df):
    """Train multiple models, compare performance, return the best one."""

    print("\n🔧 Preprocessing text data...")
    df['cleaned_text'] = df['text'].apply(preprocess_text)

    # Split data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_text'],
        df['label_encoded'],
        test_size=0.2,
        random_state=42,
        stratify=df['label_encoded']
    )

    print(f"   Training samples : {len(X_train)}")
    print(f"   Testing samples  : {len(X_test)}")

    # ── TF-IDF Vectorization ──────────────────────────────────────────────────
    print("\n📊 Creating TF-IDF vectors...")
    vectorizer = TfidfVectorizer(
        max_features=5000,    # Use top 5000 words
        ngram_range=(1, 2),   # Use single words AND word pairs
        min_df=1              # Word must appear at least once
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf  = vectorizer.transform(X_test)

    # ── Define Models ─────────────────────────────────────────────────────────
    models = {
        'Naive Bayes'        : MultinomialNB(alpha=0.1),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Support Vector Machine': LinearSVC(random_state=42, max_iter=2000)
    }

    results = {}
    print("\n🤖 Training and evaluating models...")
    print("=" * 65)

    for name, model in models.items():
        # Train the model
        model.fit(X_train_tfidf, y_train)

        # Make predictions
        y_pred = model.predict(X_test_tfidf)

        # Calculate metrics
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1   = f1_score(y_test, y_pred, zero_division=0)
        cm   = confusion_matrix(y_test, y_pred)

        results[name] = {
            'model'    : model,
            'accuracy' : acc,
            'precision': prec,
            'recall'   : rec,
            'f1_score' : f1,
            'confusion_matrix': cm.tolist()
        }

        print(f"\n  Model      : {name}")
        print(f"  Accuracy   : {acc*100:.2f}%")
        print(f"  Precision  : {prec*100:.2f}%")
        print(f"  Recall     : {rec*100:.2f}%")
        print(f"  F1 Score   : {f1*100:.2f}%")

    print("=" * 65)

    # ── Select Best Model ─────────────────────────────────────────────────────
    best_name = max(results, key=lambda k: results[k]['f1_score'])
    best_result = results[best_name]

    print(f"\n🏆 Best Model: {best_name}")
    print(f"   F1 Score  : {best_result['f1_score']*100:.2f}%")
    print(f"   Accuracy  : {best_result['accuracy']*100:.2f}%")

    return best_result['model'], vectorizer, results, best_name


# ──────────────────────────────────────────────────────────────────────────────
# Step 4: Save Model and Vectorizer
# ──────────────────────────────────────────────────────────────────────────────

def save_model(model, vectorizer, results, best_name):
    """Save the trained model, vectorizer, and metrics to disk."""

    # Save model and vectorizer using joblib
    joblib.dump(model, 'model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')

    # Save metrics as a JSON-serializable dictionary
    metrics_to_save = {}
    for name, data in results.items():
        metrics_to_save[name] = {
            'accuracy' : round(data['accuracy'], 4),
            'precision': round(data['precision'], 4),
            'recall'   : round(data['recall'], 4),
            'f1_score' : round(data['f1_score'], 4),
            'confusion_matrix': data['confusion_matrix']
        }

    import json
    with open('model_metrics.json', 'w') as f:
        json.dump({
            'best_model': best_name,
            'metrics'   : metrics_to_save
        }, f, indent=2)

    print("\n💾 Files saved:")
    print("   model.pkl        - Trained ML model")
    print("   vectorizer.pkl   - TF-IDF vectorizer")
    print("   model_metrics.json - Model performance metrics")


# ──────────────────────────────────────────────────────────────────────────────
# Main Execution
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 65)
    print("   EMAIL SPAM DETECTOR - Model Training")
    print("=" * 65)

    # Load dataset
    df = load_dataset('spam.csv')

    # Train models and get the best one
    best_model, vectorizer, results, best_name = train_and_evaluate(df)

    # Save everything to disk
    save_model(best_model, vectorizer, results, best_name)

    print("\n✅ Training complete! Model is ready to use.")
    print("   Run: python app.py   to start the web application.")
    print("=" * 65)

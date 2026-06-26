"""
predict.py - Email Prediction Helper
--------------------------------------
Loads the saved model and vectorizer, preprocesses input text,
and returns a spam/ham prediction with confidence percentage.
"""

import joblib
import string
import os
import numpy as np

# Try to import NLTK (optional)
try:
    import nltk
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    NLTK_AVAILABLE = True
    STOP_WORDS = set(stopwords.words('english'))
    stemmer = PorterStemmer()
except Exception:
    NLTK_AVAILABLE = False
    STOP_WORDS = {'i','me','my','we','our','you','your','he','she','it','they','this',
                  'that','a','an','the','and','or','but','in','on','at','to','for',
                  'is','are','was','were','be','been','have','has','had','do','does',
                  'did','will','would','could','should','may','might','can','not','no'}


def preprocess_text(text):
    """
    Clean and preprocess input email text for prediction.
    Must match the same preprocessing done during training.
    """
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    if NLTK_AVAILABLE:
        words = [stemmer.stem(w) for w in words if w not in STOP_WORDS]
    else:
        words = [w for w in words if w not in STOP_WORDS]
    return ' '.join(words)


def get_spam_keywords(text):
    """
    Identify common spam indicator words in the email text.
    Returns a list of detected spam keywords.
    """
    spam_keywords = [
        'free', 'win', 'winner', 'won', 'prize', 'congratulations',
        'click', 'claim', 'offer', 'urgent', 'guaranteed', 'money',
        'cash', 'credit', 'loan', 'discount', 'limited', 'exclusive',
        'buy', 'earn', 'income', 'profit', 'bonus', 'lottery',
        'billion', 'million', '$', '£', '€', 'selected', 'lucky',
        'verify', 'suspended', 'account', 'password', 'update',
        'confirm', 'warning', 'alert', 'risk', 'expire'
    ]
    text_lower = text.lower()
    found = [kw for kw in spam_keywords if kw in text_lower]
    return found[:5]  # Return up to 5 keywords


def generate_reason(prediction, confidence, email_text):
    """
    Generate a human-readable explanation for the prediction result.
    """
    if prediction == 1:  # SPAM
        keywords = get_spam_keywords(email_text)
        if keywords:
            kw_str = ', '.join([f'"{k}"' for k in keywords])
            return f"Contains spam indicator words ({kw_str}), promotional language, and suspicious patterns detected by the model."
        else:
            return "The pattern of words, urgency signals, and sentence structure match known spam characteristics."
    else:  # HAM (not spam)
        return "Normal conversational language detected. No suspicious keywords, promotional content, or phishing patterns found."


# Load model and vectorizer once (cached at module level)
_model = None
_vectorizer = None

def load_model():
    """Load the model and vectorizer from disk (lazy loading)."""
    global _model, _vectorizer
    if _model is None:
        if not os.path.exists('model.pkl') or not os.path.exists('vectorizer.pkl'):
            raise FileNotFoundError(
                "Model files not found! Please run 'python train_model.py' first."
            )
        _model = joblib.load('model.pkl')
        _vectorizer = joblib.load('vectorizer.pkl')
    return _model, _vectorizer


def predict_email(email_text):
    """
    Main prediction function.

    Parameters:
        email_text (str): The raw email content to classify.

    Returns:
        dict: {
            'prediction'  : 'Spam' or 'Not Spam',
            'label'       : 1 (spam) or 0 (ham),
            'confidence'  : float (0-100),
            'reason'      : str explanation,
            'status'      : 'success' or 'error',
            'message'     : error message if any
        }
    """
    try:
        # Validate input
        if not email_text or not email_text.strip():
            return {'status': 'error', 'message': 'Email text cannot be empty.'}

        if len(email_text.strip()) < 5:
            return {'status': 'error', 'message': 'Email text is too short to analyze.'}

        # Load model
        model, vectorizer = load_model()

        # Preprocess text
        cleaned = preprocess_text(email_text)

        # Vectorize
        text_vector = vectorizer.transform([cleaned])

        # Predict
        label = int(model.predict(text_vector)[0])

        # Get confidence score
        # LinearSVC uses decision_function; others use predict_proba
        try:
            proba = model.predict_proba(text_vector)[0]
            confidence = float(max(proba)) * 100
        except AttributeError:
            # For LinearSVC - use decision function distance as proxy
            decision = model.decision_function(text_vector)[0]
            confidence = min(99.9, 50 + abs(float(decision)) * 15)

        # Round confidence to 1 decimal place
        confidence = round(confidence, 1)

        # Cap at 99.9%
        confidence = min(confidence, 99.9)

        # Generate result
        prediction_label = 'Spam' if label == 1 else 'Not Spam'
        reason = generate_reason(label, confidence, email_text)

        return {
            'status'    : 'success',
            'prediction': prediction_label,
            'label'     : label,
            'confidence': confidence,
            'reason'    : reason
        }

    except FileNotFoundError as e:
        return {'status': 'error', 'message': str(e)}
    except Exception as e:
        return {'status': 'error', 'message': f'Prediction error: {str(e)}'}


# ── Quick Test ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    test_emails = [
        "Congratulations! You have won a free iPhone. Click here now to claim your prize!",
        "Hello Sir, tomorrow's class will start at 10 AM in Room 204.",
        "URGENT: Your bank account has been suspended. Verify your details immediately!"
    ]

    print("\nTesting predict.py...")
    for email in test_emails:
        result = predict_email(email)
        print(f"\nEmail   : {email[:60]}...")
        print(f"Result  : {result.get('prediction')} ({result.get('confidence')}%)")
        print(f"Reason  : {result.get('reason')}")

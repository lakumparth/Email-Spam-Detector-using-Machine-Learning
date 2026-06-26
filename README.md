# 🛡️ SpamShield – Email Spam Detector

A beginner-to-intermediate machine learning project that classifies email text as **Spam** or **Not Spam (Ham)** using NLP and scikit-learn. Built with Python, Flask, Bootstrap 5, and SQLite.

---

## 📸 Features

| Feature | Description |
|---|---|
| 🔍 Email Detection | Paste email text and get instant spam/ham classification |
| 📊 Confidence Score | Each result shows a confidence percentage (0–99.9%) |
| 💾 Prediction History | All checks saved to SQLite with search and export |
| 📈 Dashboard | Pie and bar charts via Chart.js, model metrics |
| 🤖 Multi-model Training | Naive Bayes, Logistic Regression, SVM compared automatically |
| 🔒 Security | Input sanitization, length limits, SQL injection protection |

---

## 🗂️ Project Structure

```
Email-Spam-Detector/
│
├── app.py              ← Flask web application (main entry point)
├── train_model.py      ← ML model training script
├── predict.py          ← Prediction helper module
├── spam.csv            ← Sample email dataset
├── model.pkl           ← Saved ML model (generated after training)
├── vectorizer.pkl      ← Saved TF-IDF vectorizer (generated after training)
├── model_metrics.json  ← Model performance metrics (generated after training)
├── database.db         ← SQLite database (auto-created on first run)
├── requirements.txt    ← Python dependencies
│
├── templates/
│   ├── base.html       ← Shared layout with navbar & footer
│   ├── index.html      ← Home page
│   ├── detect.html     ← Spam detection page
│   ├── history.html    ← Prediction history page
│   └── dashboard.html  ← Analytics dashboard
│
└── static/
    ├── css/            ← Custom stylesheets (optional)
    ├── js/             ← Custom JavaScript (optional)
    └── images/         ← Images/icons (optional)
```

---

## ⚙️ Installation Guide

### Step 1 – Clone or download the project

```bash
git clone https://github.com/your-username/Email-Spam-Detector.git
cd Email-Spam-Detector
```

Or simply extract the downloaded ZIP folder.

### Step 2 – Create a virtual environment (recommended)

```bash
python -m venv venv

# Activate on Windows:
venv\Scripts\activate

# Activate on Mac/Linux:
source venv/bin/activate
```

### Step 3 – Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 – Train the ML model

This step loads `spam.csv`, trains all models, and saves the best one.

```bash
python train_model.py
```

**Expected output:**
```
EMAIL SPAM DETECTOR - Model Training
...
Model      : Naive Bayes        → Accuracy: 94.XX%
Model      : Logistic Regression → Accuracy: 96.XX%
Model      : Support Vector Machine → Accuracy: 97.XX%

Best Model : Support Vector Machine
Files saved: model.pkl, vectorizer.pkl, model_metrics.json
Training complete!
```

### Step 5 – Run the web application

```bash
python app.py
```

Open your browser and go to: **http://127.0.0.1:5000**

---

## 🧠 Machine Learning Pipeline

```
Raw Email Text
     ↓
Lowercase + Remove Punctuation
     ↓
Remove Stop Words (NLTK)
     ↓
Stemming (PorterStemmer)
     ↓
TF-IDF Vectorization (top 5000 features, unigrams + bigrams)
     ↓
Model Training:
  • Naive Bayes (MultinomialNB)
  • Logistic Regression
  • Linear SVC (Support Vector Machine)
     ↓
Best model selected by F1 Score
     ↓
Saved as model.pkl + vectorizer.pkl
```

---

## 📊 Evaluation Metrics

The training script reports for each model:

- **Accuracy** – % of correct predictions
- **Precision** – Of all emails flagged as spam, how many actually are
- **Recall** – Of all actual spam, how many were caught
- **F1 Score** – Harmonic mean of precision and recall (used to pick best model)
- **Confusion Matrix** – TP / FP / TN / FN breakdown

---

## 📡 API Endpoint

### POST `/predict`

Accepts JSON body, returns prediction result.

**Request:**
```json
{
  "email_text": "Congratulations! You have won a free iPhone!"
}
```

**Response:**
```json
{
  "status": "success",
  "prediction": "Spam",
  "label": 1,
  "confidence": 98.4,
  "reason": "Contains spam indicator words (\"free\", \"win\", \"congratulations\"), promotional language..."
}
```

---

## 🔒 Security Features

- HTML input sanitization (prevents XSS)
- Maximum input length (5000 characters)
- Parameterized SQL queries (prevents SQL injection)
- Flask `SECRET_KEY` configuration
- Request size limit (1MB max)

---

## 🛠️ Technologies Used

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Backend | Flask 3.0 |
| Frontend | HTML5, CSS3, Bootstrap 5, Chart.js |
| ML Library | Scikit-learn |
| NLP | NLTK (stopwords, stemming) |
| Vectorization | TF-IDF (scikit-learn) |
| Model Storage | Joblib |
| Database | SQLite (built-in) |

---

## 📋 Dataset Format

The project uses a `spam.csv` file with two columns:

```csv
label,text
ham,"Hello, how are you?"
spam,"Congratulations! You won $1000. Click here."
```

You can replace the sample CSV with a larger dataset like the [UCI SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection).

---

## 🎓 Suitable For

- College Final Year / Mini Project
- Machine Learning + Web Development integration demo
- Cybersecurity awareness tool
- Learning Flask, NLP, and scikit-learn

---

## 📄 License

This project is for educational purposes. Free to use, modify, and submit as a college project.

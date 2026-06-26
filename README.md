# Email-Spam-Detector-using-Machine-Learning
Create a beginner-to-intermediate level cybersecurity and machine learning project called "Email Spam Detector". The application should classify incoming email messages as Spam or Not Spam (Ham) using a trained machine learning model.  The project should have a clean and modern web interface and be suitable for a college final-year or mini project.


# 📧 Email Spam Detector

A Machine Learning-based web application that detects whether an email is **Spam** or **Not Spam (Ham)**. The project uses Natural Language Processing (NLP) and Machine Learning techniques to analyze email text and classify it accurately. It also provides a user-friendly web interface for real-time spam detection.

---

## 📌 Project Overview

Spam emails are unwanted messages that may contain advertisements, phishing links, or malicious content. This project helps users identify spam emails automatically using a trained machine learning model.

The application allows users to paste email text into a web page and instantly receive a prediction along with the confidence score.

---

## 🎯 Objectives

* Detect spam emails using Machine Learning.
* Improve email security awareness.
* Demonstrate the use of NLP in cybersecurity.
* Provide a simple and interactive web interface.
* Store prediction history for future reference.

---

## 🚀 Features

* Detect Spam and Non-Spam emails
* Clean and responsive web interface
* Machine Learning prediction using trained model
* Confidence score display
* Prediction history
* Dashboard with statistics
* SQLite database support
* Easy to use and beginner friendly

---

## 🛠️ Technologies Used

### Programming Language

* Python 3

### Backend

* Flask

### Frontend

* HTML
* CSS
* Bootstrap
* JavaScript

### Machine Learning

* Scikit-learn
* Pandas
* NumPy
* Joblib

### Database

* SQLite

---

## 📂 Project Structure

```
Email-Spam-Detector/

│── app.py
│── train_model.py
│── predict.py
│── model.pkl
│── vectorizer.pkl
│── spam.csv
│── database.db
│── requirements.txt
│── README.md
│
├── templates/
│   ├── index.html
│   ├── result.html
│   ├── history.html
│   └── dashboard.html
│
├── static/
│   ├── css/
│   ├── js/
│   └── images/
│
└── models/
```

---

## ⚙️ Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/Email-Spam-Detector.git
```

### Step 2: Open the Project Folder

```bash
cd Email-Spam-Detector
```

### Step 3: Create a Virtual Environment (Optional)

```bash
python -m venv venv
```

Activate the virtual environment:

**Windows**

```bash
venv\Scripts\activate
```

**Linux/macOS**

```bash
source venv/bin/activate
```

### Step 4: Install Required Packages

```bash
pip install -r requirements.txt
```

### Step 5: Train the Model

```bash
python train_model.py
```

### Step 6: Run the Flask Application

```bash
python app.py
```

### Step 7: Open Your Browser

```
http://127.0.0.1:5000
```

---

## 📊 Machine Learning Workflow

1. Load dataset
2. Clean email text
3. Remove punctuation
4. Convert text to lowercase
5. Remove stop words
6. Apply stemming
7. Convert text into TF-IDF vectors
8. Train the model
9. Save the trained model
10. Predict new emails

---

## 📈 Algorithms Used

* Multinomial Naive Bayes
* Logistic Regression
* Support Vector Machine (Optional)

---

## 📋 Dataset

The project uses a spam email dataset with the following format:

| Label | Email Text                                        |
| ----- | ------------------------------------------------- |
| ham   | Hello, how are you?                               |
| spam  | Congratulations! You won ₹50,000. Click here now! |

---

## 💻 How to Use

1. Start the application.
2. Open the homepage.
3. Paste the email content into the text box.
4. Click **Check Email**.
5. View the prediction result.
6. Check the confidence score.
7. View prediction history from the dashboard.

---

## 📊 Evaluation Metrics

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

---

## 🔒 Security Features

* Input validation
* SQL Injection prevention
* HTML escaping
* Safe handling of user input
* Maximum input length validation

---

## 📸 Sample Output

### Example 1

**Input**

```
Congratulations!
You have won a free iPhone.
Click here to claim your prize.
```

**Prediction**

```
Spam
Confidence: 98%
```

---

### Example 2

**Input**

```
Hello Team,

Tomorrow's meeting is scheduled at 10 AM.

Thank you.
```

**Prediction**

```
Not Spam
Confidence: 99%
```

---

## 🔮 Future Enhancements

* Email attachment scanning
* URL reputation checking
* Phishing link detection
* Gmail API integration
* Real-time email monitoring
* Deep Learning models
* Admin dashboard
* User authentication
* Multi-language support

---

## 🎓 Learning Outcomes

After completing this project, you will understand:

* Machine Learning basics
* Natural Language Processing (NLP)
* Flask web development
* Email spam detection
* Data preprocessing
* TF-IDF Vectorization
* Model training and testing
* SQLite database integration
* GitHub project management

---

## 🤝 Contributing

Contributions are welcome.

1. Fork the repository.
2. Create a new branch.
3. Commit your changes.
4. Push the branch.
5. Open a Pull Request.

---


## 👨‍💻 Author

Lakum Parth Ashvinbhai

Diploma Computer Engineering

Cyber Security & Machine Learning Project

---

## ⭐ If you like this project

Give this repository a ⭐ on GitHub if you found it helpful.

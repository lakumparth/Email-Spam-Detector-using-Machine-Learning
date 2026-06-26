"""
app.py - Email Spam Detector Flask Application
------------------------------------------------
Main web application that serves:
- Home page          : /
- Detection page     : /detect
- Prediction API     : /predict (POST)
- History page       : /history
- Dashboard page     : /dashboard
- Export CSV         : /export
- Delete history     : /delete/<id>
"""

import os
import json
import sqlite3
import html
from datetime import datetime

from flask import (
    Flask, render_template, request,
    jsonify, redirect, url_for, send_file, g
)

# Import our prediction module
from predict import predict_email

# ──────────────────────────────────────────────────────────────────────────────
# Flask App Configuration
# ──────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)
app.config['SECRET_KEY'] = 'spam-detector-secret-2024'
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024  # Max 1MB request
DATABASE = 'database.db'
MAX_EMAIL_LENGTH = 5000  # Maximum allowed email characters


# ──────────────────────────────────────────────────────────────────────────────
# Database Helper Functions
# ──────────────────────────────────────────────────────────────────────────────

def get_db():
    """Get database connection (creates one per request context)."""
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        db.row_factory = sqlite3.Row  # Access columns by name
    return db


@app.teardown_appcontext
def close_connection(exception):
    """Close database connection at end of each request."""
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()


def init_db():
    """Initialize the SQLite database and create tables if they don't exist."""
    with app.app_context():
        db = get_db()
        db.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                email_text  TEXT NOT NULL,
                prediction  TEXT NOT NULL,
                confidence  REAL NOT NULL,
                reason      TEXT,
                created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        db.commit()


def save_prediction(email_text, prediction, confidence, reason):
    """Save a prediction result to the database."""
    # Store only the first 300 characters of the email as a snippet
    snippet = email_text[:300]
    db = get_db()
    db.execute(
        'INSERT INTO predictions (email_text, prediction, confidence, reason) VALUES (?, ?, ?, ?)',
        (snippet, prediction, confidence, reason)
    )
    db.commit()


def get_stats():
    """Get overall statistics from the database."""
    db = get_db()
    total = db.execute('SELECT COUNT(*) FROM predictions').fetchone()[0]
    spam  = db.execute("SELECT COUNT(*) FROM predictions WHERE prediction = 'Spam'").fetchone()[0]
    safe  = total - spam
    spam_pct = round((spam / total * 100), 1) if total > 0 else 0
    return {
        'total'   : total,
        'spam'    : spam,
        'safe'    : safe,
        'spam_pct': spam_pct
    }


# ──────────────────────────────────────────────────────────────────────────────
# Load Model Metrics (from training)
# ──────────────────────────────────────────────────────────────────────────────

def load_metrics():
    """Load model performance metrics from JSON file."""
    if os.path.exists('model_metrics.json'):
        with open('model_metrics.json') as f:
            return json.load(f)
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    """Home page."""
    model_ready = os.path.exists('model.pkl')
    metrics = load_metrics()
    return render_template('index.html', model_ready=model_ready, metrics=metrics)


@app.route('/detect')
def detect():
    """Spam detection page."""
    return render_template('detect.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint: Accepts email text, returns prediction result as JSON.
    POST /predict
    Body: { "email_text": "..." }
    """
    try:
        data = request.get_json(silent=True)

        # Support both JSON and form submissions
        if data:
            email_text = data.get('email_text', '')
        else:
            email_text = request.form.get('email_text', '')

        # Input validation
        email_text = str(email_text).strip()

        if not email_text:
            return jsonify({'status': 'error', 'message': 'Please enter email text.'}), 400

        # Security: Sanitize input to prevent HTML injection
        email_text_safe = html.unescape(email_text)[:MAX_EMAIL_LENGTH]

        # Get prediction from ML model
        result = predict_email(email_text_safe)

        if result['status'] == 'success':
            # Save to database
            save_prediction(
                email_text_safe,
                result['prediction'],
                result['confidence'],
                result['reason']
            )

        return jsonify(result)

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/history')
def history():
    """Prediction history page."""
    db = get_db()
    search = request.args.get('search', '').strip()

    if search:
        rows = db.execute(
            "SELECT * FROM predictions WHERE email_text LIKE ? ORDER BY created_at DESC LIMIT 50",
            (f'%{search}%',)
        ).fetchall()
    else:
        rows = db.execute(
            'SELECT * FROM predictions ORDER BY created_at DESC LIMIT 50'
        ).fetchall()

    predictions = [dict(row) for row in rows]
    return render_template('history.html', predictions=predictions, search=search)


@app.route('/delete/<int:pred_id>', methods=['POST'])
def delete_prediction(pred_id):
    """Delete a single prediction record."""
    db = get_db()
    db.execute('DELETE FROM predictions WHERE id = ?', (pred_id,))
    db.commit()
    return redirect(url_for('history'))


@app.route('/delete_all', methods=['POST'])
def delete_all():
    """Delete all prediction history."""
    db = get_db()
    db.execute('DELETE FROM predictions')
    db.commit()
    return redirect(url_for('history'))


@app.route('/dashboard')
def dashboard():
    """Dashboard page with charts and statistics."""
    stats = get_stats()
    metrics = load_metrics()

    # Get last 7 days data for bar chart
    db = get_db()
    daily_data = db.execute('''
        SELECT DATE(created_at) as day,
               COUNT(*) as total,
               SUM(CASE WHEN prediction = 'Spam' THEN 1 ELSE 0 END) as spam_count
        FROM predictions
        GROUP BY DATE(created_at)
        ORDER BY day DESC
        LIMIT 7
    ''').fetchall()

    daily_stats = [dict(row) for row in daily_data][::-1]  # Reverse for chronological order

    return render_template('dashboard.html',
                           stats=stats,
                           metrics=metrics,
                           daily_stats=daily_stats)


@app.route('/export')
def export_csv():
    """Export prediction history as CSV file."""
    import csv
    import io

    db = get_db()
    rows = db.execute('SELECT * FROM predictions ORDER BY created_at DESC').fetchall()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['ID', 'Email Snippet', 'Prediction', 'Confidence (%)', 'Reason', 'Date & Time'])

    for row in rows:
        writer.writerow([
            row['id'],
            row['email_text'][:100] + '...' if len(row['email_text']) > 100 else row['email_text'],
            row['prediction'],
            row['confidence'],
            row['reason'],
            row['created_at']
        ])

    output.seek(0)

    from flask import Response
    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': 'attachment; filename=spam_history.csv'}
    )


# ──────────────────────────────────────────────────────────────────────────────
# Error Handlers
# ──────────────────────────────────────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return render_template('index.html'), 404

@app.errorhandler(413)
def too_large(e):
    return jsonify({'status': 'error', 'message': 'Input too large. Maximum 1MB allowed.'}), 413


# ──────────────────────────────────────────────────────────────────────────────
# App Entry Point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # Initialize database
    init_db()
    print("\n" + "=" * 55)
    print("  EMAIL SPAM DETECTOR - Flask Application")
    print("=" * 55)

    if not os.path.exists('model.pkl'):
        print("⚠️  Model not found! Training model first...")
        os.system('python train_model.py')

    print("🚀 Starting server at: http://127.0.0.1:5000")
    print("   Press CTRL+C to stop the server.")
    print("=" * 55 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5000)

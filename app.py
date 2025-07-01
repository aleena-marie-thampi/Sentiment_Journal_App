import os
import pickle
from flask import Flask, request, jsonify, render_template, session, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from urllib.parse import quote_plus
from datetime import datetime, timedelta
from pymongo import MongoClient
from bson import ObjectId
from collections import Counter
from dotenv import load_dotenv

load_dotenv()
DB_USERNAME = os.getenv("DB_USERNAME")
DB_PASSWORD = quote_plus(os.getenv("DB_PASSWORD"))
SECRET_KEY = os.getenv("SECRET_KEY")
MODEL_PATH = "model.pkl"

app = Flask(__name__)
app.secret_key = SECRET_KEY

client = MongoClient(f"mongodb+srv://{DB_USERNAME}:{DB_PASSWORD}@sendimentdb.f5psubg.mongodb.net/?retryWrites=true&w=majority")
db = client['sentiment_journal']
entries = db['journal_entries']
users_collection = db['users']

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin):
    def __init__(self, user_data):
        self.id = str(user_data['_id'])
        self.username = user_data['username']

@login_manager.user_loader
def load_user(user_id):
    try:
        user_data = users_collection.find_one({'_id': ObjectId(user_id)})
        return User(user_data) if user_data else None
    except:
        return None

from model import (
    predict_sent,
    generate_wordclouds,
    generate_journal_insights,
    evaluate_model,
    compare_models,
    generate_training_wordclouds_once
)

training_clouds = generate_training_wordclouds_once()

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)



@app.route('/health')
def health_check():
    return "OK", 200

@app.route('/')
def home():
    if not current_user.is_authenticated:
        return render_template('index.html', now=datetime.utcnow)

    today = datetime.utcnow().date()
    streak = 0
    latest_entry = None
    mood_counts = Counter()
    recent_entries = []

    user_entries = list(entries.find({'user_id': current_user.id}).sort('date', -1))

    for entry in user_entries:
        entry_date = datetime.strptime(entry['date'], "%Y-%m-%d").date()
        sentiment = entry.get('sentiment', 'Neutral')
        mood_counts[sentiment] += 1
        if latest_entry is None:
            latest_entry = entry
        if entry_date == today - timedelta(days=streak):
            streak += 1

    total = sum(mood_counts.values()) or 1
    mood_percent = {
        'Positive': int((mood_counts['Positive'] / total) * 100),
        'Neutral': int((mood_counts['Neutral'] / total) * 100),
        'Negative': int((mood_counts['Negative'] / total) * 100)
    }

    total_entries = len(user_entries)
    total_words = sum(len(e.get('text', '').split()) for e in user_entries)
    total_words_str = f"{total_words / 1000:.1f}k" if total_words >= 1000 else str(total_words)

    for e in user_entries[:3]:
        recent_entries.append({
            "title": "Entry",
            "date": e.get("date", ""),
            "text": e.get("text", ""),
            "sentiment": e.get("sentiment", "Neutral")
        })

    return render_template('index.html',
        mood_percent=mood_percent,
        latest_entry=latest_entry,
        recent_entries=recent_entries,
        streak=streak,
        total_entries=total_entries,
        total_words=total_words_str,
        now=datetime.utcnow
    )

@app.route('/journal')
@login_required
def journal():
    today_str = datetime.utcnow().strftime("%Y-%m-%d")
    entry = entries.find_one({'user_id': current_user.id, 'date': today_str})
    existing_text = entry['text'] if entry else ''

    formatted_probs = {}
    if entry and 'probabilities' in entry:
        formatted_probs = {k: f"{v * 100:.1f}%" for k, v in entry['probabilities'].items()}

    return render_template(
        'journal.html',
        existing_text=existing_text,
        has_entry=bool(entry),
        sentiment=entry.get('sentiment') if entry else None,
        confidence=(entry.get('confidence') * 100) if entry and 'confidence' in entry else None,
        formatted_probs=formatted_probs
    )

@app.route('/calendar')
@login_required
def calendar():
    return render_template('calendar.html')

@app.route('/delete_today', methods=['POST'])
@login_required
def delete_today():
    today_str = datetime.utcnow().strftime("%Y-%m-%d")
    result = entries.delete_one({'user_id': current_user.id, 'date': today_str})
    flash('Today\'s entry deleted.' if result.deleted_count else 'No entry found.', 'success' if result.deleted_count else 'danger')
    return redirect(url_for('journal'))

@app.route('/analyze', methods=['POST'])
@login_required
def analyze():
    data = request.get_json()
    text = data['entry']
    date_today = datetime.utcnow().strftime("%Y-%m-%d")
    result = predict_sent(text)

    entries.update_one(
        {'date': date_today, 'user_id': current_user.id},
        {'$set': {
            'text': text,
            'sentiment': result['label'],
            'confidence': round(result['confidence'], 2),
            'probabilities': result['probabilities']
        }},
        upsert=True
    )

    return jsonify({
        'sentiment': result['label'],
        'confidence': result['confidence'],
        'probabilities': result['probabilities']
    })

@app.route('/get_moods')
@login_required
def get_moods():
    events = []
    for doc in entries.find({'user_id': current_user.id}):
        mood = doc.get("sentiment", "Neutral")
        date = doc.get("date")
        color = {"Positive": "green", "Negative": "red", "Neutral": "orange"}.get(mood, "gray")
        events.append({"title": mood, "start": date, "color": color})
    return jsonify(events)

@app.route('/get_entry')
@login_required
def get_entry():
    date = request.args.get('date')
    doc = entries.find_one({'date': date, 'user_id': current_user.id})
    if doc:
        return jsonify({
            'success': True,
            'date': doc.get('date'),
            'text': doc.get('text'),
            'sentiment': doc.get('sentiment')
        })
    return jsonify({'success': False, 'message': 'No entry found for that date'})

@app.route('/delete_entry', methods=['POST'])
@login_required
def delete_entry():
    data = request.get_json()
    date = data.get('date')
    result = entries.delete_one({'date': date, 'user_id': current_user.id})
    return jsonify({'success': result.deleted_count > 0})

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if users_collection.find_one({'username': username}):
            flash('Username already exists', 'danger')
            return redirect(url_for('register'))
        hashed_pw = generate_password_hash(password)
        users_collection.insert_one({'username': username, 'password': hashed_pw})
        flash('Registration successful. Please login.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user_data = users_collection.find_one({'username': username})
        if user_data and check_password_hash(user_data['password'], password):
            user = User(user_data)
            login_user(user)
            flash('Logged in successfully!', 'success')
            return redirect(url_for('home'))
        flash('Invalid username or password', 'danger')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))

@app.route('/get_all_entries')
@login_required
def get_all_entries():
    user_entries = entries.find({'user_id': current_user.id})
    return jsonify([{'date': e.get('date'), 'text': e.get('text')} for e in user_entries])

@app.route('/entry/<date>')
@login_required
def view_entry(date):
    entry = entries.find_one({'date': date, 'user_id': current_user.id})
    if entry:
        formatted_probs = {k: f"{v * 100:.1f}%" for k, v in entry['probabilities'].items()} if 'probabilities' in entry else {}
        return render_template('entry.html', entry=entry, formatted_probs=formatted_probs)
    flash('Entry not found', 'danger')
    return redirect(url_for('calendar'))

@app.route('/entry/<date>/delete', methods=['POST'])
@login_required
def delete_entry_page(date):
    entries.delete_one({'date': date, 'user_id': current_user.id})
    return redirect(url_for('calendar'))

@app.route('/insights')
@login_required
def insights():
    user_entries = list(entries.find({'user_id': current_user.id}))
    plots = generate_journal_insights(user_entries)
    return render_template('insights.html', **plots)

@app.route('/wordclouds')
@login_required
def wordclouds():
    user_entries = list(entries.find({'user_id': current_user.id}))
    user_clouds = generate_wordclouds(user_entries)
    return render_template('wordclouds.html', clouds=user_clouds, training_clouds=training_clouds)

results = compare_models()

@app.route('/model-comparison')
@login_required
def model_comparison():
    return render_template('comparison.html', results=results)

@app.route('/evaluation')
@login_required
def evaluation():
    results = evaluate_model()
    return render_template('evaluation.html', **results)

@app.template_filter('file_exists')
def file_exists_filter(path):
    return os.path.exists(path)

@app.context_processor
def inject_now():
    return {'now': datetime.utcnow()}

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)


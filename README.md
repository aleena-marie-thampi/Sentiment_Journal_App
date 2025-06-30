
# 💙 Sentiment Journal App

A full-stack AI-powered journal web app for tracking your emotions over time, analyzing your writing using machine learning, and visualizing mood trends with stats, word clouds, and more.

## 🔧 Features

- ✍️ Write daily journal entries
- 🤖 Automatic sentiment analysis (Positive / Neutral / Negative)
- 📅 Mood calendar with color-coded entries
- 📊 Insights dashboard (word/char stats, sentiment distribution)
- ☁️ Word clouds (based on training data + user journal)
- 🔐 User authentication (register/login/logout)
- 📈 Model evaluation (accuracy, classification report, confusion matrix)
- ⚖️ Model comparison (Logistic Regression, Naive Bayes, Random Forest)

## 🧠 AI Model

- Preprocessing: TF-IDF with n-grams
- Sentiment Labeling: VADER + Ambiguous samples
- Classifier: Logistic Regression (default), Naive Bayes, Random Forest
- Evaluation: Accuracy, precision/recall/F1, confusion matrix

## 📁 Folder Structure

```
sentiment-journal/
│
├── app.py                  # Main Flask backend
├── model.py                # ML training + prediction
├── templates/              # HTML templates (Jinja2)
│   ├── base.html
│   ├── index.html
│   ├── journal.html
│   ├── calendar.html
│   ├── entry.html
│   ├── insights.html
│   ├── wordclouds.html
│   ├── evaluation.html
│   ├── comparison.html
├── static/
│   └── style.css           # Custom CSS styles
├── train.csv               # Base dataset
├── amb.csv                 # Manually labeled ambiguous samples
├── model.pkl               # Saved model & vectorizer
```

## ⚙️ Setup Instructions

1. **Clone the repo**
   ```bash
   git clone https://github.com/your-username/sentiment-journal.git
   cd sentiment-journal
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set MongoDB URI**
   In `app.py`, update:
   ```python
   username = "YourMongoUsername"
   password = quote_plus("YourMongoPassword")
   ```

4. **Run the app**
   ```bash
   python app.py
   ```

5. **Access**
   Open `http://127.0.0.1:5000` in your browser.

## 🚀 Deployment (Render)

1. Upload your project to GitHub
2. Create a new **Web Service** in [Render](https://render.com)
3. Add environment variable:
   ```
   PYTHON_VERSION = 3.10
   ```
4. Add `requirements.txt` and `start` command: `python app.py`
5. Deploy and enjoy 🎉

## 🙌 Credits

- Built with Flask, MongoDB, NLTK, scikit-learn, Seaborn, Matplotlib
- Inspired by AccelerateX Internship Prompt

---

> Made with 💙 by Aleena Marie Thampi

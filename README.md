# 🧠 Sentiment Journal Web App 📓

A full-stack AI-powered journaling platform that helps users understand and track their emotions through daily writing. Built with **Flask**, **MongoDB**, and a **custom-trained machine learning model**, this app analyzes journal entries for sentiment, visualizes mood trends, and offers emotional insights.

---

## 🌟 Features

### ✍️ Daily Journal
- Securely write and save journal entries.
- Each entry is analyzed using a **Random Forest sentiment classifier**.
- Moods are auto-labeled as **Positive**, **Neutral**, or **Negative**.

### 📆 Mood Calendar
- Interactive, color-coded calendar.
- Click on any date to preview and view full journal entries.
- Tracks journaling streaks.

### 📊 Insights & Dashboard
- View recent entries, streak count, and word totals.
- Mood distribution overview.
- Motivational quote generator.

### ☁️ Word Cloud Visualizations
- **Training Data Word Cloud**: Shows dominant terms from the model’s dataset.
- **User Word Cloud**: Personalized cloud based on your own entries.

### 🔍 Model Evaluation & Comparison
- Compare performance of:
  - Random Forest (default)
  - Logistic Regression
  - Naive Bayes
- View **accuracy**, **confusion matrix**, and **classification report**.

### 🔐 Authentication
- User registration and secure login.
- Each user accesses only their journal entries.

---

## 🧠 Sentiment Model

- **Algorithm**: Random Forest Classifier (best-performing model)
- **Training Data**: Combined train.csv + amb.csv (ambiguous/contrastive cases)
- **Preprocessing**:
  - Stopword removal
  - Lemmatization
  - Negation handling (not_good)
  - TF-IDF vectorization (1–3 grams)
- **Evaluation Metrics**:
  - Accuracy, Precision, Recall, F1-score
  - Visualized with confusion matrices

---

## 🗂 Project Structure

```
📁 sentiment-journal-app/
├── app.py               # Main Flask app
├── model.py             # ML logic and utilities
├── model.pkl            # Pretrained sentiment model
├── train.csv            # Primary dataset
├── amb.csv              # Ambiguous edge cases
├── requirements.txt
├── templates/
│   ├── index.html
│   ├── journal.html
│   ├── calendar.html
│   ├── entry.html
│   ├── login.html
│   ├── register.html
│   ├── base.html
│   ├── evaluation.html
│   ├── wordclouds.html
│   ├── insights.html
│   └── comparison.html
├── static/
│   └── style.css
└── README.md
```


---

## 🚀 Run on Replit

  1. Fork this repo on Replit
  2. Add secrets (🔒 icon):
    DB_USERNAME=your_username
    DB_PASSWORD=your_password
    SECRET_KEY=your_secret_key

  3. (If model.pkl is large)
     Add to app.py:
     wget https://your-link.com/model.pkl

  4. Click ▶️ Run
     App runs at: https://your-replit-name.username.repl.co

---

## 📊 Evaluation & Comparison Pages

| Page            | Purpose                                                  |
|-----------------|----------------------------------------------------------|
| /evaluation   | Show model accuracy, classification report, confusion matrix |
| /model-comparison | Compare RF, LR, and NB classifiers using real data        |

---

## 📁 Dataset Overview

### train.csv
- Labeled dataset for training.
- Clean, structured sentiment examples.

### amb.csv
- Ambiguous/mixed examples (100+), used for nuance testing:
  - *“I passed the exam but feel empty.”* → Neutral
  - *“It was fun, but I missed home.”* → Positive/Neutral

---

## 🛠 Technologies Used

- **Frontend**: HTML, CSS, Jinja2
- **Backend**: Python, Flask, Flask-Login
- **Database**: MongoDB Atlas
- **ML / NLP**: scikit-learn, NLTK, TextBlob
- **Visualization**: matplotlib, seaborn, wordcloud
- **Hosting**: Replit

---

## 🔮 Future Enhancements

- Export journals as PDF
- Add emoji-based mood selection
- Dark mode
- Graphs for mood trends
- AI-powered motivation suggestions

---

## 👩‍💻 Author

**Aleena Marie Thampi**  
- B.Tech CSE Student | Full-Stack & AI Enthusiast  
- GitHub: [@aleena-marie-thampi](https://github.com/aleena-marie-thampi)
- Video Demo: Included in the repo
- App:https://62800371-0d87-4457-affb-5ff6306cefc6-00-21ui26sbydzhw.sisko.replit.dev/

---

## 📃 License

This project is licensed under the **MIT License** — feel free to use, modify, and contribute.

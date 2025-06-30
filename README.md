
# Sentiment Journal Web App 🧠📓

Welcome to the **Sentiment Journal Web App**, a full-stack AI-powered application that allows users to **write daily journals**, view their **mood calendar**, and gain **emotional insights** powered by **Natural Language Processing (NLP)**. Built with Flask, MongoDB, and a custom-trained machine learning model.

---

## 🌟 Features

### 📝 Journal Entry
- Users can write daily journal entries.
- Each entry is analyzed using a trained **Random Forest sentiment model**.
- Mood (Positive, Negative, Neutral) is predicted and saved.

### 📆 Mood Calendar
- Visual calendar displays entries by date.
- Color-coded moods (e.g., green = positive, red = negative).
- Click on any date to view journal preview + full page option.

### 📊 Dashboard Insights
- **Recent Entries** displayed as cards.
- **Streak counter**, **Total entries**, **Words written** stats.
- Mood distribution chart.
- **Motivational quote** on the homepage.

### 🌥️ Word Cloud
- Two types of word clouds:
  - **Training Data Word Cloud**: Shows commonly associated words with moods based on the training dataset.
  - **User Journal Word Cloud**: Personalized based on the user’s journal entries.

### 🔐 Authentication
- User registration and login with secure password hashing.
- Each user sees only their entries.

### 🤖 Model Comparison
- Compare different models:
  - Logistic Regression
  - Naive Bayes
  - Random Forest (default)
- Accuracy, confusion matrix, classification report included.

---

## 🧠 ML Model Summary

- Training Dataset: `train.csv` + `amb.csv` (ambiguous examples).
- Labels: Positive, Neutral, Negative.
- Preprocessing includes stopword removal, lemmatization, and vectorization using CountVectorizer.
- Trained on a Random Forest Classifier (best performance).
- Saved as a `.pckl` file and loaded in Flask app.

---

## 🗂️ Project Structure

```
sentiment-journal-app/
│
├── static/
│   └── style.css                # Custom blue-themed styling
│
├── templates/
│   ├── base.html                # Shared navbar layout
│   ├── index.html               # Homepage with dashboard features
│   ├── journal.html             # Journal entry page
│   ├── calendar.html            # Mood calendar with date popups
│   ├── entry_detail.html        # Full entry view + delete
│   ├── login.html, register.html
│   ├── evaluate.html            # Model evaluation page
│   └── compare.html             # Model comparison page
│
├── model.py                    # Model training, prediction, word cloud, evaluation
├── app.py                      # Flask app logic
├── train.csv                   # Primary training data
├── amb.csv                     # Ambiguous data (50+ edge cases)
├── model.pckl                  # Saved trained model
├── requirements.txt            # Required Python packages
└── README.md                   # This file
```

---

## ⚙️ Installation Instructions

### 🧩 Prerequisites
- Python 3.10+
- pip
- MongoDB Atlas account (or local MongoDB)

### 🔧 Setup Steps

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/sentiment-journal-app.git
cd sentiment-journal-app

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up MongoDB URI in app.py
MONGO_URI = "your-mongodb-atlas-uri"

# 5. Run the app
python app.py
```

Open your browser at `http://127.0.0.1:5000`

---

## 🚀 Deployment on Render

### Step-by-Step Guide

1. Go to [https://render.com](https://render.com) and sign up.
2. Create a new Web Service:
   - Connect your GitHub repo.
   - Environment: Python 3.
3. In **Build Command**, set:
   ```
   pip install -r requirements.txt
   ```
4. In **Start Command**, set:
   ```
   gunicorn app:app
   ```
5. Add the following **Environment Variables**:
   - `MONGO_URI`: your MongoDB Atlas URI
   - (Optional) `SECRET_KEY`: Flask session key

6. Click **Deploy**. Wait for Render to build & host your app live!

---

## 🛠️ Technologies Used

- **Frontend**: HTML, CSS (custom blue & white theme), JS (for calendar)
- **Backend**: Python Flask
- **Database**: MongoDB Atlas
- **ML Libraries**: scikit-learn, nltk
- **Visualization**: WordCloud, matplotlib
- **Authentication**: flask-login

---

## 🧪 Evaluation & Model Comparison

- View detailed model evaluation with accuracy, precision, recall.
- Compare LR, Naive Bayes, and RF using your own data.
- Understand how ambiguous examples impact model accuracy.

---

## 🔮 Future Enhancements

- Add emoji-based mood selection for manual override.
- Export journals as PDF.
- Notifications/reminders to write entries.
- AI-generated daily motivation based on recent mood trends.
- Graphical trends over weeks/months.

---

## 📬 Contact

Made with 💙 by **Aleena Marie Thampi**  
Feel free to contribute or reach out for suggestions!

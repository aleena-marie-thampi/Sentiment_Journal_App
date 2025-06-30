import numpy as np
import pandas as pd
import re
import nltk
import pickle
import io
import base64
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('vader_lexicon')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = re.sub('[^a-zA-Z\s]', '', str(text))
    text = text.lower()
    tokens = text.split()

    processed = []
    skip = False
    for i in range(len(tokens)):
        if skip:
            skip = False
            continue
        if tokens[i] == 'not' and i + 1 < len(tokens):
            processed.append(f"not_{tokens[i+1]}")
            skip = True
        else:
            if tokens[i] not in stop_words:
                processed.append(lemmatizer.lemmatize(tokens[i]))
    return ' '.join(processed)

df = pd.read_csv('train.csv', encoding='ISO-8859-1').dropna()
df = df.drop(columns=['textID', 'selected_text', 'sentiment', 'Time of Tweet',
                      'Age of User', 'Country', 'Population -2020',
                      'Land Area (Km²)', 'Density (P/Km²)'], errors='ignore')
df = df.drop_duplicates('text')
df['text'] = df['text'].apply(preprocess)

sia = SentimentIntensityAnalyzer()
def get_polarity(text): return sia.polarity_scores(text)['compound']
def label_sentiment(score):
    return 'Positive' if score >= 0.05 else 'Negative' if score <= -0.05 else 'Neutral'

df['polarity'] = df['text'].apply(get_polarity)
df['sentiment'] = df['polarity'].apply(label_sentiment)

amb_df = pd.read_csv('amb.csv').dropna()
if amb_df['sentiment'].dtype == object:
    amb_df['text'] = amb_df['text'].apply(preprocess)

label_map = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
reverse_map = {1: 'Positive', 0: 'Neutral', -1: 'Negative'}
df['sentiment'] = df['sentiment'].map(label_map)
amb_df['sentiment'] = amb_df['sentiment'].map(label_map)

combined_df = pd.concat([df[['text', 'sentiment']], amb_df[['text', 'sentiment']]], ignore_index=True).drop_duplicates('text')

X = combined_df['text']
y = combined_df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=10000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_vec, y_train)

def predict_sent(text):
    clean = preprocess(text)
    vec = vectorizer.transform([clean])
    prediction = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    label = reverse_map[prediction]
    class_indices = list(model.classes_)
    probabilities = {reverse_map[c]: float(p) for c, p in zip(class_indices, proba)}
    return {
        'label': label,
        'confidence': max(probabilities.values()),
        'probabilities': probabilities
    }

def generate_journal_insights(entries):
    if not entries:
        return {'sentiment_chart': None, 'word_boxplot': None, 'char_boxplot': None}

    df = pd.DataFrame(entries)
    df['text'] = df['text'].fillna('')
    df['word_count'] = df['text'].apply(lambda x: len(x.split()))
    df['char_count'] = df['text'].apply(len)

    def plot_to_base64(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')

    
    sentiment_chart = plot_to_base64(sns.countplot(data=df, x='sentiment', palette='pastel').figure)
    plt.close()
    word_box = sns.boxplot(data=df, x='sentiment', y='word_count', palette='coolwarm')
    word_boxplot = plot_to_base64(word_box.figure)
    plt.close()
    char_box = sns.boxplot(data=df, x='sentiment', y='char_count', palette='Blues')
    char_boxplot = plot_to_base64(char_box.figure)
    plt.close()

    return {
        'sentiment_chart': sentiment_chart,
        'word_boxplot': word_boxplot,
        'char_boxplot': char_boxplot
    }

def evaluate_model():
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Negative', 'Neutral', 'Positive'])
    cm = confusion_matrix(y_test, y_pred, labels=[-1, 0, 1])

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Neg', 'Neu', 'Pos'], yticklabels=['Neg', 'Neu', 'Pos'])
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    confusion_img = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    return {
        'accuracy': f'{acc * 100:.2f}%',
        'report': report,
        'confusion_img': confusion_img
    }

training_cloud_cache = {}

def generate_training_wordclouds_once():
    global training_cloud_cache
    if training_cloud_cache:
        return training_cloud_cache

    df = pd.read_csv('train.csv', encoding='ISO-8859-1').dropna()
    amb_df = pd.read_csv('amb.csv').dropna()
    df = df.drop_duplicates('text')
    df['sentiment'] = df['text'].apply(lambda t: label_sentiment(get_polarity(preprocess(t))))
    amb_df['text'] = amb_df['text'].apply(preprocess)
    amb_df['sentiment'] = amb_df['sentiment'].map(label_map)

    combined = pd.concat([df[['text', 'sentiment']], amb_df[['text', 'sentiment']]])
    combined['sentiment'] = combined['sentiment'].map(reverse_map)

    images = {}
    for mood in ['Positive', 'Neutral', 'Negative']:
        text = " ".join(combined[combined['sentiment'] == mood]['text'])
        if not text.strip():
            images[mood] = None
            continue
        wc = WordCloud(width=500, height=300, background_color='white', colormap='coolwarm').generate(text)
        buf = io.BytesIO()
        wc.to_image().save(buf, format='PNG')
        buf.seek(0)
        images[mood] = base64.b64encode(buf.read()).decode('utf-8')

    training_cloud_cache = images
    return images

def generate_wordclouds(entries):
    if not entries:
        return {'Positive': None, 'Neutral': None, 'Negative': None}
    
    df = pd.DataFrame(entries)
    df['text'] = df['text'].fillna('')
    df['sentiment'] = df['sentiment'].fillna('Neutral')
    
    images = {}
    for mood in ['Positive', 'Neutral', 'Negative']:
        text = " ".join(df[df['sentiment'] == mood]['text'])
        if not text.strip():
            images[mood] = None
            continue
        wc = WordCloud(width=500, height=300, background_color='white', colormap='coolwarm').generate(text)
        buf = io.BytesIO()
        wc.to_image().save(buf, format='PNG')
        buf.seek(0)
        images[mood] = base64.b64encode(buf.read()).decode('utf-8')
    
    return images

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

def compare_models():
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=500),
        'Naive Bayes': MultinomialNB()
        
    }
    results = []

    for name, clf in models.items():
        clf.fit(X_train_vec, y_train)
        y_pred = clf.predict(X_test_vec)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=['Negative', 'Neutral', 'Positive'])

        cm = confusion_matrix(y_test, y_pred, labels=[-1, 0, 1])
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Neg', 'Neu', 'Pos'], yticklabels=['Neg', 'Neu', 'Pos'])
        ax.set_title(f'{name} - Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        confusion_img = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

        results.append({
            'model': name,
            'accuracy': f'{acc * 100:.2f}%',
            'report': report,
            'confusion_img': confusion_img
        })

    return results

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
    pickle.dump(vectorizer, f)

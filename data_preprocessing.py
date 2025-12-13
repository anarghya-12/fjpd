import re
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from scipy.sparse import hstack

# nltk
import nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))

# Regex + keyword setup
URL_RE = re.compile(r'https?://\S+|www\.\S+')
EMAIL_RE = re.compile(r'\S+@\S+')
CURRENCY_RE = re.compile(r'[\$\€\£]|rs\.|rupee|₹')
PHONE_RE = re.compile(r'\b\d{6,}\b')

SUSPICIOUS_KEYWORDS = [
    "work from home", "earn", "no experience", "bank details", "instant", "limited slots",
    "apply now", "make money", "earn money", "easy money", "click here", "wire transfer",
    "pay per day", "daily pay", "immediate hiring", "earn $", "send bank", "vacancy urgent"
]

# TEXT PROCESSING
def clean_text(s):
    if pd.isna(s):
        return ""
    s = str(s)
    s = s.replace('\n', ' ')
    s = URL_RE.sub(" ", s)
    s = EMAIL_RE.sub(" ", s)
    s = re.sub(r'[^A-Za-z0-9\s\$\€\£\₹]', " ", s)
    s = re.sub(r'\s+', ' ', s).strip().lower()
    return s


# NUMERIC FEATURES
def extract_numeric_features(text):
    features = {}
    features['char_len'] = len(text)
    features['word_count'] = len(text.split())
    features['avg_word_len'] = (
        sum(len(w) for w in text.split()) / features['word_count']
        if features['word_count'] > 0 else 0
    )
    features['num_urls'] = 1 if URL_RE.search(text) else 0
    features['num_emails'] = 1 if EMAIL_RE.search(text) else 0
    features['num_currency'] = len(CURRENCY_RE.findall(text))
    features['num_digits'] = sum(c.isdigit() for c in text)
    features['num_phone_like'] = 1 if PHONE_RE.search(text) else 0

    text_low = text.lower()
    features['suspicious_kw_count'] = sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in text_low)

    features['unique_char_frac'] = len(set(text_low)) / (len(text_low)+1)

    return features


# LOAD DATASET
df = pd.read_csv("fake_job_postings.csv", on_bad_lines='skip', engine='python')
print("Loaded:", df.shape)

df["text"] = (
    df["title"].fillna("") + " " +
    df["company_profile"].fillna("") + " " +
    df["description"].fillna("") + " " +
    df["requirements"].fillna("")
)

df["text_clean"] = df["text"].apply(clean_text)
df["label"] = df["fraudulent"].fillna(0).astype(int)

# numeric feature matrix
num_feats = [
    'char_len','word_count','avg_word_len','num_urls','num_emails','num_currency',
    'num_digits','num_phone_like','suspicious_kw_count','unique_char_frac'
]

feat_df = pd.DataFrame([extract_numeric_features(t) for t in df['text_clean']])[num_feats]

# TRAIN-TEST SPLIT
X_text = df['text_clean']
y = df['label']

X_train_t, X_test_t, y_train, y_test, feat_train, feat_test = train_test_split(
    X_text, y, feat_df, test_size=0.2, random_state=42, stratify=y
)

# TF-IDF
tfidf = TfidfVectorizer(stop_words='english', max_features=10000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train_t)
X_test_tfidf = tfidf.transform(X_test_t)

# scaler for numeric features
scaler = StandardScaler(with_mean=False)
scaler.fit(feat_train.values)

feat_train_scaled = scaler.transform(feat_train.values)
feat_test_scaled = scaler.transform(feat_test.values)

# combine vec+scaler
X_train_final = hstack([X_train_tfidf, feat_train_scaled])
X_test_final = hstack([X_test_tfidf, feat_test_scaled])

print("Preprocessing and train/test split completed")

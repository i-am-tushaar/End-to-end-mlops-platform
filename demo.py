import re
import string
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import warnings
warnings.filterwarnings("ignore")


# ==========================
# Text Preprocessing
# ==========================
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    text = " ".join([
        lemmatizer.lemmatize(word)
        for word in text.split()
        if word not in stop_words
    ])

    return text.strip()


# ==========================
# Load Data
# ==========================
def load_data(filepath):
    df = pd.read_csv(filepath)

    df["review"] = df["review"].astype(str).apply(preprocess_text)

    df = df[df["sentiment"].isin(["positive", "negative"])]
    df["sentiment"] = df["sentiment"].map({"negative": 0, "positive": 1})

    X = df["review"]
    y = df["sentiment"]

    return train_test_split(X, y, test_size=0.2, random_state=42)


# ==========================
# Train Model (Pipeline)
# ==========================
def train_model(X_train, X_test, y_train, y_test):

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LogisticRegression())
    ])

    param_grid = {
        "tfidf__max_features": [3000, 5000],
        "clf__C": [0.1, 1, 10],
        "clf__penalty": ["l1", "l2"],
        "clf__solver": ["liblinear"]
    }

    grid = GridSearchCV(pipeline, param_grid, cv=5, scoring="f1", n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    # Evaluation
    y_pred = best_model.predict(X_test)

    print("\nModel Performance:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))

    print("\nBest Params:", grid.best_params_)

    return best_model


# ==========================
# Prediction
# ==========================
def predict_sentiment(text, model):
    processed = preprocess_text(text)
    pred = model.predict([processed])[0]

    return "Positive" if pred == 1 else "Negative"


# ==========================
# Main
# ==========================
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data("notebooks/data.csv")

    model = train_model(X_train, X_test, y_train, y_test)

    # Example prediction
    sample_text = "The movie was boring and waste of time"
    result = predict_sentiment(sample_text, model)

    print(f"\nText: {sample_text}")
    print(f"Predicted Sentiment: {result}")
import pandas as pd
import re
import emoji
import nltk

from prefect import task, flow
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# =========================
# TEXT PREPROCESSING
# =========================
stemmer = PorterStemmer()
stop_words = set(stopwords.words("english")) - {"not", "no", "nor"}

def clean_text(text):
    text = str(text).lower()
    text = emoji.replace_emoji(text, replace="")
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z\s]", "", text)

    words = [
        stemmer.stem(word)
        for word in text.split()
        if word not in stop_words
    ]
    return " ".join(words)

# =========================
# PREFECT TASKS
# =========================

@task
def load_data(file_path):
    """Load dataset"""
    return pd.read_csv("data.csv")


@task
def preprocess_data(df):
    """Clean text & map sentiment"""
    df = df[["Review text", "Ratings"]].dropna().drop_duplicates()

    df["clean_review"] = df["Review text"].apply(clean_text)

    # Binary sentiment
    df["sentiment"] = df["Ratings"].apply(lambda r: 0 if r <= 2 else 1)

    return df["clean_review"], df["sentiment"]


@task
def split_train_test(X, y, test_size=0.3):
    """Split into train and test"""
    return train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=42
    )


@task
def vectorize_text(X_train, X_test):
    """TF-IDF Vectorization"""
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        min_df=3,
        max_df=0.9,
        stop_words="english"
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    return X_train_vec, X_test_vec


@task
def train_model(X_train_vec, y_train):
    """Train Logistic Regression model"""
    model = LogisticRegression(
        C=0.5,
        max_iter=1000,
        class_weight="balanced"
    )
    model.fit(X_train_vec, y_train)
    return model


@task
def evaluate_model(model, X_train_vec, y_train, X_test_vec, y_test):
    """Evaluate model using F1-score"""
    train_pred = model.predict(X_train_vec)
    test_pred = model.predict(X_test_vec)

    train_f1 = f1_score(y_train, train_pred, average="macro")
    test_f1 = f1_score(y_test, test_pred, average="macro")

    return train_f1, test_f1


# =========================
# PREFECT FLOW
# =========================
@flow(name="Flipkart Sentiment Analysis Flow (Logistic)")
def sentiment_workflow():
    DATA_PATH = "data.csv"

    df = load_data(DATA_PATH)
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    X_train_vec, X_test_vec = vectorize_text(X_train, X_test)
    model = train_model(X_train_vec, y_train)

    train_f1, test_f1 = evaluate_model(
        model, X_train_vec, y_train, X_test_vec, y_test
    )

    print("Train F1 Score:", round(train_f1, 4))
    print("Test F1 Score :", round(test_f1, 4))


# =========================
# DEPLOYMENT
# =========================
if __name__ == "__main__":
    sentiment_workflow.serve(
        name="flipkart-sentiment-logistic-deployment",
        cron="*/5 * * * *"   # every 5 minutes
    )

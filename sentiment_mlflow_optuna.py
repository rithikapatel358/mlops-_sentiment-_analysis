import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import optuna
from optuna.integration.mlflow import MLflowCallback

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score

import joblib
import time
import os
import re
import warnings

warnings.filterwarnings("ignore")
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("data.csv")

# Keep only required columns
df = df[["Review text", "Ratings"]]
df = df.dropna().drop_duplicates()

# =========================
# TEXT CLEANING
# =========================

import re
import emoji
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

# Keep negation words (important for sentiment)
stop_words = set(stopwords.words("english")) - {"not", "no", "nor"}

def clean_text(text):
    # Convert to string & lowercase
    text = str(text).lower()

    # Remove emojis
    text = emoji.replace_emoji(text, replace="")

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove punctuation & numbers
    text = re.sub(r"[^a-z\s]", "", text)

    # Tokenization
    words = text.split()

    # Remove stopwords & apply stemming
    words = [
        stemmer.stem(word)
        for word in words
        if word not in stop_words
    ]

    # Join back to string
    text = " ".join(words)

    return text


df["clean_review"] = df["Review text"].apply(clean_text)

# =========================
# SENTIMENT MAPPING
# =========================
# 1–2 → Negative
# 3–5 → Positive

def rating_to_sentiment(r):
    if r <= 2:
        return 0
    else:
        return 1

df["sentiment"] = df["Ratings"].apply(rating_to_sentiment)

X = df["clean_review"]
y = df["sentiment"]

# =========================
# TRAIN–TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    stratify=y,
    random_state=42
)

# =========================
# PIPELINE
# =========================
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("model", LogisticRegression())
])

# =========================
# OPTUNA OBJECTIVES
# =========================
def objective_lr(trial):
    pipeline.set_params(
        tfidf__ngram_range=(1,2),
        tfidf__max_features=trial.suggest_int("tfidf__max_features", 2000, 5000, step=500),
        tfidf__min_df=trial.suggest_int("tfidf__min_df", 3, 7),
        tfidf__max_df=trial.suggest_float("tfidf__max_df", 0.7, 0.9),
        model=LogisticRegression(
            C=trial.suggest_float("model__C", 0.01, 2.0, log=True),
            max_iter=1000,
            class_weight="balanced"
        )

    )

    skf = StratifiedKFold(n_splits=5, shuffle=True)
    return cross_val_score(
        pipeline, X_train, y_train,
        scoring="f1_macro",
        cv=skf
    ).mean()


def objective_nb(trial):
    pipeline.set_params(
        tfidf__ngram_range=(1,2),
        tfidf__max_features=trial.suggest_int("tfidf__max_features", 3000, 10000, step=1000),
        tfidf__min_df=trial.suggest_int("tfidf__min_df", 3, 7),
        tfidf__max_df=trial.suggest_float("tfidf__max_df", 0.7, 0.9),
        model=MultinomialNB(
            alpha=trial.suggest_float("model__alpha", 0.01, 1.0, log=True)
        )
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True)
    return cross_val_score(
        pipeline, X_train, y_train,
        scoring="f1_macro",
        cv=skf
    ).mean()


def objective_svm(trial):
    pipeline.set_params(
        tfidf__ngram_range=(1,2),
        tfidf__max_features=trial.suggest_int("tfidf__max_features", 3000, 10000, step=1000),
        tfidf__min_df=trial.suggest_int("tfidf__min_df", 3, 7),
        tfidf__max_df=trial.suggest_float("tfidf__max_df", 0.7, 0.9),
        model=LinearSVC(
            C=trial.suggest_float("model__C", 0.01, 2.0, log=True),
            class_weight="balanced"
        )

    )

    skf = StratifiedKFold(n_splits=5, shuffle=True)
    return cross_val_score(
        pipeline, X_train, y_train,
        scoring="f1_macro",
        cv=skf
    ).mean()

# =========================
# MODEL MAP
# =========================
objectives = {
    "LogisticRegression": objective_lr,
    "NaiveBayes": objective_nb,
    "LinearSVM": objective_svm
}

# =========================
# MLFLOW EXPERIMENT
# =========================
mlflow.set_experiment("FLIPKART_SENTIMENT_ANALYSIS")

results = {}

# =========================
# TRAINING LOOP
# =========================
for model_name, obj_fn in objectives.items():
    print(f"\n--- Optimizing {model_name} ---")

    mlflow_cb = MLflowCallback(
        metric_name="cv_f1_macro",
        mlflow_kwargs={"nested": True}
    )

    study = optuna.create_study(direction="maximize")

    start_time = time.time()
    study.optimize(obj_fn, n_trials=20, callbacks=[mlflow_cb])
    fit_time = time.time() - start_time

    best_params = study.best_params
    best_cv_f1 = study.best_value

    pipeline.set_params(**best_params)
    pipeline.fit(X_train, y_train)

    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    train_f1 = f1_score(y_train, y_train_pred, average="macro")
    test_f1 = f1_score(y_test, y_test_pred, average="macro")

    model_path = f"{model_name}_model.pkl"
    joblib.dump(pipeline, model_path)
    model_size = os.path.getsize(model_path)

    mlflow.log_param("model", model_name)
    for k, v in best_params.items():
        mlflow.log_param(k, v)

    mlflow.log_metric("cv_f1_macro", best_cv_f1)
    mlflow.log_metric("train_f1", train_f1)
    mlflow.log_metric("test_f1", test_f1)
    mlflow.log_metric("fit_time", fit_time)
    mlflow.log_metric("model_size_bytes", model_size)

    mlflow.sklearn.log_model(
        pipeline,
        artifact_path="model",
        registered_model_name="FlipkartSentimentModel"
    )

    os.remove(model_path)

    results[model_name] = {
        "cv_f1": best_cv_f1,
        "train_f1": train_f1,
        "test_f1": test_f1
    }

    mlflow.end_run()

# =========================
# SUMMARY
# =========================
print("\n--- FINAL SUMMARY ---")
for model, res in results.items():
    print(
        f"{model} | CV F1={res['cv_f1']:.4f} | "
        f"Train F1={res['train_f1']:.4f} | "
        f"Test F1={res['test_f1']:.4f}"
    )

"""Classify financial news as bearish, bullish or neutral.

Pipeline: Pandas to load, clean and explore the data; scikit-learn for a
TF-IDF + Logistic Regression model, compared against a Naive Bayes baseline.

Usage:
    python classify.py                             # train, evaluate, demo
    python classify.py "Profit falls 20% on weak demand"  # also score your own headlines
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

DATA_DIR = Path(__file__).parent / "data"
LABELS = {0: "bearish", 1: "bullish", 2: "neutral"}


def load(name):
    df = pd.read_csv(DATA_DIR / name)
    df["sentiment"] = df["label"].map(LABELS)
    df["clean"] = (
        df["text"]
        .str.replace(r"https?://\S+", " ", regex=True)  # links carry no sentiment
        .str.replace(r"\$[A-Za-z.]+", " TICKER ", regex=True)  # $AAPL, $BRK.B -> TICKER
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    return df.drop_duplicates(subset="clean")


def explore(df):
    print("=== Training data ===")
    print(f"{len(df):,} headlines\n")
    print(df["sentiment"].value_counts(normalize=True).mul(100).round(1).astype(str).add("%").to_string(), "\n")
    words = df["clean"].str.split().str.len()
    print("Average words per headline by class:")
    print(words.groupby(df["sentiment"]).mean().round(1).to_string(), "\n")


def build_model(classifier):
    # Unigrams + bigrams so phrases like "cuts guidance" or "beats estimates" count.
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        classifier,
    )


def top_words(model, n=8):
    vec, clf = model.named_steps["tfidfvectorizer"], model.named_steps["logisticregression"]
    vocab = vec.get_feature_names_out()
    print("=== Most indicative words per class ===")
    for label, weights in zip(clf.classes_, clf.coef_):
        best = weights.argsort()[::-1][:n]
        print(f"{label:>8}: {', '.join(vocab[best])}")
    print()


def main():
    train, test = load("sent_train.csv"), load("sent_valid.csv")
    explore(train)

    candidates = {
        "Naive Bayes (baseline)": build_model(MultinomialNB()),
        # class_weight="balanced" stops the model from just predicting the majority class (neutral).
        "Logistic Regression": build_model(LogisticRegression(max_iter=1000, C=5, class_weight="balanced")),
    }

    # The CSV is sorted by label, so shuffle before splitting into folds.
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print("=== 5-fold cross-validation, macro F1 (training set) ===")
    for name, model in candidates.items():
        scores = cross_val_score(model, train["clean"], train["sentiment"], cv=folds, scoring="f1_macro")
        print(f"{name:<24} {scores.mean():.3f} ± {scores.std():.3f}")
    print()

    model = candidates["Logistic Regression"].fit(train["clean"], train["sentiment"])
    preds = model.predict(test["clean"])

    print(f"=== Held-out test set ({len(test):,} headlines) ===")
    print(classification_report(test["sentiment"], preds, digits=3))
    labels = model.classes_
    cm = pd.DataFrame(confusion_matrix(test["sentiment"], preds, labels=labels), index=labels, columns=labels)
    print("Confusion matrix (rows = true, columns = predicted):")
    print(cm, "\n")

    top_words(model)

    examples = sys.argv[1:] or [
        "Company beats earnings expectations and raises full-year outlook",
        "Shares slump after regulator fines bank over compliance failures",
        "Firm to hold quarterly earnings call on Tuesday",
    ]
    print("=== Predictions on new headlines ===")
    cleaned = pd.Series(examples).str.replace(r"\$[A-Za-z.]+", " TICKER ", regex=True)
    probs = pd.DataFrame(model.predict_proba(cleaned), columns=model.classes_, index=examples)
    probs["prediction"] = model.predict(cleaned)
    print(probs.round(2).to_string())


if __name__ == "__main__":
    main()

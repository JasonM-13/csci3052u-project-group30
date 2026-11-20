import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


def read_jsonl(path):
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def macro_f1(actual, predicted):
    return f1_score(
        actual,
        predicted,
        labels=["smile", "ok", "cry"],
        average="macro",
        zero_division=0,
    )


# The script is in eda/, so go up to the repo root first.
folder = Path(__file__).resolve().parents[1] / "data" / "sentiment"
train = read_jsonl(folder / "train.jsonl")
valid = read_jsonl(folder / "valid.jsonl")
test = read_jsonl(folder / "test.jsonl")

train_text = [row["sentence"] for row in train]
train_labels = [row["label"] for row in train]
valid_text = [row["sentence"] for row in valid]
valid_labels = [row["label"] for row in valid]
test_labels = {row["id"]: row["label"] for row in test}

# 1. Feature ablation. Change the character features and keep C fixed.
feature_sets = {
    "Unigrams only": (1, 1),
    "Unigrams and bigrams": (1, 2),
    "Bigrams only": (2, 2),
}

print("FEATURE ABLATION (validation macro F1)")
for name, ngram_range in feature_sets.items():
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=ngram_range)
    x_train = vectorizer.fit_transform(train_text)
    x_valid = vectorizer.transform(valid_text)

    model = LogisticRegression(C=3, max_iter=1000, random_state=42)
    model.fit(x_train, train_labels)

    score = macro_f1(valid_labels, model.predict(x_valid))
    print(f"{name}: {score:.4f}")

    # Save test predictions from the documented bigram-only baseline.
    if ngram_range == (2, 2):
        x_test = vectorizer.transform(row["sentence"] for row in test)
        lr_test_predictions = dict(zip(test_labels, model.predict(x_test)))


# 2. Robustness. Score the same test predictions by review length.
short_ids = [row["id"] for row in test if len(row["sentence"]) <= 512]
long_ids = [row["id"] for row in test if len(row["sentence"]) > 512]

predictions = {"Logistic regression": lr_test_predictions}
bert_files = {
    "Multilingual BERT": "bert-base-multilingual-cased_test_preds.jsonl",
    "Chinese BERT": "bert-base-chinese_test_preds.jsonl",
    "Cantonese BERT": "hon9kon9ize_bert-base-cantonese_test_preds.jsonl",
}

for name, filename in bert_files.items():
    rows = read_jsonl(folder / "predictions" / filename)
    saved_labels = {row["id"]: row["label"] for row in rows}
    if len(rows) != len(test) or saved_labels != test_labels:
        raise ValueError(f"{filename} does not match test.jsonl")

    predictions[name] = {row["id"]: row["pred"] for row in rows}

print("\nREVIEW LENGTH (test macro F1)")
print(f"{'Model':<24} {'<=512 chars':>12} {'>512 chars':>12}")
print(f"{'Number of reviews':<24} {len(short_ids):>12} {len(long_ids):>12}")

for name, model_predictions in predictions.items():
    short_f1 = macro_f1(
        [test_labels[i] for i in short_ids],
        [model_predictions[i] for i in short_ids],
    )
    long_f1 = macro_f1(
        [test_labels[i] for i in long_ids],
        [model_predictions[i] for i in long_ids],
    )
    print(f"{name:<24} {short_f1:>12.4f} {long_f1:>12.4f}")
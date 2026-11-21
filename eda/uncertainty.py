import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

LABELS = ["smile", "ok", "cry"]
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 42
LR_SEEDS = [0, 1, 2, 3, 4]


def read_jsonl(path):
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def macro_f1(actual, predicted):
    return f1_score(
        actual, predicted, labels=LABELS, average="macro", zero_division=0
    )


# The script is in eda/, so go up to the repo root first.
folder = Path(__file__).resolve().parents[1] / "data" / "sentiment"
train = read_jsonl(folder / "train.jsonl")
test = read_jsonl(folder / "test.jsonl")

train_text = [row["sentence"] for row in train]
train_labels = [row["label"] for row in train]
test_text = [row["sentence"] for row in test]
test_labels = np.array([row["label"] for row in test])
test_ids = [row["id"] for row in test]

# 1. Logistic regression over several seeds (C=3, char bigrams, as in lr_baseline.py).
vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 2))
x_train = vectorizer.fit_transform(train_text)
x_test = vectorizer.transform(test_text)

print("LOGISTIC REGRESSION OVER SEEDS (test macro F1)")
lr_scores = []
for seed in LR_SEEDS:
    model = LogisticRegression(C=3, max_iter=1000, random_state=seed)
    model.fit(x_train, train_labels)
    predictions = model.predict(x_test)
    lr_scores.append(macro_f1(test_labels, predictions))
    print(f"seed {seed}: {lr_scores[-1]:.4f}")
print(f"mean {np.mean(lr_scores):.4f}, SD {np.std(lr_scores, ddof=1):.4f}")

# Keep the last LR run's predictions (all seeds should be identical).
predictions_by_model = {"Logistic regression": np.array(predictions)}

# 2. Load the saved BERT test predictions (one training seed each, seed 42).
bert_files = {
    "Multilingual BERT": "bert-base-multilingual-cased_test_preds.jsonl",
    "Chinese BERT": "bert-base-chinese_test_preds.jsonl",
    "Cantonese BERT": "hon9kon9ize_bert-base-cantonese_test_preds.jsonl",
}
for name, filename in bert_files.items():
    rows = read_jsonl(folder / "predictions" / filename)
    by_id = {row["id"]: row for row in rows}
    if len(rows) != len(test) or any(
        by_id[i]["label"] != label for i, label in zip(test_ids, test_labels)
    ):
        raise ValueError(f"{filename} does not match test.jsonl")
    predictions_by_model[name] = np.array([by_id[i]["pred"] for i in test_ids])

# 3. Paired bootstrap: resample the 999 test reviews with replacement.
# Every model is scored on the same resample, so differences are paired.
rng = np.random.default_rng(BOOTSTRAP_SEED)
n = len(test_labels)
samples = rng.integers(0, n, size=(N_BOOTSTRAP, n))

boot = {name: np.empty(N_BOOTSTRAP) for name in predictions_by_model}
for b, index in enumerate(samples):
    for name, predicted in predictions_by_model.items():
        boot[name][b] = macro_f1(test_labels[index], predicted[index])


def interval(values):
    return np.percentile(values, 2.5), np.percentile(values, 97.5)


print(f"\nBOOTSTRAP 95% CI ({N_BOOTSTRAP} resamples, test macro F1)")
print(f"{'Model':<22} {'Observed':>9} {'Lower':>8} {'Upper':>8}")
for name, predicted in predictions_by_model.items():
    low, high = interval(boot[name])
    print(f"{name:<22} {macro_f1(test_labels, predicted):>9.4f} {low:>8.4f} {high:>8.4f}")

comparisons = [
    ("Cantonese BERT", "Logistic regression"),
    ("Chinese BERT", "Logistic regression"),
    ("Multilingual BERT", "Logistic regression"),
    ("Cantonese BERT", "Chinese BERT"),
    ("Cantonese BERT", "Multilingual BERT"),
]

print("\nPAIRED DIFFERENCES (A minus B, test macro F1)")
print(f"{'A vs B':<40} {'Diff':>7} {'Lower':>8} {'Upper':>8} {'Zero in CI':>11}")
for a, b in comparisons:
    observed = macro_f1(test_labels, predictions_by_model[a]) - macro_f1(
        test_labels, predictions_by_model[b]
    )
    low, high = interval(boot[a] - boot[b])
    contains_zero = "yes" if low <= 0 <= high else "no"
    print(f"{a + ' vs ' + b:<40} {observed:>7.4f} {low:>8.4f} {high:>8.4f} {contains_zero:>11}")

# 4. Exact McNemar test: only reviews where exactly one model is right count.
print("\nMCNEMAR TEST (exact, two-sided)")
print(f"{'A vs B':<40} {'A only':>7} {'B only':>7} {'p-value':>9}")
for a, b in comparisons:
    a_right = predictions_by_model[a] == test_labels
    b_right = predictions_by_model[b] == test_labels
    a_only = int(np.sum(a_right & ~b_right))
    b_only = int(np.sum(~a_right & b_right))
    p_value = binomtest(a_only, a_only + b_only, 0.5).pvalue
    print(f"{a + ' vs ' + b:<40} {a_only:>7} {b_only:>7} {p_value:>9.4f}")

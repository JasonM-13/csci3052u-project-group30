# Plots and extra numbers for M4 section 3.2 and table 3.1.
# Reuses the exact setup from lr_baseline.py (char bigram TF-IDF, LogisticRegression, max_iter=1000,
# random_state=42). Makes two PNGs in docs/: c_sweep.png and confusion_matrix.png.
# Also prints train macro F1 and fit time for the 3.1 table. Run from data/sentiment.
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, ConfusionMatrixDisplay
from load_data import load_dataset, validate_labels

LABELS = ["smile", "ok", "cry"]
BEST_C = 3.0
C_VALUES = [0.0001, 0.001, 0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0, 1000.0]

train = load_dataset("train.jsonl"); validate_labels(train)
valid = load_dataset("valid.jsonl"); validate_labels(valid)
train_texts = [r["sentence"] for r in train]; train_labels = [r["label"] for r in train]
valid_texts = [r["sentence"] for r in valid]; valid_labels = [r["label"] for r in valid]

vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 2))
X_train = vectorizer.fit_transform(train_texts)
X_valid = vectorizer.transform(valid_texts)

# plot 1: C sweep (full range, so the underfit end shows)
scores = []
for c in C_VALUES:
    m = LogisticRegression(C=c, max_iter=1000, random_state=42).fit(X_train, train_labels)
    f1 = f1_score(valid_labels, m.predict(X_valid), labels=LABELS, average="macro")
    scores.append(f1)
    print(f"C={c}\tvalid macro F1={f1:.4f}")

plt.figure(figsize=(6, 4))
plt.semilogx(C_VALUES, scores, marker="o")
plt.axvline(BEST_C, linestyle="--", color="grey", label=f"chosen C={BEST_C:g}")
plt.xlabel("C (inverse regularisation strength, log scale)")
plt.ylabel("validation macro F1")
plt.title("Logistic regression, char bigram TF-IDF")
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig("../../docs/c_sweep.png", dpi=150)

# plot 2: confusion matrix at the chosen C, plus train F1 and fit time for table 3.1
t0 = time.time()
model = LogisticRegression(C=BEST_C, max_iter=1000, random_state=42).fit(X_train, train_labels)
fit_seconds = time.time() - t0
valid_pred = model.predict(X_valid)
train_f1 = f1_score(train_labels, model.predict(X_train), labels=LABELS, average="macro")
valid_f1 = f1_score(valid_labels, valid_pred, labels=LABELS, average="macro")
print(f"C={BEST_C:g}: train macro F1={train_f1:.4f}  valid macro F1={valid_f1:.4f}  fit time={fit_seconds:.1f}s")

fig, ax = plt.subplots(figsize=(5, 4.5))
ConfusionMatrixDisplay.from_predictions(valid_labels, valid_pred, labels=LABELS, ax=ax, colorbar=False)
ax.set_title(f"Validation confusion matrix, C={BEST_C:g}")
plt.tight_layout()
plt.savefig("../../docs/confusion_matrix.png", dpi=150)
print("saved docs/c_sweep.png and docs/confusion_matrix.png")

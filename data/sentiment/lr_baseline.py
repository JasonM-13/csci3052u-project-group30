from load_data import load_dataset, validate_labels
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, classification_report

train = load_dataset("train.jsonl")
validate_labels(train)

valid = load_dataset("valid.jsonl")
validate_labels(valid)

train_texts = [r["sentence"] for r in train]
train_labels = [r["label"] for r in train]

valid_texts = [i["sentence"] for i in valid]
valid_label = [i["label"] for i in valid]

vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 2))
train_vector = vectorizer.fit_transform(train_texts)
valid_vector = vectorizer.transform(valid_texts)

print(train_vector.shape)
# Swept C from 0.0001 to 1000 first. Below 0.1 the model underfit (macro F1 was at 0.69 ),
# above 10 there isn't any meaningful gain (10 vs 1000 differed by one review), so we left the highest range to be 10
c_values = [0.1, 0.3, 1.0, 3.0, 10.0]
best_score = 0
best_c = 0
for c in c_values:

    model = LogisticRegression(C=c, max_iter=1000, random_state=42)
    model.fit(train_vector, train_labels)
    valid_pred = model.predict(valid_vector)
    macro_f1 = f1_score(
        valid_label, valid_pred, labels=["smile", "ok", "cry"], average="macro"
    )
    print(f"regularization strength: {c}\t Valid macro F1: {macro_f1:.4f}")
    if macro_f1 > best_score:
        best_score = macro_f1
        best_c = c
print(f"best valid score: {best_score:.4f}")
print(f"best C: {best_c}")

model = LogisticRegression(C=best_c, max_iter=1000, random_state=42)
model.fit(train_vector, train_labels)
valid_pred = model.predict(valid_vector)
macro_f1 = f1_score(
    valid_label, valid_pred, labels=["smile", "ok", "cry"], average="macro"
)


valid_set = set(valid_texts)
train_set = set(train_texts)

overlap = valid_set.intersection(train_set)

print(f"overlap: {len(overlap)}")
print(f"Macro F1: {macro_f1:.4f}")
print(classification_report(valid_label, valid_pred))

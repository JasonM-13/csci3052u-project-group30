print("Starting naive baseline...")

from collections import Counter
from load_data import load_dataset, validate_labels
from sklearn.metrics import f1_score

train_data = load_dataset("train.jsonl")
valid_data = load_dataset("valid.jsonl")

print("Data loaded successfully!")
print("Training records:", len(train_data))
print("Validation records:", len(valid_data))

validate_labels(train_data)
validate_labels(valid_data)


train_labels = [record["label"] for record in train_data]
label_counts = Counter(train_labels)

majority_label = label_counts.most_common(1)[0][0]

print("Training label counts:", label_counts)
print("Naive baseline prediction:", majority_label)

# Predict the same label for every validation example
y_true = [record["label"] for record in valid_data]
y_pred = [majority_label] * len(valid_data)

macro_f1 = f1_score(
    y_true,
    y_pred,
    labels=["smile", "ok", "cry"],
    average="macro",
    zero_division=0
)

print(f"Validation macro F1: {macro_f1:.4f}")

import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.metrics import f1_score, classification_report
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from load_data import load_dataset, validate_labels

# Run from data/sentiment:  python bert_finetune.py MODEL_NAME [--smoke] [--test]
#   --smoke  tiny run (200 train reviews, 100 valid, 1 epoch) to check the script works
#   --test   also score the test set after training
model_name = (
    sys.argv[1]
    if len(sys.argv) > 1 and not sys.argv[1].startswith("--")
    else "bert-base-chinese"
)
SMOKE = "--smoke" in sys.argv
RUN_TEST = "--test" in sys.argv

# Same settings as the CantoNLU paper. The paper does not report max length or seed, those two are ours.
LEARNING_RATE = 2e-5
BATCH_SIZE = 16
EPOCHS = 3
MAX_LENGTH = 512
SEED = 42
for word in sys.argv:
    if word.startswith("--seed="):
        SEED = int(word.split("=")[1])
train = load_dataset("train.jsonl")
validate_labels(train)

valid = load_dataset("valid.jsonl")
validate_labels(valid)

if SMOKE:
    train = train[:200]
    valid = valid[:100]
    EPOCHS = 1

train_texts = [r["sentence"] for r in train]
train_labels = [r["label"] for r in train]

valid_texts = [i["sentence"] for i in valid]
valid_label = [i["label"] for i in valid]

labels_to_id = {"smile": 2, "cry": 0, "ok": 1}
train_ids = [labels_to_id[l] for l in train_labels]
valid_ids = [labels_to_id[l] for l in valid_label]

tokenizer = AutoTokenizer.from_pretrained(model_name)

id_to_label = {0: "cry", 1: "ok", 2: "smile"}
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

# Tokenize: text -> list of vocabulary numbers. Anything past 512 pieces is cut off the end.
train_enc = tokenizer(train_texts, truncation=True, max_length=MAX_LENGTH)
valid_enc = tokenizer(valid_texts, truncation=True, max_length=MAX_LENGTH)

print(f"model: {model_name}  smoke: {SMOKE}  test: {RUN_TEST}")
print(f"train: {len(train_ids)}  valid: {len(valid_ids)}")
print(train_enc["input_ids"][0][:20])
print(tokenizer.decode(train_enc["input_ids"][0][:20]))
cut = sum(1 for ids in train_enc["input_ids"] if len(ids) == MAX_LENGTH)
print(f"train reviews cut off at {MAX_LENGTH} tokens: {cut} of {len(train_ids)}")


# Wraps the tokenized reviews and their label numbers so the trainer can ask for review number i.
class ReviewDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, label_ids):
        self.encodings = encodings
        self.label_ids = label_ids

    def __len__(self):
        return len(self.label_ids)

    def __getitem__(self, i):
        item = {key: values[i] for key, values in self.encodings.items()}
        item["labels"] = self.label_ids[i]
        return item


train_dataset = ReviewDataset(train_enc, train_ids)
valid_dataset = ReviewDataset(valid_enc, valid_ids)


# The model gives 3 scores per review. The prediction is the position of the biggest score.
def compute_metrics(eval_pred):
    scores, true_ids = eval_pred
    pred_ids = np.argmax(scores, axis=1)
    return {"macro_f1": f1_score(true_ids, pred_ids, labels=[0, 1, 2], average="macro")}


# Training settings. No checkpoints are saved and no "best epoch" is picked:
# the model after epoch 3 is the model, so validation is never used to choose anything.
args = TrainingArguments(
    output_dir="bert_tmp",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=64,
    num_train_epochs=EPOCHS,
    seed=SEED,
    fp16=True,
    eval_strategy="epoch",
    save_strategy="no",
    logging_steps=50,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
)

t0 = time.time()
trainer.train()
train_seconds = time.time() - t0

os.makedirs("predictions/trainer_batch16", exist_ok=True)
out_prefix = (
    "predictions/trainer_batch16/"
    + model_name.replace("/", "_")
    + f"_seed{SEED}"
    + ("_smoke" if SMOKE else "")
)


def score_split(split_name, dataset, records, true_ids):
    scores = trainer.predict(dataset).predictions
    pred_ids = np.argmax(scores, axis=1)
    true_words = [id_to_label[i] for i in true_ids]
    pred_words = [id_to_label[int(i)] for i in pred_ids]

    macro_f1 = f1_score(
        true_words, pred_words, labels=["smile", "ok", "cry"], average="macro"
    )
    print(f"\n{split_name} macro F1: {macro_f1:.4f}")
    print(
        classification_report(
            true_words, pred_words, labels=["smile", "ok", "cry"], digits=4
        )
    )

    with open(f"{out_prefix}_{split_name}_preds.jsonl", "w", encoding="utf-8") as f:
        for record, t, p in zip(records, true_words, pred_words):
            f.write(json.dumps({"id": record.get("id"), "label": t, "pred": p}) + "\n")
    return macro_f1


results = {
    "model": model_name,
    "learning_rate": LEARNING_RATE,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "max_length": MAX_LENGTH,
    "seed": SEED,
    "fp16": True,
    "train_size": len(train_ids),
    "train_reviews_cut_at_max_length": cut,
    "train_seconds": round(train_seconds, 1),
    "valid_macro_f1_per_epoch": [
        round(log["eval_macro_f1"], 4)
        for log in trainer.state.log_history
        if "eval_macro_f1" in log
    ],
}

results["train_macro_f1"] = round(
    score_split("train", train_dataset, train, train_ids), 4
)
results["valid_macro_f1"] = round(
    score_split("valid", valid_dataset, valid, valid_ids), 4
)

if RUN_TEST:
    test = load_dataset("test.jsonl")
    validate_labels(test)
    test_ids = [labels_to_id[r["label"]] for r in test]
    test_enc = tokenizer(
        [r["sentence"] for r in test], truncation=True, max_length=MAX_LENGTH
    )
    results["test_macro_f1"] = round(
        score_split("test", ReviewDataset(test_enc, test_ids), test, test_ids), 4
    )

with open(f"{out_prefix}_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\ntrain time: {train_seconds:.0f}s")
print(json.dumps(results, indent=2))

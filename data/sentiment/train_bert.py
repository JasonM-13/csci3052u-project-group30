"""Fine-tune a BERT model for 3-class Cantonese sentiment (smile / ok / cry).

Mirrors the settings in predictions/*_results.json: lr 2e-5, effective batch 16,
3 epochs, max length 512, fp16, linear decay with no warmup, final-epoch model.

Run from data/sentiment, for example:
    python train_bert.py --model hon9kon9ize/bert-base-cantonese --seed 42
Writes three files per run to predictions/seeds/: results JSON plus valid and
test predictions (with softmax probabilities).
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
)

LABELS = ["smile", "ok", "cry"]
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, help="Hugging Face model name or local folder")
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument("--batch-size", type=int, default=8, help="reviews per forward pass")
parser.add_argument("--grad-accum", type=int, default=2, help="batches per optimizer step")
parser.add_argument("--eval-batch-size", type=int, default=32)
parser.add_argument("--max-length", type=int, default=512)
parser.add_argument("--no-fp16", action="store_true", help="train in full fp32")
parser.add_argument("--limit", type=int, default=0, help="use only the first N reviews per split (testing)")
parser.add_argument("--out-dir", default="predictions/seeds")
args = parser.parse_args()


def read_jsonl(path):
    with open(path, encoding="utf-8") as file:
        rows = [json.loads(line) for line in file if line.strip()]
    return rows[: args.limit] if args.limit else rows


def macro_f1(actual, predicted):
    return f1_score(actual, predicted, labels=LABELS, average="macro", zero_division=0)


set_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_fp16 = device.type == "cuda" and not args.no_fp16
print(f"device: {device}  fp16: {use_fp16}  torch {torch.__version__}  transformers {transformers.__version__}")

splits = {name: read_jsonl(f"{name}.jsonl") for name in ["train", "valid", "test"]}
tokenizer = AutoTokenizer.from_pretrained(args.model)


def encode(rows):
    texts = [row["sentence"] for row in rows]
    full_lengths = [len(ids) for ids in tokenizer(texts)["input_ids"]]
    encoded = tokenizer(texts, truncation=True, max_length=args.max_length)
    examples = [
        {"input_ids": ids, "attention_mask": mask, "labels": LABEL_TO_ID[row["label"]]}
        for ids, mask, row in zip(encoded["input_ids"], encoded["attention_mask"], rows)
    ]
    cut = sum(length > args.max_length for length in full_lengths)
    return examples, cut, full_lengths


data, cut_counts, token_lengths = {}, {}, {}
for name, rows in splits.items():
    data[name], cut_counts[name], token_lengths[name] = encode(rows)
print(f"reviews cut at {args.max_length} tokens: " + ", ".join(f"{k} {v}" for k, v in cut_counts.items()))


def collate(batch):
    return tokenizer.pad(batch, return_tensors="pt")


generator = torch.Generator().manual_seed(args.seed)
train_loader = DataLoader(data["train"], batch_size=args.batch_size, shuffle=True,
                          collate_fn=collate, generator=generator)


def eval_loader(name):
    return DataLoader(data[name], batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate)


model = AutoModelForSequenceClassification.from_pretrained(
    args.model,
    num_labels=len(LABELS),
    id2label=dict(enumerate(LABELS)),
    label2id=LABEL_TO_ID,
).to(device)

# Same optimizer and schedule as the Hugging Face Trainer defaults.
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
scheduler = get_linear_schedule_with_warmup(optimizer, 0, steps_per_epoch * args.epochs)
scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)


@torch.no_grad()
def predict(name):
    model.eval()
    probabilities = []
    for batch in eval_loader(name):
        batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_fp16):
            logits = model(**batch).logits
        probabilities.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    return np.concatenate(probabilities)


def score(name, probabilities):
    actual = [row["label"] for row in splits[name]]
    predicted = [LABELS[i] for i in probabilities.argmax(axis=1)]
    return macro_f1(actual, predicted)


if device.type == "cuda":
    torch.cuda.reset_peak_memory_stats()
start = time.time()
valid_per_epoch = []
for epoch in range(1, args.epochs + 1):
    model.train()
    optimizer.zero_grad()
    for step, batch in enumerate(train_loader, start=1):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_fp16):
            loss = model(**batch).loss / args.grad_accum
        scaler.scale(loss).backward()
        if step % args.grad_accum == 0 or step == len(train_loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
        if step % 100 == 0:
            print(f"  epoch {epoch} step {step}/{len(train_loader)} loss {loss.item() * args.grad_accum:.4f}", flush=True)
    valid_per_epoch.append(round(score("valid", predict("valid")), 4))
    print(f"epoch {epoch}: valid macro F1 {valid_per_epoch[-1]:.4f}  ({time.time() - start:.0f} s)", flush=True)
train_seconds = time.time() - start

# Final-epoch model, as in the original runs (no best-epoch selection).
probabilities = {name: predict(name) for name in ["train", "valid", "test"]}

out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)
stem = f"{args.model.replace('/', '_')}_seed{args.seed}"

for name in ["valid", "test"]:
    with open(out_dir / f"{stem}_{name}_preds.jsonl", "w", encoding="utf-8") as file:
        for row, probs in zip(splits[name], probabilities[name]):
            record = {
                "id": row["id"],
                "label": row["label"],
                "pred": LABELS[int(probs.argmax())],
                "probs": {label: round(float(p), 6) for label, p in zip(LABELS, probs)},
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

results = {
    "model": args.model,
    "learning_rate": args.lr,
    "batch_size": args.batch_size * args.grad_accum,
    "per_device_batch_size": args.batch_size,
    "grad_accum": args.grad_accum,
    "epochs": args.epochs,
    "max_length": args.max_length,
    "seed": args.seed,
    "fp16": use_fp16,
    "train_size": len(splits["train"]),
    "train_reviews_cut_at_max_length": cut_counts["train"],
    "test_reviews_cut_at_max_length": cut_counts["test"],
    "train_seconds": round(train_seconds, 1),
    "valid_macro_f1_per_epoch": valid_per_epoch,
    "train_macro_f1": round(score("train", probabilities["train"]), 4),
    "valid_macro_f1": round(score("valid", probabilities["valid"]), 4),
    "test_macro_f1": round(score("test", probabilities["test"]), 4),
    "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
    "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if device.type == "cuda" else None,
    "torch_version": torch.__version__,
    "transformers_version": transformers.__version__,
}
with open(out_dir / f"{stem}_results.json", "w", encoding="utf-8") as file:
    json.dump(results, file, indent=2)
with open(out_dir / f"{stem}_test_token_lengths.json", "w", encoding="utf-8") as file:
    json.dump({str(row["id"]): n for row, n in zip(splits["test"], token_lengths["test"])}, file)
print(json.dumps(results, indent=2))

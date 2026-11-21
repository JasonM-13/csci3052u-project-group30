#!/usr/bin/env bash
# Max-length ablation: Cantonese and Chinese BERT at 128 tokens, seeds 42-44.
# Same settings as run_bert_seeds.sh except --max-length 128.
# Run from data/sentiment with the BERT venv active, AFTER run_bert_seeds.sh is done:
#     bash run_bert_len128.sh
# Outputs go to predictions/seeds_len128/ so the 512-token results are never overwritten.
# Finished runs are skipped, so it is safe to stop (Ctrl+C) and start again.

MAX_LENGTH=128
OUT_DIR="predictions/seeds_len128"
MODELS=(
  "hon9kon9ize/bert-base-cantonese"
  "bert-base-chinese"
)
SEEDS=(42 43 44)

mkdir -p "${OUT_DIR}/logs"
for seed in "${SEEDS[@]}"; do
  for model in "${MODELS[@]}"; do
    stem="${model//\//_}_seed${seed}"
    if [ -f "${OUT_DIR}/${stem}_results.json" ]; then
      echo "skip  ${stem} (already done)"
      continue
    fi
    echo "start ${stem}  max_length ${MAX_LENGTH}  $(date +%H:%M:%S)"
    python train_bert.py --model "$model" --seed "$seed" \
      --max-length "$MAX_LENGTH" --out-dir "$OUT_DIR" "$@" \
      > "${OUT_DIR}/logs/${stem}.log" 2>&1
    if [ $? -ne 0 ]; then
      echo "FAILED ${stem}: see ${OUT_DIR}/logs/${stem}.log"
      exit 1
    fi
    grep -E '"(valid|test)_macro_f1"|train_seconds' "${OUT_DIR}/${stem}_results.json"
  done
done
echo "all runs done  $(date +%H:%M:%S)"

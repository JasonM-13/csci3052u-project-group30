#!/usr/bin/env bash
# Train 3 BERT models x 5 seeds. Run from data/sentiment with the BERT venv active:
#     bash run_bert_seeds.sh
# Finished runs are skipped, so it is safe to stop (Ctrl+C) and start again.
# Each run's console output goes to predictions/seeds/logs/.

MODELS=(
  "hon9kon9ize/bert-base-cantonese"
  "bert-base-chinese"
  "bert-base-multilingual-cased"
)
SEEDS=(42 43 44 45 46)

mkdir -p predictions/seeds/logs
for seed in "${SEEDS[@]}"; do
  for model in "${MODELS[@]}"; do
    stem="${model//\//_}_seed${seed}"
    if [ -f "predictions/seeds/${stem}_results.json" ]; then
      echo "skip  ${stem} (already done)"
      continue
    fi
    echo "start ${stem}  $(date +%H:%M:%S)"
    python train_bert.py --model "$model" --seed "$seed" "$@" \
      > "predictions/seeds/logs/${stem}.log" 2>&1
    if [ $? -ne 0 ]; then
      echo "FAILED ${stem}: see predictions/seeds/logs/${stem}.log"
      exit 1
    fi
    grep -E '"(valid|test)_macro_f1"|train_seconds' "predictions/seeds/${stem}_results.json"
  done
done
echo "all runs done  $(date +%H:%M:%S)"

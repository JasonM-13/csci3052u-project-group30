# Uncertainty analysis: method notes (issue #19, report section 4.4)

Reference for how the significance tests in section 4.4 were computed.
Code: `eda/welch_walkthrough.py` (step-by-step, checked against scipy) and
`eda/uncertainty.py` (all tables). Run both from the repo root with `python -u`.

## Data

- 3 BERT models (Cantonese, Chinese, multilingual) x 5 seeds (42-46), trained with
  `data/sentiment/train_bert.py` / `run_bert_seeds.sh`: lr 2e-5, effective batch 16
  (8 x 2 gradient accumulation), 3 epochs, max length 512 tokens, fp16, final-epoch model.
- Logistic regression (char bigrams, C=3): deterministic, test macro F1 0.7556 for every seed.
- Every test uses **test macro F1 only**, one value per seed. Validation scores are reported
  but never used to choose anything.

## p-value

The probability, **assuming the null hypothesis H0 is true**, of getting a result
**at least as extreme** as ours. Here H0 = "the two models have the same mean test macro F1".

- Small p (< 0.05): data this extreme would be rare if H0 were true, so **reject H0**.
- p >= 0.05: **fail to reject H0**. This does *not* prove the models are equal; with 5 seeds,
  small real differences can go undetected.
- p is **not** the probability that H0 (or our hypothesis) is true.
- Two-sided: both tails count, since a difference in either direction is of interest.

## Welch's t-test (BERT vs BERT)

Compares two means without assuming equal variances (multilingual BERT's SD is about 3x the others').

| Step | Formula | Cantonese - Chinese |
|---|---|---|
| Means | mean of the 5 scores | 0.79308, 0.79866 |
| SDs | s = sqrt( sum (x - mean)^2 / (n - 1) ) | 0.00449, 0.00381 |
| Difference | mean_A - mean_B | -0.00558 |
| Variance of each mean | v = s^2 / n | 4.03e-6, 2.91e-6 |
| Standard error of difference | sqrt(v_A + v_B) | 0.00263 |
| t | difference / SE | -2.119 |
| Degrees of freedom | (v_A + v_B)^2 / ( v_A^2/(n_A-1) + v_B^2/(n_B-1) ) | 7.80 |
| p (two-sided) | 2 x P(T <= -abs(t)) | 0.0678 |
| 95% CI | difference +/- t_crit(df) x SE | [-0.0117, +0.0005] |
| Decision (alpha 0.05) | | fail to reject H0 |

Notes:
- **n - 1 vs n:** n - 1 is used once, inside the SD (one degree of freedom is spent estimating
  the mean). The SE divides by n because averaging n values divides the variance by n.
- **Why squares add:** independent noise adds as variances, not SDs.
- **Degrees of freedom:** 8 (= 5 + 5 - 2) if both spreads are equal; falls toward 4 when one
  model's noise dominates (e.g. 5.1 and 4.8 for the comparisons with multilingual BERT).
  Fewer df means fatter tails, so larger p and wider CIs.

## One-sample t-test (BERT vs LR)

LR has no seed variance (SD = 0), so each BERT model's 5 scores are tested against LR's fixed
value 0.7556: SE = s / sqrt(n), t = (mean - 0.7556) / SE, df = n - 1 = 4.

## Results (test macro F1)

| Comparison | Test | Difference | 95% CI | p | Decision |
|---|---|---|---|---|---|
| Cantonese - Chinese | Welch | -0.006 | [-0.012, +0.001] | 0.068 | fail to reject |
| Cantonese - Multilingual | Welch | +0.009 | [-0.006, +0.024] | 0.19 | fail to reject |
| Chinese - Multilingual | Welch | +0.014 | [-0.001, +0.029] | 0.056 | fail to reject |
| Cantonese - LR | one-sample | +0.038 | [+0.032, +0.043] | < 0.001 | reject |
| Chinese - LR | one-sample | +0.043 | [+0.038, +0.048] | < 0.001 | reject |
| Multilingual - LR | one-sample | +0.029 | [+0.014, +0.044] | 0.006 | reject |

## Limitations

- 5 seeds give a rough variance estimate; t-tests on 5 values have low power.
- t-tests assume roughly normal per-seed scores, which can't be checked well with n = 5.
- Seed variance only; test-set sampling uncertainty is measured separately by the bootstrap
  in `eda/uncertainty.py` (about +/-0.024 for one run, larger than the seed SD).

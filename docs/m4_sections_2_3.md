
## 2. Theoretical Justification

### 2.1 Choice of model family

They used a single transformer for the script. We are going to test whether the BERT-based transformers are what's causing the SA score on macro F1. That's why we use logistic regression, to test whether it's a transformer thing or not.

(draft note: feels like this repeats the hypothesis, rewrite to lead with the model choice)

### 2.2 Loss function and optimiser
Cross-entropy is what was used for the loss function.
We can see the loss getting much smaller.

For the optimizer we used scikit-learn's default solver (draft, unfinished).

### 2.3 Bias-variance trade-off



### 2.4 Evaluation methodology



### 2.5 Alternatives considered


## 3. Preliminary Results

### 3.1 Results table

!!!!!!! copied from docs/baseline_table.md. Template also wants train F1 and training time columns, script does not print them yet.

| Model  | Features |  Split |  Macro F1 |
|--------|---------|--------|-----------|
| Majority Class | none | Validation| 0.1667 |
| Logistic Regression | char bigram TF-IDF, C=3 | Validation | 0.7705 |
| hon9kon9ize/bert-base-cantonese |  fine-tuned end to end    | test | 0.719 |

Run from data/sentiment: `python naive_baseline.py` and `python lr_baseline.py`.

### 3.2 Visualisations

!!!!!!! pending: C vs validation macro F1 (log x-axis) and confusion matrix, PNGs to go in docs/.

### 3.3 Key findings

!!!!!!! copied from docs/error_analysis.md (Differences between models + Patterns).

Majority scores 0.1667 because it predicts one class and gets F1 of 0 on the other two. LR at 0.7705 on the same split means char bigrams carry real sentiment signal with no word order at all.

ok had the smallest F1 score, 0.68. The most common pair was ok misclassified as smile and vice versa. 219 of the 231 misses involve ok on one side, only 12 are smile/cry swaps, which means the model separates the extremes well and the middle is where most of the misclassification is.

### 3.4 Error analysis

!!!!!!! copied from docs/error_analysis.md (Incorrect predictions + Hypotheses). Pair counts moved here from 2.2.

Out of the 999 reviews in validation the model got 231 reviews wrong after tuning it. The number of misses for each class is 70, 99, 62 that are smile, ok and cry respectively, so the class that was most commonly misclassified is ok.

```
label smile ->  predicted label ok: 65
label ok ->  predicted label smile: 65
label cry ->  predicted label ok: 55
label ok ->  predicted label cry: 34
label cry ->  predicted label smile: 7
label smile ->  predicted label cry: 5
```

Counts come from the `wrong` list and Counter at the bottom of `lr_baseline.py`.

Hypotheses to check when BERT runs:

1. Noisy ok labels. Mixed reviews tagged ok confuse any model. If BERT's ok F1 stays near 0.68, the labels are the ceiling, not the architecture.
2. Dilution. The verdict in the title is outvoted by the body. BERT reads in order and could weight the opening. Check: do the 12 swaps flip.
3. Negation. 唔 inside positive phrases like 唔差 fools a bigram counter. Check: same 12.
4. Truncation. BERT sees only 512, LR saw everything. Check: do BERT's misses cluster in the 12% of reviews over 512.

### 3.5 Limitations

!!!!!!! copied from docs/error_analysis.md (Discrepancies with the paper).

Our macro F1 of 0.7705 is on validation, the paper got 0.719 on test. Though our tuned logistic regression seems like it performed better than the paper's BERT model, it can't be concluded that a simple model can outperform the BERT architecture, because we used the validation dataset while they used the test dataset. Not only that, we ran the model many times and used the best C value for our dataset. Once we run the tuned model on the test dataset we can concretely conclude.

# Error analysis: logistic regression baseline

## Incorrect predictions

Out of the 999 reviews in validation the model got 231 reviews wrong after tuning it. The number of misses for each class is 70, 99, 62 that are smile, ok and cry respectively, so the class that was most commonly misclassified is ok.
The following is the output of each misclassified label with the predicted class. The code is in lr_baseline.py.

```
label smile ->  predicted label ok: 65
label ok ->  predicted label smile: 65
label cry ->  predicted label ok: 55
label ok ->  predicted label cry: 34
label cry ->  predicted label smile: 7
label smile ->  predicted label cry: 5
```

smile  65+ 5 = 70
ok 65+ 34 = 99
cry 55+ 7= 62

Counts come from the `wrong` list and Counter at the bottom of `lr_baseline.py`, run with `python lr_baseline.py` from the `data/sentiment/` folder.

## Patterns in incorrect predictions

ok had the smallest F1 score, 0.68.
The most common pair was ok misclassified as smile and vice versa.

219 of the 231 misses involve ok on one side, only 12 are smile/cry swaps, which means the model separates the extremes well and the middle is where most of the misclassification is.


Hypotheses to check when BERT runs:

1. Noisy ok labels. Mixed reviews tagged ok confuse any model. If BERT's ok F1 stays near 0.68, the labels are the ceiling, not the architecture.
2. Dilution. The verdict in the title is outvoted by the body. BERT reads in order and could weight the opening. Check: do the 12 swaps flip.
3. Negation. 唔 inside positive phrases like 唔差 fools a bigram counter. Check: same 12.
4. Truncation. BERT sees only 512, LR saw everything. Check: do BERT's misses cluster in the 12% of reviews over 512.

## Discrepancies with the paper

Our macro F1 of 0.7705 is on validation, the paper got 0.719.
Though our tuned logistic regression seems like it performed better than the paper's BERT model, it can't be concluded that a simple model can outperform the BERT architecture, because we used the validation dataset while they used the test dataset. Not only that, we ran the model many times and used the best C value for our dataset. Once we run the tuned model on the test dataset we can concretely conclude.

## Differences between models

Majority scores 0.1667 because it predicts one class and gets F1 of 0 on the other two. LR at 0.7705 on the same split means char bigrams carry real sentiment signal with no word order at all.

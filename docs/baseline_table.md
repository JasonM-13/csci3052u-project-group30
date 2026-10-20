| Model  | Features |  Split |  Macro F1 |
|--------|---------|--------|-----------| 
| Majority Class | none | Validation| 0.1667 | 
| Logistic Regression | char bigram TF-IDF, C=3 | Validation | 0.7705 | 
| hon9kon9ize/bert-base-cantonese |  fine-tuned end to end    | test | 0.719 |

This doesn't mean that logistic regression is better the data set used to find the macro F1 is validations while the paper used the test dataset that means the benchmark result's aren't a fair test we also ran the regression multiple times with different C values and we used the best values. The two become comparable once the tuned model is run on the test split, once.

Run from data/sentiment
`python naive_baseline.py`

`python lr_baseline.py`

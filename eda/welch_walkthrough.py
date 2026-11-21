"""Step-by-step reference for the seed-level significance tests in section 4.4.

Every number is computed by hand (numpy + the t-distribution) and printed with
its intermediate steps, then checked against scipy's built-in tests so the two
must agree. Run from the repo root:
    python -u eda/welch_walkthrough.py

Inputs: test macro F1 from data/sentiment/predictions/seeds/*_results.json
(3 BERT models x seeds 42-46, from train_bert.py) and the logistic regression
test score (char bigrams, C=3, deterministic so identical for every seed).
See docs/uncertainty_method.md for the explanation in words.
"""

import json
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

ALPHA = 0.05
SEEDS = [42, 43, 44, 45, 46]
MODELS = {
    "Cantonese BERT": "hon9kon9ize_bert-base-cantonese",
    "Chinese BERT": "bert-base-chinese",
    "Multilingual BERT": "bert-base-multilingual-cased",
}

folder = Path(__file__).resolve().parents[1] / "data" / "sentiment"


def read_jsonl(path):
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


# ---------------------------------------------------------------- inputs
scores = {}
for name, stem in MODELS.items():
    values = []
    for seed in SEEDS:
        path = folder / "predictions" / "seeds" / f"{stem}_seed{seed}_results.json"
        values.append(json.loads(path.read_text(encoding="utf-8"))["test_macro_f1"])
    scores[name] = np.array(values)

train = read_jsonl(folder / "train.jsonl")
test = read_jsonl(folder / "test.jsonl")
vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 2))
x_train = vectorizer.fit_transform([row["sentence"] for row in train])
model = LogisticRegression(C=3, max_iter=1000, random_state=42)
model.fit(x_train, [row["label"] for row in train])
lr_predictions = model.predict(vectorizer.transform([row["sentence"] for row in test]))
lr_score = f1_score([row["label"] for row in test], lr_predictions, average="macro")

print("INPUT: test macro F1 per seed", SEEDS)
for name, values in scores.items():
    print(f"  {name:<18} " + "  ".join(f"{v:.4f}" for v in values))
print(f"  {'Logistic regress.':<18} {lr_score:.4f} (same for every seed)\n")


def check(label, mine, scipy_value):
    assert np.isclose(mine, scipy_value, rtol=1e-6, atol=1e-10), f"{label}: {mine} vs {scipy_value}"


def decision(p):
    return "reject H0 (difference is significant)" if p < ALPHA else "fail to reject H0 (not significant)"


# ------------------------------------------------- Welch t-test, by hand
def welch(name_a, name_b):
    a, b = scores[name_a], scores[name_b]
    n_a, n_b = len(a), len(b)
    print("=" * 72)
    print(f"WELCH t-TEST: {name_a} - {name_b}")
    print("H0: both models have the same mean test macro F1\n")

    mean_a, mean_b = a.mean(), b.mean()
    sd_a, sd_b = a.std(ddof=1), b.std(ddof=1)  # ddof=1 -> divide by n-1
    print(f"1. Means:   mean_A = {mean_a:.5f}   mean_B = {mean_b:.5f}")
    print(f"   SDs (divide by n-1): s_A = {sd_a:.5f}   s_B = {sd_b:.5f}")

    diff = mean_a - mean_b
    print(f"2. Difference = mean_A - mean_B = {diff:+.5f}")

    v_a, v_b = sd_a**2 / n_a, sd_b**2 / n_b  # variance of each mean
    print(f"3. Variance of each mean:  v_A = s_A^2/n_A = {v_a:.3e}   v_B = s_B^2/n_B = {v_b:.3e}")

    se = np.sqrt(v_a + v_b)
    print(f"4. Standard error of the difference = sqrt(v_A + v_B) = {se:.5f}")

    t = diff / se
    print(f"5. t = difference / SE = {t:.3f}   (how many SEs apart the means are)")

    df = (v_a + v_b) ** 2 / (v_a**2 / (n_a - 1) + v_b**2 / (n_b - 1))
    print(f"6. Welch degrees of freedom = (v_A+v_B)^2 / (v_A^2/(n_A-1) + v_B^2/(n_B-1)) = {df:.2f}")
    print(f"   (would be {n_a + n_b - 2} if the spreads were equal; lower when they differ)")

    left = stats.t.cdf(-abs(t), df)
    p = 2 * left
    print(f"7. P(T <= -|t|) = {left:.4f}   P(T >= +|t|) = {left:.4f}")
    print(f"   two-sided p = {p:.4f}")

    t_crit = stats.t.ppf(1 - ALPHA / 2, df)
    low, high = diff - t_crit * se, diff + t_crit * se
    print(f"8. 95% CI = difference +/- t_crit * SE, t_crit({df:.2f} df) = {t_crit:.3f}")
    print(f"   = {diff:+.5f} +/- {t_crit * se:.5f} = [{low:+.4f}, {high:+.4f}]")
    print(f"9. Decision at alpha = {ALPHA}: {decision(p)}")

    reference = stats.ttest_ind(a, b, equal_var=False)
    ci = reference.confidence_interval()
    check("t", t, reference.statistic)
    check("df", df, reference.df)
    check("p", p, reference.pvalue)
    check("CI low", low, ci.low)
    check("CI high", high, ci.high)
    print("   checked: matches scipy.stats.ttest_ind(equal_var=False)\n")


# -------------------------------- one-sample t-test against LR, by hand
def one_sample(name):
    a = scores[name]
    n = len(a)
    print("=" * 72)
    print(f"ONE-SAMPLE t-TEST: {name} - logistic regression")
    print("LR has no seed variance (SD = 0), so we test whether the BERT seed")
    print(f"scores differ from LR's fixed value {lr_score:.4f}.")
    print("H0: the BERT model's mean test macro F1 equals LR's score\n")

    mean, sd = a.mean(), a.std(ddof=1)
    diff = mean - lr_score
    se = sd / np.sqrt(n)
    t = diff / se
    df = n - 1
    p = 2 * stats.t.cdf(-abs(t), df)
    t_crit = stats.t.ppf(1 - ALPHA / 2, df)
    low, high = diff - t_crit * se, diff + t_crit * se
    print(f"1. mean = {mean:.5f}   s = {sd:.5f}   difference = {diff:+.5f}")
    print(f"2. SE = s / sqrt(n) = {sd:.5f} / sqrt({n}) = {se:.5f}")
    print(f"3. t = {t:.2f}   df = n - 1 = {df}")
    print(f"4. two-sided p = {p:.2e}")
    print(f"5. 95% CI = {diff:+.5f} +/- {t_crit:.3f} * {se:.5f} = [{low:+.4f}, {high:+.4f}]")
    print(f"6. Decision at alpha = {ALPHA}: {decision(p)}")

    reference = stats.ttest_1samp(a, lr_score)
    ci = reference.confidence_interval()
    check("t", t, reference.statistic)
    check("p", p, reference.pvalue)
    check("CI low", low + lr_score, ci.low)
    check("CI high", high + lr_score, ci.high)
    print("   checked: matches scipy.stats.ttest_1samp\n")


welch("Cantonese BERT", "Chinese BERT")
welch("Cantonese BERT", "Multilingual BERT")
welch("Chinese BERT", "Multilingual BERT")
for model_name in MODELS:
    one_sample(model_name)
print("All hand calculations agree with scipy.")

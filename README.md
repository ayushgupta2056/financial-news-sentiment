# Financial News Sentiment Classifier

Classifies financial news headlines as **bearish**, **bullish** or **neutral** using classical machine learning: Pandas for data handling and scikit-learn for modelling.

The goal is a fast, cheap and explainable baseline for scoring market news flow, the kind of model a research team would set up before trying an LLM. It trains in seconds on a laptop and shows *which words* drive each prediction.

**Tech stack:** Python · Pandas · scikit-learn · TF-IDF · Logistic Regression · Naive Bayes · cross-validation

## Results

Trained on 9.4k headlines and evaluated on a held-out set of 2.4k headlines:

| Model | Macro F1 (5-fold CV) | Test accuracy | Test macro F1 |
|---|---|---|---|
| Multinomial Naive Bayes (baseline) | 0.54 | n/a | n/a |
| **TF-IDF + Logistic Regression** | **0.74** | **0.83** | **0.78** |

| Class | Precision | Recall | F1 |
|---|---|---|---|
| bearish | 0.69 | 0.71 | 0.70 |
| bullish | 0.72 | 0.78 | 0.75 |
| neutral | 0.91 | 0.88 | 0.89 |

The words the model learned as most indicative make financial sense:

- **bearish:** down, lower, downgraded, misses, falls, sues, weak, cuts
- **bullish:** up, higher, rises, positive, beat, beats, jump

## How it works

1. **Load and clean (Pandas):** URLs are removed, cashtags such as `$AAPL` become a generic `TICKER` token so the model learns language rather than company names, and duplicate headlines are dropped.
2. **Explore:** class balance (65% neutral, 20% bullish, 15% bearish) and headline length per class.
3. **Features:** TF-IDF on word unigrams and bigrams, so phrases like "beats estimates" and "cuts guidance" are captured.
4. **Model:** Logistic Regression with `class_weight="balanced"`, so the model doesn't just predict the majority class. It is compared against a Multinomial Naive Bayes baseline using shuffled, stratified 5-fold cross-validation.
5. **Evaluate:** classification report and confusion matrix on the held-out set, plus the top-weighted words per class.

**Lesson learned:** the training file is sorted by label. Without shuffling, cross-validation reported a misleading 0.55 macro F1. With shuffled stratified folds it reports 0.74, which matches the held-out test score.

## Project structure

```
financial-news-sentiment/
├── classify.py          # full pipeline: load → clean → explore → train → evaluate → predict
├── requirements.txt
└── data/
    ├── sent_train.csv   # 9.5k labelled headlines (training)
    └── sent_valid.csv   # 2.4k labelled headlines (held-out test)
```

## Run it

```bash
git clone https://github.com/ayushgupta2056/financial-news-sentiment.git
cd financial-news-sentiment
pip install -r requirements.txt

python classify.py                                                   # train, evaluate, demo
python classify.py "Company beats earnings expectations and raises outlook"   # score your own headline
```

Example output for new headlines:

```
                                                                  bearish  bullish  neutral prediction
Company beats earnings expectations and raises full-year outlook     0.02     0.90     0.08    bullish
Shares slump after regulator fines bank over compliance failures     0.96     0.04     0.01    bearish
Firm to hold quarterly earnings call on Tuesday                      0.03     0.07     0.90    neutral
```

## Next steps

- Compare against a zero-shot LLM classifier on the same test set (accuracy vs. cost and latency).
- Apply the model to earnings-call transcripts and aggregate sentence-level sentiment per call.
- Join daily sentiment with stock returns to test whether sentiment has predictive value.

## Data

[Twitter Financial News Sentiment](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment) by zeroshot (MIT licence): 11.9k finance-related news headlines posted on Twitter, labelled 0 = bearish, 1 = bullish, 2 = neutral.

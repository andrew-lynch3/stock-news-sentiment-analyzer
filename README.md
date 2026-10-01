# Stock News Sentiment Analyzer

A financial-news sentiment classifier (TF-IDF + Logistic Regression) trained on the Financial PhraseBank dataset, then applied to live headlines for 10 large-cap stocks to see whether news sentiment lines up with recent price movement.

Originally built as a final project for CSC 371 (Introduction to NLP).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/andrew-lynch3/stock-news-sentiment-analyzer/blob/main/Notebooks/Sentiment_Analyzer.ipynb)

## Summary

- **Classifier:** labels a financial sentence as positive, negative, or neutral. Test macro-F1 is **0.773** (accuracy 0.831).
- **Application:** scores 10 recent Yahoo Finance headlines for each of 10 tickers and compares the sentiment score to the 30-day price change.
- **Finding:** no detectable relationship between sentiment and price change in this sample (Pearson r = -0.21, n = 10). The sample is far too small to conclude anything. See [Limitations](#limitations).

## Method

1. **Data:** Financial PhraseBank (Malo et al., 2014), the 75%-agreement subset: 3,453 sentences labeled by finance professionals (2,146 neutral, 887 positive, 420 negative).
2. **Preprocessing:** lowercase, strip URLs, numbers, and punctuation, remove stopwords, Porter-stem each token.
3. **Features:** TF-IDF with unigrams and bigrams (`min_df=2`, `max_df=0.95`, sublinear TF), giving a 5,252-term vocabulary.
4. **Model:** multinomial Logistic Regression with balanced class weights.
5. **Split:** stratified 80/20 (2,762 train / 691 test).
6. **Application:** for each ticker, classify its recent headlines from `yfinance` and compute `score = (positive - negative) / total headlines`. Compare against the percent change in closing price over the past month.

## Results

### Classifier (held-out test set, 691 sentences)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| negative | 0.671 | 0.679 | 0.675 | 84 |
| neutral | 0.886 | 0.902 | 0.894 | 429 |
| positive | 0.769 | 0.730 | 0.749 | 178 |
| **macro avg** | **0.775** | **0.770** | **0.773** | 691 |

Overall accuracy is 0.831. Macro-F1 is the headline metric because the data is about 62% neutral. Negative is the weakest class, which fits it having the fewest training examples.

![Confusion matrix](results/confusion_matrix.png)

The highest-weighted features look sensible: *fell, decreas, drop, declin* for negative; *increas, rose, improv, grew* for positive.

### Sentiment vs. price movement (run on Oct 1, 2026)

| Ticker | Sentiment score | 30-day price change |
|---|---|---|
| AAPL | 0.0 | +2.4% |
| MSFT | 0.4 | +2.4% |
| GOOGL | 0.2 | +2.8% |
| AMZN | 0.2 | -2.3% |
| META | 0.0 | +25.4% |
| TSLA | -0.1 | -0.4% |
| NVDA | 0.1 | +5.1% |
| JPM | 0.1 | -6.8% |
| WMT | 0.1 | -1.9% |
| DIS | 0.1 | -1.2% |

![Sentiment vs. price movement](results/sentiment_vs_price.png)

Pearson r across the 10 tickers is **-0.205**. Spearman is -0.21 (p = 0.56). META's +25% move drives the result: without it, Pearson r is +0.15, so even the sign is unstable. This is not evidence for or against sentiment as a signal.

## Limitations

- **Tiny sample.** Ten headlines per ticker and ten tickers. A correlation across 10 points has no statistical power.
- **Domain shift.** The classifier is trained on analyst-report language but applied to news headlines. For example, headlines like "Micron Earnings Crush Views" and "Alphabet shares up in premarket trade after Gemini 4 Argon launch" were labeled neutral. Most live headlines came back neutral, so scores fall in a narrow range (-0.1 to 0.4).
- **Headlines are not always about the company.** Some market-wide headlines appear under several tickers, which adds noise to the per-ticker scores.
- **Timing.** The headlines are from the last few days, but the price change covers the previous month. The sentiment is measured after most of the price move, so it cannot be predictive here. Headlines often follow price rather than lead it.
- **Bag-of-words ceiling.** TF-IDF cannot handle negation or context. A contextual model would likely score higher on the labeled data.
- **Live data.** `yfinance` news changes every run, so rerunning the notebook gives different scores and a different correlation.

### Possible next steps

- Compare against FinBERT or VADER on the same test split.
- Collect headlines over a longer period and test sentiment on day *t* against the return on day *t+1*, rather than comparing against past price moves.

## Repository structure

```
├── README.md
├── requirements.txt
├── notebooks/
│   └── sentiment_analyzer.ipynb
├── results/                  # figures used in this README
└── data/
    └── headlines_snapshot.csv  # headlines used in the run above
```

## Running it

In Colab, click the badge above and run all cells. Locally:

```
pip install -r requirements.txt
jupyter notebook notebooks/sentiment_analyzer.ipynb
```

The notebook downloads Financial PhraseBank from the Hugging Face hub on its own. The `yfinance` headlines are live, so results will differ from the numbers above.

## Data and licensing

Financial PhraseBank is released under CC BY-NC-SA 3.0 and is not redistributed in this repository. Citation: Malo, P., Sinha, A., Korhonen, P., Wallenius, J., and Takala, P. (2014). *Good debt or bad debt: Detecting semantic orientations in economic texts.* Journal of the Association for Information Science and Technology.

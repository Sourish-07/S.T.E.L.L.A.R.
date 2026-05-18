# Empirical Evaluation of Cross-Sectional Equity Signals Under Backtest Overfitting Diagnostics

[![SSRN](https://img.shields.io/badge/SSRN-6078546-blue)](https://dx.doi.org/10.2139/ssrn.6078546)
[![QuantConnect](https://img.shields.io/badge/QuantConnect-Live%20Replication-brightgreen)](https://www.quantconnect.cloud/backtest/c3b036b3c9c13a2161a484db6037baac/?theme=chrome)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org)

Reproducible research pipeline evaluating cross-sectional equity return prediction signals under
rigorous backtest overfitting diagnostics. Central finding: a Probability of Backtest Overfitting
(PBO) of approximately **0.60**, meaning the majority of model configurations that appear
predictive in-sample fail out-of-sample.

---

## Paper

> **An Empirical Evaluation of Cross-Sectional Equity Signals Under Backtest Overfitting Diagnostics**
> SSRN Preprint — DOI: https://dx.doi.org/10.2139/ssrn.6078546

---

## Results

### Research pipeline (CPCV, full historical dataset)

| Metric | Value |
|---|---|
| Mean In-Sample Sharpe | 0.22 |
| Mean Out-of-Sample Sharpe | 0.03 |
| Probability of Backtest Overfitting (PBO) | ~0.60 |
| Average Daily Turnover | 0.57 |

### Live QuantConnect replication (Jan 2020 – Aug 2024, out-of-sample)

| Metric | Value |
|---|---|
| Sharpe Ratio | -0.572 |
| Alpha | -0.126 |
| Probabilistic Sharpe Ratio | 0.006% |
| Net Profit | -45.5% |
| Max Drawdown | 46.5% |
| Total Fees | $18,220 |

The collapse from in-sample Sharpe 0.22 to out-of-sample -0.57 is a direct, executable
demonstration of the PBO finding. A Probabilistic Sharpe of 0.006% means there is
statistically zero evidence of a real edge — exactly what CPCV predicted.

The negative absolute performance reflects regime sensitivity: the strategy shorts
recent winners and longs recent losers. The 2020–2024 period was dominated by
mega-cap technology momentum, the worst possible environment for a mean-reversion
short book — a concrete illustration of the overfitting fragility documented in the paper.

**QuantConnect backtest (public):** [CPCV_PBO_OOS_Backtest_2020_2024](https://www.quantconnect.cloud/backtest/c3b036b3c9c13a2161a484db6037baac/?theme=chrome)

---

## Methodology

### Signals (9 features, Table 1 of paper)

| Category | Feature |
|---|---|
| Momentum | Rank(z-score(r_{t-5:t})), r_{t-10:t}, r_{t-20:t} |
| Reversal | Rank(z-score(-r_{t-1})), -r_{t-3:t}, -r_{t-5:t}) |
| Volume | Rank(z-score(relative volume)), volume surprise, turnover |

All features cross-sectionally standardised by date. The QC replication uses `mr_5`
(negative 5-day return) — the primary reversal signal.

### Validation framework

**Combinatorially Purged Cross-Validation (CPCV):** 20 folds, 105-day embargo at fold
boundaries to prevent leakage from overlapping return windows (López de Prado, 2018).

**Probability of Backtest Overfitting (PBO):** Frequency with which in-sample top-ranked
configurations underperform out-of-sample. PBO = 0.60 quantifies selection bias
(Bailey et al., 2017).

**Transaction cost sensitivity:** Sharpe degrades materially even at 5 bps one-way.
See `src/cost_sweep.py` and `figures/cost_sweep.png`.

---

## Repository Structure

```
src/
  makedataset.py      Master dataset construction (Kaggle + yfinance)
  features.py         9 cross-sectional features (momentum, reversal, volume)
  run_cv.py           CPCV training loop (SGDClassifier, 20 folds)
  backtest.py         Portfolio construction and performance metrics
  pbo.py              PBO computation (Bailey et al. 2017)
  cost_sweep.py       Transaction cost sensitivity analysis
  validation.py       CPCV split generator with purging and embargo
figures/              Paper figures (equity curve, PBO scatter, cost sweep)
results/              Generated output files (CSV, parquet)
data/                 Instructions for obtaining the raw dataset
quantconnect/         Live replication code (main.py)
legacy/               Archived earlier experimental work
```

---

## Reproducibility

### 1. Obtain data

Download from Kaggle following `data/README.md`:
[Price and Volume Data for All US Stocks & ETFs](https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs)

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Build the dataset

```bash
python src/makedataset.py
```

### 4. Run CPCV and generate predictions

```bash
python src/run_cv.py
```

### 5. Backtest and PBO analysis

```bash
python src/backtest.py
python src/pbo.py
python src/cost_sweep.py
```

Output files → `results/`. Figures → `figures/`.

---

## Limitations

This study does not model market impact, borrow costs, or sector neutrality.
Features are intentionally simple and exclude alternative data or nonlinear
transformations. Results are historical and may not generalise to future conditions.
The paper's objective is methodological illustration, not a deployable strategy.

The QuantConnect replication uses weekly rebalancing and a 50-name universe due to
free-tier order limits (10,000 orders per backtest). The signal, portfolio construction,
and cost parameters are otherwise identical to the research codebase.

---

## Citation

```
An Empirical Evaluation of Cross-Sectional Equity Signals Under Backtest Overfitting Diagnostics.
SSRN Preprint. DOI: https://dx.doi.org/10.2139/ssrn.6078546
```

## References

- Bailey, D. H., Borwein, J. M., López de Prado, M., & Zhu, Q. J. (2017). The probability of backtest overfitting. *Journal of Computational Finance*, 20(4), 39–72.
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
- Jegadeesh, N. & Titman, S. (1993). Returns to buying winners and selling losers. *Journal of Finance*, 48(1), 65–91.
- Lehmann, B. N. (1990). Fads, martingales, and market efficiency. *Quarterly Journal of Economics*, 105(1), 1–28.

---

*For research and educational purposes only. Not financial advice.*

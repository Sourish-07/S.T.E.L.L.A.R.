# Empirical Evaluation of Cross-Validated Momentum Strategies

This repository contains code and results accompanying an independent empirical
study of momentum-based trading strategies evaluated under rigorous
cross-validation frameworks.

## Overview

- **Dataset:** 25M+ daily U.S. equity observations  
- **Time period:** Multiple decades  
- **Methodology:**
  - Purged Cross-Validation
  - Combinatorially Purged Cross-Validation (CPCV)
  - Probability of Backtest Overfitting (PBO)
- **Focus:**
  - Out-of-sample robustness
  - Transaction costs
  - Turnover and drawdowns

## Paper

The accompanying research paper is publicly available as a preprint on SSRN:

> **An Empirical Evaluation of Cross-Sectional Equity Signals Under Backtest Overfitting Diagnostics**  
> SSRN Preprint (DOI): https://dx.doi.org/10.2139/ssrn.6078546

Please cite the SSRN version if referencing this work.

## Repository Structure
```
src/ Core modeling, backtesting, and validation code
figures/ Figures used in the accompanying paper
results/ Generated result files (CSV and parquet outputs)
data/ Instructions for obtaining raw data
legacy/ Archived earlier experimental work
```

## Reproducibility

### 1. Obtain the data
Download the raw dataset from Kaggle following the instructions in: data/README.md

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Construct datasets
```bash
python src/makedataset.py
```

### 4. Run cross-validation and backtests
```bash
python src/run_cv.py
```

### Notes
This repository is intended for research and educational purposes only.
It does not constitute financial or investment advice.

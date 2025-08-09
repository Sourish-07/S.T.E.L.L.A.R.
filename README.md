# S.T.E.L.L.A.R.
Stock market prediction app using Random Forest models, Twitter sentiment analysis, and real-time stock data scraping.

# Stock Market Prediction with Twitter Sentiment Analysis & Real-Time Data

## Overview
This project predicts stock market movement by combining machine learning models, Twitter sentiment analysis, and real-time stock price data. It began as part of my research with Inspirit AI and has evolved into a Streamlit web application capable of predicting the movement of 25 selected stocks.

---

## Project History

### **1. Inspirit AI Research Phase**
- Conducted as part of the Inspirit AI high school research program with mentor guidance.
- Dataset used: *Stock Prediction GAN + Twitter Sentiment Analysis* by Hanna Yukhymenko.
- Dataset shape: `(63,676, 12)` — 10 original columns + 2 engineered:
  - `% change in stock price` (for classification — positive or negative daily change).
  - `overall change` (1 or 0 for positive/negative change).
- Performed 80–20 train-test split.
- Tested **4 models**:
  1. Linear Regression
  2. Random Forest Regressor
  3. Logistic Regression
  4. Random Forest Classifier
- Final choice:
  - **Random Forest Regressor** (for regression tasks)
  - **Random Forest Classifier** (for classification tasks)
- Tuned hyperparameters:
  - Regression: `max_depth`, `max_features`, `n_estimators`, `min_samples_split`, `bootstrap`
  - Classification: `n_estimators`, `max_depth`, `max_features`, `min_samples_leaf`, `max_samples`, `bootstrap`
- Evaluated performance using confusion matrices and accuracy graphs.

---

### **2. Streamlit Web App Development**
- Implemented a user-friendly Streamlit interface.
- Allowed users to select a stock and view predictions.
- Added dropdowns that display both ticker and company name for better UX.

---

### **3. Real-Time Data Integration**
- Extended app to scrape **real-time stock prices**.
- Combined with **Twitter sentiment scores** (from 2021–2022 dataset) to provide up-to-date predictions.
- Supports **25 stocks** with live updates.

---

## Features
- **Dual prediction modes**: Regression (price change magnitude) & Classification (up/down movement).
- **Twitter sentiment integration**.
- **Live stock price scraping** for current-day predictions.
- Clear **visualization** of model predictions.
- Simple, intuitive **web interface**.

---
Acknowledgements
-Shrish Mudumby Venugopal - Post Inspirit project development

-Inspirit AI — Research mentorship and guidance.
  *Tommy Pawelski - Research mentor

-Hanna Yukhymenko — Original dataset creator.

## How to Run Locally
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/your-repo.git
2. Navigate to the project folder:
   ```bash
   cd your-repo
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
4. Run the Streamlit app:
   ```bash
   streamlit run app.py

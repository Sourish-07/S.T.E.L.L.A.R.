import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import cloudpickle
from datetime import date
import feedparser
from yahoo_fin import news
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import datetime as dt
import pytz
import subprocess
import sys

try:
    from transformers import AutoTokenizer
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers==4.53.0"])
    from transformers import AutoTokenizer
    
# === Load model and vectorizer ===
with open("clean_stock_bot.pkl", "rb") as f:
    model = cloudpickle.load(f)

with open("clean_vectorizer.pkl", "rb") as f:
    vectorizer = cloudpickle.load(f)

# === Load and preprocess tweet + price CSVs ===
tweets_df = pd.read_csv('stock_tweets.csv')
prices_df = pd.read_csv('stock_yfinance_data.csv')

# Clean tweet data
tweets_df_clean = tweets_df.drop_duplicates(subset=["Tweet", "Stock Name", "Date"])
tweets_df_clean["Date"] = tweets_df_clean["Date"].str.split().str[0]

# Merge full dataset
full_data = tweets_df_clean.merge(prices_df, on=["Date", "Stock Name"])
full_data['% change in stock price'] = (full_data['Close'] - full_data['Open']) / full_data['Open']
full_data['overall change'] = (full_data['% change in stock price'] > 0).astype(int)

# === Define stocks and UI ===
stocks = sorted(full_data['Stock Name'].unique())
st.set_page_config(layout="wide")
st.title("Stock Sentiment Prediction Dashboard")

# === Tabs ===
tabs = st.tabs(["📅 Research Project Timeline", "🔮 Prediction"])
#===Yahoo Finance News Scraper=====
def get_yahoo_headlines(ticker):
    try:
        raw_news = news.get_yf_rss(ticker)
        headlines = [item['title'] for item in raw_news[:5]]
        return headlines
    except Exception:
        return []


# === Tab 1: Show merged dataset ===
with tabs[0]:
    st.subheader("About This Project")
    st.markdown("""
        My name is Sourish Mudumby Venugopal, and I developed this web application as part of a research project focused on applying artificial intelligence to financial forecasting.
                
        This project explores how **Twitter sentiment** can be used to predict the future movement of stocks.

        The dataset includes tweets from 2021 to 2022 alongside real stock prices accquired via Kaggle. 
                
        A Random Forest Classifier was trained using tweet data and used to predict whether a stock's price would increase or decrease.

        **Key Features:**
        - Integrated tweet and price dataset (2021–2022)
        - Model trained on labeled tweet sentiment tied to price changes
        - Real-time prediction using current market data - Imported Finbert for real-time data

        This work was supported by Inspirit AI and builds on principles of Natural Language Processing (NLP), Machine Learning, and Financial Forecasting.
        """)

#----------------------------------------------------------------------------------
    st.subheader("📅 Research Project Timeline")
        # Function to display each week's document in a styled box
    def display_weekly_document(week_num, file_path, description):
        # HTML and CSS for the styled box
        box_html = f"""
        <div style="border: 2px solid #FFFFFF; border-radius: 15px; padding: 15px; margin: 10px 0; display: flex; align-items: center;">
            <img src="https://google.oit.ncsu.edu/wp-content/uploads/sites/6/2021/01/Google_Docs.max-2800x2800-1-768x768.png" alt="file-icon" style="width: 40px; height: 40px; margin-right: 15px;">
            <div style="flex-grow: 1;">
                <strong>Week {week_num}: {description}</strong><br>
                <a href="file:///{file_path}" download style="text-decoration: none; color: #FFFFFF;">Click to Download</a>
            </div>
        </div>
        """
        st.markdown(box_html, unsafe_allow_html=True)

    # Example Usage:
    # Replace these paths with your actual file paths
    file_paths = [
        "Stellar\\DOCS\\Week 1.docx",
        "Stellar\\DOCS\\Week 2.docx",
        "Stellar\\DOCS\\Week 3.docx",
        "Stellar\\DOCS\\Week 4.docx",
        "Stellar\\DOCS\\Week 5.docx",
        "Stellar\\DOCS\\Week 6.docx",
        "Stellar\\DOCS\\Week 7.docx",
        "Stellar\\DOCS\\Week 8.docx",
        "Stellar\\DOCS\\Week 9.docx",
        "Stellar\\DOCS\\Week 10.docx",

    ]

    # List of descriptions for each week
    descriptions = [
        "Domain and Research Question",
        "Literature Review",
        "Pre-Processed Data",
        "Data Analysis",
        "Devloped Baseline Models",
        "Switched from Regression to Classification",
        "Tuned Hyperparameters for Random Forest",
        "Started Working on Research Paper Draft",
        "Finalized paper and Drafted Elevator Pitch - Started thinking about Streamlit",
        "Went over future deliveries (Science Fair , Research Journal etc.)",
    ]

    # Loop to display each week
    for week_num, (file_path, description) in enumerate(zip(file_paths, descriptions), 1):
        display_weekly_document(week_num, file_path, description)
    st.markdown("Links have been removed for privacy reasons")
#----------------------------------------------------------------------------------
    st.markdown("------------------------------------------------------")
    st.subheader("'full_data' - file used to train R.F.C in google collab")
    st.dataframe(full_data, use_container_width=True)

    # Graphs
    st.markdown("------------------------------------------------------")
    st.subheader("Stock Graphs from week 4")
    # 3D Plotly Graph (Tweet vs Price vs Volume)
    import plotly.express as px
    fig = px.scatter_3d(full_data, x='Date', y='Close', z='Volume',
                        color='Stock Name', 
                        title='3D Scatter of Stock Data - x = Date , y = Close , z = Volume')
    fig.update_traces(marker=dict(size=2))
    fig.update_layout (height=700)
    st.plotly_chart(fig, use_container_width=True)
    
    # Second Graph 
    fig = px.scatter_3d(full_data, x='Date', y='Stock Name', z='Close',
              color='Stock Name', size='Volume', size_max=18,
              symbol='Stock Name', opacity=0.7,title='3D Scatter of Stock Data - x = Date , y = Stock Name , z = Close')
    fig.update_traces(marker=dict(size=2))
    fig.update_layout(height=700)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("------------------------------------------------------")
    st.subheader("Accuracy of R.F.C.")
    st.markdown("Confusion Matrix - 0 indicates stock went down 1 indicates stock went up")
    st.image("Confusion Matrix.png")
    st.markdown("ROC Curve - Shows association established by random forest between twitter sentiments and stock prices ")
    st.image("ROC Curve.png")
    #Thank you message
    st.markdown(
        """
        <div style="text-align: center; font-size: 30px; margin-top: 50px;">
            🙏 Thank you for reading!
        </div>
        """,
        unsafe_allow_html=True
    )
    #add spacing before footer 
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    #Footer
    st.markdown(
        """
        <hr style="margin-top: 50px;">
        <div style="text-align: center; color: gray; font-size: 14px;">
            📧 Contact: sourishvenugopal11@gmail.com
        </div>
        """,
        unsafe_allow_html=True
    )


# === Tab 2: Prediction ===
with tabs[1]:

    st.subheader("Predict Stock Movement")

    # Map ticker → company name using full_data
    stock_labels = {
    row["Stock Name"]: row["Company Name"]
    for _, row in full_data.drop_duplicates("Stock Name").iterrows()
    }

    # Create dropdown options like "AAPL – Apple"
    dropdown_options = [f"{ticker} – {stock_labels[ticker]}" for ticker in stock_labels]

    # User selects
    selected_option = st.selectbox("Select a stock", dropdown_options)
    selected_stock = selected_option.split(" – ")[0]  # Extract ticker
    company_name = stock_labels[selected_stock]   
    selected_date = st.date_input("Choose a date", value=date(2022, 1, 3))
    selected_str = selected_date.strftime('%Y-%m-%d')

    st.write(f"**Date selected:** {selected_str}")

    # Filter data
    tweet_data = full_data[(full_data['Stock Name'] == selected_stock) & (full_data['Date'] == selected_str)]

    # 🔁 Real-time fallback if today's date
    if selected_date == date.today():
        st.info("Pulling real-time stock data from yfinance...")

        try:
            full_day = yf.download(selected_stock, period="1d", interval="1m", progress=False)

            if not full_day.empty and "Open" in full_day.columns and "Close" in full_day.columns:
                # Get current date and set 9:30 AM as official market open
                now = dt.datetime.now()
                eastern = pytz.timezone("US/Eastern")
                market_open_time = eastern.localize(dt.datetime.combine(now.date(), dt.time(9, 30))).astimezone(dt.timezone.utc)


                # Filter data starting from market open
                intraday = full_day[full_day.index >= market_open_time]

                if not intraday.empty:
                    today_open = float(full_day["Open"].iloc[0])
                    today_close = float(full_day["Close"].iloc[-1])
                    change = (today_close - today_open) / today_open
                    st.markdown(f"📊 Real-time % price change today: **{change:.2%}**")
                    st.caption("ℹ️ Based on intraday data from official market open (9:30 AM EST).")
                else:
                    st.warning("⚠️ Not enough data after market open to calculate change.")
            else:
                st.warning("❌ Could not extract 'Open' and 'Close' from full-day data.")

            # Optional: show the raw data
            st.write("📄 Raw real-time full-day data:")
            st.dataframe(full_day)

        except Exception as e:
            st.warning(f"❌ Error fetching real-time price data: {e}")


    def load_finbert_model():
        tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        model_finbert = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        return tokenizer, model_finbert

    tokenizer, model_finbert = load_finbert_model()

    # === HEADLINE SENTIMENT SECTION ===
    st.markdown("### 🧠 Headline-Based Sentiment Prediction (RoBERTa)")
    headlines = get_yahoo_headlines(selected_stock)

    def analyze_sentiment_finbert(headlines):
        results = []
        sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
    
        for text in headlines:
            inputs = tokenizer(text, return_tensors="pt", truncation=True)
            with torch.no_grad():
                logits = model_finbert(**inputs).logits
                probs = torch.nn.functional.softmax(logits, dim=1)[0]
                pred = torch.argmax(probs).item()
                label = ["positive", "neutral", "negative"][pred]
                sentiment_counts[label] += 1
                results.append((text, label))
    
        return results, sentiment_counts

    # === Display
    if headlines:
        st.write("#### Headlines Analyzed:")
        results, sentiment_counts = analyze_sentiment_finbert(headlines)

        for i, (headline, sentiment) in enumerate(results, 1):
            color = {"positive": "🟢", "negative": "🔴", "neutral": "🟡"}.get(sentiment, "⚪️")
            st.write(f"{i}. {headline}")
            st.caption(f"{color} Sentiment: `{sentiment}`")

        if sentiment_counts["positive"] > sentiment_counts["negative"]:
            st.success("🟢 Final Prediction: Stock is predicted to go **UP** based on headline sentiment.")
        elif sentiment_counts["negative"] > sentiment_counts["positive"]:
            st.error("🔴 Final Prediction: Stock is predicted to go **DOWN** based on headline sentiment.")
        else:
            st.info("🟡 Final Prediction: Headline sentiment is **neutral or mixed**.")

        st.caption("Prediction made using **RoBERTa sentiment model (no API)**.")
    else:
        st.warning("⚠️ No recent headlines found for this ticker.")

    # === TWEET SENTIMENT (if available) ===
    st.markdown("### 🐦 Tweet Sentiment (Historical) - Random Forest")

    if not tweet_data.empty:
        tweets = tweet_data['Tweet'].astype(str).tolist()
        X_vectorized = vectorizer.transform(tweets)
        prediction = model.predict(X_vectorized)
        avg_pred = round(np.mean(prediction))

        if avg_pred == 1:
            st.success(f"📈 Based on tweet sentiment, {selected_stock} was predicted to go **UP** on {selected_str}.")
        else:
            st.error(f"📉 Based on tweet sentiment, {selected_stock} was predicted to go **DOWN** on {selected_str}.")

        st.caption("Prediction made using **historical tweet sentiment and stock model**.")
    else:
        st.info("No tweet data available for this date.")


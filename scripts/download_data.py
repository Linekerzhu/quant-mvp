
import yfinance as yf
import pandas as pd
import numpy as np
import os
import requests
from io import StringIO

print("Fetching S&P 500 tickers...")
url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
html = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}).text
tables = pd.read_html(StringIO(html))
sp500_tickers = tables[0]["Symbol"].tolist()
sp500_tickers = [t.replace(".", "-") for t in sp500_tickers]

print(f"Downloading {len(sp500_tickers)} tickers...")
# Download history
data = yf.download(sp500_tickers, start="2023-01-01", end="2026-03-06", group_by="ticker", threads=True, auto_adjust=False)

print("Formatting data...")
dfs = []
for ticker in sp500_tickers:
    if ticker in data and not data[ticker].empty:
        df = data[ticker].dropna(subset=["Close"]).copy()
        if df.empty: continue
        df["symbol"] = ticker
        dfs.append(df)

if not dfs:
    print("Warning: Failed to download data!")
    exit(1)

result = pd.concat(dfs).reset_index()
result.rename(columns={"Date": "date", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume", "Adj Close": "adj_close"}, inplace=True)
result.columns = [c.lower() for c in result.columns]

os.makedirs("data/backtest", exist_ok=True)
result.to_parquet("data/backtest/sp500_2023_2026.parquet")
print(f"Saved {len(result)} rows to data/backtest/sp500_2023_2026.parquet!")

import pandas as pd
import yfinance as yf

def extract_data(stocks, start, end):
    def data(stock):
        return yf.download(stock, start=start, end=end)

    stocks_map = map(data, stocks)
    return pd.concat(stocks_map, keys=stocks, names=["Tickers", "Date"])

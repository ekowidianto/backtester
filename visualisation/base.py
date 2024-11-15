import matplotlib.pyplot as plt
import pandas as pd


def plot_buy_sell(symbol, df_prices: pd.DataFrame):
    df_prices = df_prices.reset_index()
    fig = plt.figure(figsize=[14, 10])

    sub = fig.add_subplot(3, 1, (1, 2), xlabel="Date", ylabel=f"{symbol} Price")

    sub.set_title(symbol)

    sub.set_xlim(df_prices["Date"].min(), df_prices["Date"].max())
    sub.plot(
        df_prices["Date"],
        df_prices["Adj Close"],
        color="grey",
        linewidth=0.75,
        label="Adj Close Price",
    )

    # BUY signal
    buy_index = df_prices.buy_or_sell == 1.0
    sub.plot(
        df_prices.loc[buy_index]["Date"],
        df_prices[buy_index]["Adj Close"],
        "^",
        color="green",
        markersize=6,
        label="Buy",
    )

    # SELL signal
    sell_index = df_prices.buy_or_sell == -1.0
    sub.plot(
        df_prices.loc[sell_index]["Date"],
        df_prices[sell_index]["Adj Close"],
        "v",
        color="red",
        markersize=6,
        label="Sell",
    )
    sub.legend()
    return fig, sub


def plot_macd_buy_sell(symbol, df_prices: pd.DataFrame):
    df_prices = df_prices.reset_index()
    fig, _ = plot_buy_sell(symbol, df_prices)
    sub = fig.add_subplot(3, 1, 3, xlabel="Date", ylabel=f"MACD")
    sub.set_xlim(df_prices["Date"].min(), df_prices["Date"].max())
    sub.plot(df_prices["Date"], df_prices["MACD"], linewidth=0.75, label="MACD")
    sub.plot(
        df_prices["Date"],
        df_prices["MACD Signal Line"],
        linewidth=0.75,
        label="Signal Line",
    )
    sub.legend()


def plot_ma_crossover_buy_sell(symbol, df_prices):
    df_prices = df_prices.reset_index()

    fig, sub = plot_buy_sell(symbol, df_prices)

    sub.plot(
        df_prices["Date"],
        df_prices["SMA_short"],
        color="blue",
        ls="--",
        label="SMA Short",
        linewidth=0.80,
    )
    sub.plot(
        df_prices["Date"],
        df_prices["SMA_long"],
        color="green",
        ls="--",
        label="SMA Long",
        linewidth=0.80,
    )
    sub.legend()

    sub = fig.add_subplot(3, 1, 3, xlabel="Date", ylabel=f"Trading Position")
    sub.set_xlim(df_prices["Date"].min(), df_prices["Date"].max())
    sub.plot(
        df_prices["Date"],
        df_prices["trading_positions"],
        color="red",
        label="Trading Positions",
    )
    sub.legend()


def plot_sma_mean_reversion_buy_sell(symbol, df_prices: pd.DataFrame):
    df_prices = df_prices.reset_index()

    fig, sub = plot_buy_sell(symbol, df_prices)

    sub.plot(df_prices["Date"], df_prices["SMA"], color="blue", ls="--", label="SMA")
    sub.legend()

    sub = fig.add_subplot(3, 1, 3, xlabel="Date", ylabel=f"Price - SMA Diff")
    sub.set_xlim(df_prices["Date"].min(), df_prices["Date"].max())

    sub.axhline(0, color="k", linestyle="--")
    sub.plot(
        df_prices["Date"],
        df_prices["Upper_Threshold"],
        color="g",
        linestyle="--",
        label="Threshold",
    )
    sub.plot(df_prices["Date"], df_prices["Lower_Threshold"], color="g", linestyle="--")
    sub.plot(
        df_prices["Date"],
        df_prices["SMA_Price_Diff"],
        color="grey",
        label="Price - SMA Diff",
    )

    buy_positions = df_prices[df_prices["buy_or_sell"] == 1]
    sell_positions = df_prices[df_prices["buy_or_sell"] == -1]
    sub.scatter(
        buy_positions["Date"],
        buy_positions["buy_or_sell"],
        c="r",
        alpha=0.3,
    )
    sub.scatter(
        sell_positions["Date"],
        sell_positions["buy_or_sell"],
        c="b",
        alpha=0.3,
    )
    sub.legend()


def plot_lag(symbol, df_prices):
    df_prices = df_prices.reset_index()

    fig, sub = plot_buy_sell(symbol, df_prices)

    sub = fig.add_subplot(3, 1, 3, xlabel="Date", ylabel=f"Trading Position")
    sub.set_xlim(df_prices["Date"].min(), df_prices["Date"].max())
    sub.plot(
        df_prices["Date"],
        df_prices["trading_positions"],
        color="red",
        label="Trading Positions",
    )
    sub.legend()


def plot_simple_momentum(symbol, df_prices):
    df_prices = df_prices.reset_index()

    fig, sub = plot_buy_sell(symbol, df_prices)

    sub = fig.add_subplot(3, 1, 3, xlabel="Date", ylabel=f"Trading Position")
    sub.set_xlim(df_prices["Date"].min(), df_prices["Date"].max())
    sub.plot(
        df_prices["Date"],
        df_prices["trading_positions"],
        color="red",
        label="Trading Positions",
    )
    sub.legend()


def plot_rsi_buy_sell(
    symbol,
    df_prices: pd.DataFrame,
):
    df_prices = df_prices.reset_index()

    fig, _ = plot_buy_sell(symbol, df_prices)
    sub = fig.add_subplot(3, 1, 3, xlabel="Date", ylabel=f"RSI")
    sub.set_xlim(df_prices["Date"].min(), df_prices["Date"].max())
    sub.plot(df_prices["Date"], df_prices["RSI"], linewidth=0.75, label="RSI")
    sub.plot(
        df_prices["Date"], df_prices["RSI_lower_threshold"], color="g", linestyle="--"
    )
    sub.plot(
        df_prices["Date"], df_prices["RSI_upper_threshold"], color="r", linestyle="--"
    )
    sub.legend()

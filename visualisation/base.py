import matplotlib.pyplot as plt


def plot_buy_sell(symbol, df_prices):
    df_prices = df_prices.reset_index()
    fig = plt.figure(figsize=[16, 12])

    sub = fig.add_subplot(3, 1, (1, 2), xlabel="Date", ylabel=f"{symbol} Price")

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
        markersize=4,
        label="Buy",
    )

    # SELL signal
    sell_index = df_prices.buy_or_sell == -1.0
    sub.plot(
        df_prices.loc[sell_index]["Date"],
        df_prices[sell_index]["Adj Close"],
        "v",
        color="red",
        markersize=4,
        label="Sell",
    )
    sub.legend()
    return fig


def plot_macd_buy_sell(symbol, df_prices):
    df_prices = df_prices.reset_index()

    fig = plot_buy_sell(symbol, df_prices)
    sub = fig.add_subplot(3, 1, 3, xlabel="Date", ylabel=f"MACD")
    sub.plot(df_prices["Date"], df_prices["MACD"], linewidth=0.75, label="MACD")
    sub.plot(
        df_prices["Date"],
        df_prices["MACD Signal Line"],
        linewidth=0.75,
        label="Signal Line",
    )
    sub.legend()

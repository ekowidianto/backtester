import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lib.performance import PerformanceCustom


class Portfolio:
    def __init__(
        self, df_prices: pd.DataFrame, capital: float = 1e6, transaction_fee: float = 0
    ):
        self.df_prices = df_prices.copy(deep=True).reset_index()
        self.capital = capital
        self.transaction_fee = transaction_fee
        self.symbol = self.df_prices["Tickers"].iloc[0]

        self.df_portfolio = self.compute_portfolio()

    def get_performance(self) -> PerformanceCustom:
        df = self.df_portfolio[["Date", "strategy_cum_net_returns"]].copy()
        df.columns = ["Date", "Cumulative Returns"]
        return PerformanceCustom(df, self.symbol)

    def get_portfolio(self) -> pd.DataFrame:
        return self.df_portfolio

    def get_final_capital(self, with_fee=True) -> float:
        col_name = "total_holdings_after_fee" if with_fee else "total_holdings"
        return np.around(self.get_portfolio()[col_name].iloc[-1], 2)

    def compute_portfolio(self):
        df_portfolio = self.df_prices[
            ["Date", "Adj Close", "buy_or_sell", "trading_positions"]
        ].copy()

        df_portfolio["log_returns"] = self._compute_log_returns(
            df_portfolio["Adj Close"]
        )
        df_portfolio["cum_returns"] = self._compute_cum_returns(
            df_portfolio["log_returns"]
        )

        df_portfolio["strategy_log_returns"] = (
            df_portfolio["trading_positions"].shift(1) * df_portfolio["log_returns"]
        )
        df_portfolio["strategy_cum_returns"] = self._compute_cum_returns(
            df_portfolio["strategy_log_returns"]
        )
        df_portfolio["total_holdings"] = (
            df_portfolio["strategy_cum_returns"] * self.capital
        )
        df_portfolio.loc[0, "total_holdings"] = self.capital

        df_portfolio["commission_fee"] = (
            np.abs(df_portfolio["buy_or_sell"]) * self.transaction_fee
        ).cumsum()
        df_portfolio["total_holdings_after_fee"] = (
            df_portfolio["total_holdings"] - df_portfolio["commission_fee"]
        )
        df_portfolio["strategy_cum_net_returns"] = (
            df_portfolio["total_holdings_after_fee"]
            / df_portfolio["total_holdings_after_fee"].iloc[0]
        )

        return df_portfolio

    def get_final_cumulative_returns(self):
        final_row = self.df_portfolio.iloc[-1]
        cumret_dict = {
            "Type": ["Passive", "Strategy", "Strategy with Fee"],
            "Cum Ret": [
                final_row["cum_returns"],
                final_row["strategy_cum_returns"],
                final_row["strategy_cum_net_returns"],
            ],
        }
        return pd.DataFrame(cumret_dict)

    def plot_returns(self, with_fee: bool = True):
        df_portfolio = self.df_portfolio.reset_index()
        fig = plt.figure(figsize=[16, 12])

        sub = fig.add_subplot(5, 1, (1, 2), xlabel="Date", ylabel=f"Adj Close")
        sub.set_title(self.symbol)
        sub.set_xlim(df_portfolio["Date"].min(), df_portfolio["Date"].max())
        sub.plot(
            df_portfolio["Date"],
            df_portfolio["Adj Close"],
            color="grey",
            linewidth=0.75,
            label="Adj Close Price",
        )
        # BUY signal
        buy_index = df_portfolio.buy_or_sell == 1.0
        sub.plot(
            df_portfolio.loc[buy_index]["Date"],
            df_portfolio[buy_index]["Adj Close"],
            "^",
            color="green",
            markersize=4,
            label="Buy",
        )

        # SELL signal
        sell_index = df_portfolio.buy_or_sell == -1.0
        sub.plot(
            df_portfolio.loc[sell_index]["Date"],
            df_portfolio[sell_index]["Adj Close"],
            "v",
            color="red",
            markersize=4,
            label="Sell",
        )
        sub.legend()

        sub = fig.add_subplot(5, 1, (3, 4), xlabel="Date", ylabel=f"Cumulative Returns")
        sub.set_xlim(df_portfolio["Date"].min(), df_portfolio["Date"].max())
        sub.plot(
            df_portfolio["Date"],
            df_portfolio["cum_returns"],
            color="grey",
            linewidth=0.75,
            label="Passive",
        )

        strat_cum_ret = (
            df_portfolio["strategy_cum_net_returns"]
            if with_fee
            else df_portfolio["strategy_cum_returns"]
        )
        strat_label = "Strategy Return with Fee" if with_fee else "Strategy Return"
        sub.plot(
            df_portfolio["Date"],
            strat_cum_ret,
            color="blue",
            linewidth=0.75,
            label=strat_label,
        )
        sub.legend()

        sub = fig.add_subplot(5, 1, 5, xlabel="Date", ylabel=f"Positions")
        sub.set_xlim(df_portfolio["Date"].min(), df_portfolio["Date"].max())
        sub.plot(
            df_portfolio["Date"],
            df_portfolio["trading_positions"],
            color="blue",
            linewidth=0.75,
        )

    def _compute_log_returns(self, daily_price_data: pd.Series) -> pd.Series:
        return np.log(daily_price_data / daily_price_data.shift(1))

    def _compute_cum_returns(self, daily_log_returns: pd.Series) -> pd.Series:
        return daily_log_returns.fillna(0).cumsum().apply(np.exp)

    def compute_trading_opportunities(self) -> float:
        return self.get_portfolio[
            ~self.get_portfolio["trading_positions"].isna()
        ].shape[0]


class PortfolioUpdated(Portfolio):
    def __init__(
        self,
        df_prices: pd.DataFrame,
        capital: float = 1e6,
        transaction_fee: float = 0,
        num_shares: float = 100,
    ):
        self.num_shares = num_shares
        super().__init__(df_prices, capital, transaction_fee)

    def compute_portfolio(self):
        df_portfolio = self.df_prices[
            ["Date", "Adj Close", "buy_or_sell", "trading_positions"]
        ].copy()

        df_portfolio["log_returns"] = self._compute_log_returns(
            df_portfolio["Adj Close"]
        )
        df_portfolio["cum_returns"] = self._compute_cum_returns(
            df_portfolio["log_returns"]
        )

        df_portfolio["position"] = self.num_shares * df_portfolio[
            "trading_positions"
        ].shift(
            0
        )  # TODO: Change to 1
        df_portfolio["diff_shares_owned"] = (
            self.num_shares * df_portfolio["trading_positions"].diff()
        ).fillna(0)
        df_portfolio["holdings"] = df_portfolio["position"] * df_portfolio["Adj Close"]
        df_portfolio["cash"] = (
            self.capital
            - (df_portfolio["diff_shares_owned"] * df_portfolio["Adj Close"]).cumsum()
        )
        df_portfolio["total_holdings"] = df_portfolio["cash"] + df_portfolio["holdings"]

        df_portfolio["strategy_log_returns"] = self._compute_log_returns(
            df_portfolio["total_holdings"]
        )
        df_portfolio["strategy_cum_returns"] = self._compute_cum_returns(
            df_portfolio["strategy_log_returns"]
        )

        df_portfolio["commission_fee"] = (
            np.abs(df_portfolio["buy_or_sell"]) * self.transaction_fee
        )
        df_portfolio["total_holdings_after_fee"] = (
            df_portfolio["total_holdings"] - df_portfolio["commission_fee"]
        )
        df_portfolio["strategy_cum_net_returns"] = (
            df_portfolio["total_holdings_after_fee"]
            / df_portfolio["total_holdings_after_fee"].iloc[0]
        )

        return df_portfolio

    def plot_returns(self, with_fee: bool = True):
        df_portfolio = self.df_portfolio.reset_index()
        fig = plt.figure(figsize=[16, 12])

        sub = fig.add_subplot(5, 1, (1, 2), xlabel="Date", ylabel=f"Adj Close")
        sub.set_title(self.symbol)
        sub.set_xlim(df_portfolio["Date"].min(), df_portfolio["Date"].max())
        sub.plot(
            df_portfolio["Date"],
            df_portfolio["Adj Close"],
            color="grey",
            linewidth=0.75,
            label="Adj Close Price",
        )
        # BUY signal
        buy_index = df_portfolio.buy_or_sell == 1.0
        sub.plot(
            df_portfolio.loc[buy_index]["Date"],
            df_portfolio[buy_index]["Adj Close"],
            "^",
            color="green",
            markersize=4,
            label="Buy",
        )

        # SELL signal
        sell_index = df_portfolio.buy_or_sell == -1.0
        sub.plot(
            df_portfolio.loc[sell_index]["Date"],
            df_portfolio[sell_index]["Adj Close"],
            "v",
            color="red",
            markersize=4,
            label="Sell",
        )
        sub.legend()

        sub = fig.add_subplot(5, 1, (3, 4), xlabel="Date", ylabel=f"Value of Portfolio (USD)")
        sub.set_xlim(df_portfolio["Date"].min(), df_portfolio["Date"].max())

        strat_total_values = (
            df_portfolio["total_holdings_after_fee"]
            if with_fee
            else df_portfolio["total_holdings"]
        )
        strat_label = "Strategy Value with Fee" if with_fee else "Strategy Value"
        sub.plot(
            df_portfolio["Date"],
            strat_total_values,
            color="blue",
            linewidth=0.75,
            label=strat_label,
        )
        sub.legend()

        sub = fig.add_subplot(5, 1, 5, xlabel="Date", ylabel=f"Positions")
        sub.set_xlim(df_portfolio["Date"].min(), df_portfolio["Date"].max())
        sub.plot(
            df_portfolio["Date"],
            df_portfolio["trading_positions"],
            color="blue",
            linewidth=0.75,
        )

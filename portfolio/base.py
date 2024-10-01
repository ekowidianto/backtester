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

        self.df_portfolio = self.init_portfolio()

    def get_performance(self) -> PerformanceCustom:
        df = self.df_portfolio[["Date", "strategy_cum_net_returns"]].copy()
        df.columns = ["Date", "Cumulative Returns"]
        return PerformanceCustom(df)

    def get_portfolio(self) -> pd.DataFrame:
        return self.df_portfolio

    def init_portfolio(self):
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

        df_portfolio["commission_fee"] = (
            np.abs(df_portfolio["buy_or_sell"]) * self.transaction_fee
        ).cumsum()
        df_portfolio["capital"] = (
            df_portfolio["strategy_cum_returns"] * self.capital
            - df_portfolio["commission_fee"]
        )
        df_portfolio.loc[0, "capital"] = self.capital
        df_portfolio["strategy_cum_net_returns"] = (
            df_portfolio["capital"] / df_portfolio["capital"].iloc[0]
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

        sub = fig.add_subplot(3, 1, (1, 2), xlabel="Date", ylabel=f"Cumulative Returns")
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

        sub = fig.add_subplot(3, 1, 3, xlabel="Date", ylabel=f"Positions")
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
        return daily_log_returns.cumsum().apply(np.exp)

import datetime as dt
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class PerformanceCustom:
    def __init__(self, cum_returns: pd.DataFrame):
        self.cumulative_returns = cum_returns
        self.cumulative_returns.set_index("Date", inplace=True)

    def compute_n_largest_drawdowns(self, n: int = 5):
        df_drawdown = self.cumulative_returns.copy()
        df_drawdown["watermark"] = df_drawdown["Cumulative Returns"].cummax()
        df_drawdown["drawdown"] = (
            df_drawdown["watermark"] - df_drawdown["Cumulative Returns"]
        )

        breakpoints = df_drawdown["drawdown"].loc[df_drawdown["drawdown"] == 0].index

        if (
            df_drawdown["Cumulative Returns"].iloc[-1]
            < df_drawdown["watermark"].iloc[-1]
        ):
            breakpoints = breakpoints.append(df_drawdown.tail(1).index)

        drawdown_periods = (
            breakpoints[1:].to_pydatetime() - breakpoints[:-1].to_pydatetime()
        )

        self.drawdowns = []

        for i in range(drawdown_periods.shape[0]):
            start, end = breakpoints[i], breakpoints[i + 1]

            max_drawdown = df_drawdown["drawdown"].loc[start:end].max()
            self.drawdowns.append(
                {
                    "start_date": start,
                    "end_date": end,
                    "drawdown_period": drawdown_periods[i],
                    "max_drawdown": max_drawdown,
                }
            )
        df_n_drawdowns = (
            pd.DataFrame(self.drawdowns)
            .sort_values(by="max_drawdown", ascending=False)
            .reset_index(drop=True)
            .iloc[:n]
        )
        self._plot_n_drawdown(n, df_drawdown, df_n_drawdowns)
        return df_n_drawdowns

    def _plot_n_drawdown(
        self, n: int, df_drawdown: pd.DataFrame, df_n_drawdowns: pd.DataFrame
    ):
        _, ax = plt.subplots(1, figsize=(16, 10))
        df_drawdown[["Cumulative Returns"]].plot(color="green", ax=ax)
        df_drawdown[["watermark"]].plot(color="blue", ls="--", ax=ax, label="Watermark")

        for i in range(n):
            start = df_n_drawdowns.iloc[i]["start_date"]
            end = df_n_drawdowns.iloc[i]["end_date"] + dt.timedelta(days=1)
            days = df_n_drawdowns.iloc[i]["drawdown_period"].days

            df_drawdown.loc[start:end]["watermark"].plot(
                ax=ax, label=f"Drawdown period = {days} days"
            )

        ax.legend()
        plt.show()

    def compute_sharpe_ratio(self) -> float:
        daily_return = self.cumulative_returns["Cumulative Returns"].pct_change()
        avg_daily_return = daily_return.mean()
        std_daily_return = daily_return.std()
        sharpe_ratio = np.sqrt(252) * avg_daily_return / std_daily_return
        return sharpe_ratio

    def compute_annual_returns(self) -> pd.DataFrame:
        unique_years = self.cumulative_returns.index.year.unique()
        annual_returns = []
        for year in unique_years:
            tmp_df = self.cumulative_returns.loc[
                self.cumulative_returns.index.year == year
            ]
            val = (
                tmp_df["Cumulative Returns"].iloc[-1]
                / tmp_df["Cumulative Returns"].iloc[0]
                - 1
            )
            annual_returns.append(val * 100)
        return pd.DataFrame({"Year": unique_years, "Annual Return (%)": annual_returns})


class Performance:
    def __init__(self, daily_log_returns: pd.Series):
        self.daily_log_returns = daily_log_returns
        self.daily_pct_returns = np.exp(daily_log_returns) - 1
        self.cumulative_returns = daily_log_returns.cumsum().apply(np.exp)

    def compute_max_dd(self) -> float:
        return self._compute_drawdown(self.cumulative_returns).max()

    def compute_longest_drawdown_period(self) -> float:
        drawdown = self._compute_drawdown(self.cumulative_returns)
        periods = np.diff(np.append(drawdown[drawdown == 0].index, drawdown.index[-1:]))
        return periods.max() / np.timedelta64(1, "D")

    def compute_sharpe_ratio(self, method: Literal["pct_chg", "log"] = "log") -> float:
        daily_returns = (
            self.daily_log_returns if method == "log" else self.daily_pct_returns
        )
        return np.sqrt(252) * daily_returns.mean() / daily_returns.std()

    def compute_cagr(self) -> float:
        cumulative_returns = self.cumulative_returns.dropna()
        days = (cumulative_returns.index[-1] - cumulative_returns.index[0]).days
        cagr = (cumulative_returns.iloc[-1] / cumulative_returns.iloc[0]) ** (
            365.0 / days
        ) - 1
        return cagr

    def _compute_drawdown(self, cumulative_returns) -> float:
        max_gross_performance = cumulative_returns.cummax()
        return max_gross_performance - cumulative_returns

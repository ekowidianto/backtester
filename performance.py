from typing import Literal

import numpy as np
import pandas as pd


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

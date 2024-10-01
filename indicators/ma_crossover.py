import numpy as np
import pandas as pd

from lib.visualisation import plot_ma_crossover_buy_sell

from .base import Indicator
from .helper import compute_sma


class Indicator_MA_Crossover(Indicator):
    def __init__(
        self,
        symbol: str,
        price_data: pd.DataFrame,
        short_period: int = 20,
        long_period: int = 60,
    ):
        super().__init__(price_data)
        self.symbol = symbol
        self.price_data = price_data
        self.short_period = short_period
        self.long_period = long_period

    def run(self):
        self._compute_internal_workings()
        self._compute_trading_positions()
        self._compute_buy_or_sell()

    def plot(self):
        plot_ma_crossover_buy_sell(self.symbol, self.get_price_data())

    def _compute_internal_workings(self):
        adj_close_price = self.price_data[["Adj Close"]]

        df_sma_short = compute_sma(adj_close_price.copy(), self.short_period)
        df_sma_long = compute_sma(adj_close_price.copy(), self.long_period)

        self.price_data["SMA_short"] = df_sma_short.values
        self.price_data["SMA_long"] = df_sma_long.values

    def _compute_trading_positions(self):
        self.price_data["trading_positions"] = np.where(
            self.price_data["SMA_short"] > self.price_data["SMA_long"], 1, -1
        )

    def _compute_buy_or_sell(self):
        self.price_data["buy_or_sell"] = (
            self.price_data["trading_positions"].diff().clip(-1, 1)
        )

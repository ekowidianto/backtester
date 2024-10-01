import numpy as np
import pandas as pd

from lib.visualisation import plot_sma_mean_reversion_buy_sell

from .base import Indicator
from .helper import compute_sma


class Indicator_SMA_Mean_Reversion(Indicator):
    def __init__(
        self,
        symbol: str,
        price_data: pd.DataFrame,
        sma_period: int = 41,
        threshold: int = 4,
    ):
        super().__init__(price_data)
        self.symbol = symbol
        self.price_data = price_data
        self.sma_period = sma_period
        self.threshold = threshold

    def run(self):
        self._compute_internal_workings()
        self._compute_trading_positions()
        self._compute_buy_or_sell()

    def plot(self):
        plot_sma_mean_reversion_buy_sell(
            self.symbol, self.get_price_data(), self.threshold
        )

    def _compute_internal_workings(self):
        adj_close_price = self.price_data[["Adj Close"]]

        df_sma = compute_sma(adj_close_price.copy(), self.sma_period)

        self.price_data["SMA"] = df_sma.values
        self.price_data["SMA_Price_Diff"] = (
            adj_close_price["Adj Close"] - self.price_data["SMA"]
        )

    def _compute_trading_positions(self):
        self.price_data["trading_positions"] = np.where(
            self.price_data["SMA_Price_Diff"] > self.threshold,
            -1,
            np.nan,  # overbought --> sell (short)
        )

        self.price_data["trading_positions"] = np.where(
            self.price_data["SMA_Price_Diff"]
            < -self.threshold,  # oversold --> buy (long)
            1,
            self.price_data["trading_positions"],
        )

        self.price_data["trading_positions"] = (
            np
            #           +                            -
            #           -                            +
            .where(
                self.price_data["SMA_Price_Diff"]
                * self.price_data["SMA_Price_Diff"].shift(1)
                < 0,  # oversold --> buy (long)
                0,
                self.price_data["trading_positions"],
            )
        )
        self.price_data["trading_positions"] = (
            self.price_data["trading_positions"].ffill().fillna(0)
        )

    def _compute_buy_or_sell(self):
        self.price_data["buy_or_sell"] = (
            self.price_data["trading_positions"].diff().clip(-1, 1)
        )

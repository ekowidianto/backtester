import numpy as np
import pandas as pd

from lib.visualisation import plot_macd_buy_sell

from .base import Indicator
from .helper import compute_ema, crossed_above, crossed_below


class Indicator_MACD(Indicator):
    def __init__(
        self,
        symbol: str,
        price_data: pd.DataFrame,
        short_period: int = 12,
        long_period: int = 26,
        signal_period: int = 9,
    ):
        super().__init__(price_data)
        self.symbol = symbol
        self.price_data = price_data
        self.short_period = short_period
        self.long_period = long_period
        self.signal_period = signal_period

    def run(self) -> pd.DataFrame:
        self._compute_internal_workings()
        self._compute_buy_or_sell()
        self._compute_trading_positions()
        return self.price_data.copy(deep=True)

    def plot(self):
        plot_macd_buy_sell(self.symbol, self.get_price_data())

    def _compute_internal_workings(self):
        adj_close_price = self.price_data[["Adj Close"]]
        df_ema_short = compute_ema(adj_close_price, self.short_period)
        df_ema_long = compute_ema(adj_close_price, self.long_period)

        df_macd = df_ema_short - df_ema_long
        df_signal_line = compute_ema(df_macd, self.signal_period)

        self.price_data["MACD"] = df_macd.values
        self.price_data["MACD Signal Line"] = df_signal_line.values

    def _compute_trading_positions(self):
        self.price_data["trading_positions"] = (
            self.price_data["buy_or_sell"].ffill().fillna(0)
        )

    def _compute_buy_or_sell(self):
        buy_signal = crossed_above(
            self.price_data["MACD"], self.price_data["MACD Signal Line"]
        )
        sell_signal = crossed_below(
            self.price_data["MACD"], self.price_data["MACD Signal Line"]
        )
        self.price_data["buy_or_sell"] = np.where(buy_signal, 1, np.nan)
        self.price_data["buy_or_sell"] = np.where(
            sell_signal, -1, self.price_data["buy_or_sell"]
        )

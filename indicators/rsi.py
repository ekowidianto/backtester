from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd

from lib.visualisation import plot_rsi_buy_sell

from .base import Indicator
from .helper import compute_sma, crossed_above, crossed_below


class Indicator_RSI(Indicator):
    def __init__(
        self,
        symbol: str,
        price_data: pd.DataFrame,
        start_date: datetime,
        period: int = 14,
        lower_threshold: float = 30,
        long_when: Literal[
            "crossed_above_lower_threshold",
            "crossed_below_lower_threshold",
        ] = "crossed_above_lower_threshold",
        upper_threshold: float = 70,
        short_when: Literal[
            "crossed_above_upper_threshold",
            "crossed_below_upper_threshold",
        ] = "crossed_below_upper_threshold",
        long_threshold_exit: float = None,
        short_threshold_exit: float = None,
    ):
        super().__init__(price_data, start_date)
        self.symbol = symbol
        self.price_data = price_data
        self.period = period
        self.price_data["RSI_lower_threshold"] = lower_threshold
        self.price_data["RSI_upper_threshold"] = upper_threshold
        self.long_when = long_when
        self.short_when = short_when
        self.long_threshold_exit = long_threshold_exit
        self.short_threshold_exit = short_threshold_exit

    def run(self) -> pd.DataFrame:
        self._compute_internal_workings()
        self._compute_trading_positions()
        self._compute_buy_or_sell()
        return self.price_data.copy(deep=True)

    def plot(self):
        plot_rsi_buy_sell(self.symbol, self.get_price_data())

    def _compute_internal_workings(self):
        delta = self.price_data["Adj Close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # Calculate rolling averages of gains and losses
        avg_gain = gain.rolling(window=self.period).mean()
        avg_loss = loss.rolling(window=self.period).mean()
        for i in range(self.period, len(delta)):
            avg_gain[i] = (avg_gain[i - 1] * 13 + gain[i]) / self.period
            avg_loss[i] = (avg_loss[i - 1] * 13 + loss[i]) / self.period

        # Compute the Relative Strength (RS)
        rs = avg_gain / avg_loss

        # Compute the RSI
        rsi = 100 - (100 / (1 + rs))
        self.price_data["RSI"] = rsi

    def _compute_trading_positions(self):
        if self.long_when == "crossed_above_lower_threshold":
            buy_signal = crossed_above(
                self.price_data["RSI"], self.price_data["RSI_lower_threshold"]
            )
        else:
            buy_signal = crossed_below(
                self.price_data["RSI"], self.price_data["RSI_lower_threshold"]
            )

        if self.short_when == "crossed_below_upper_threshold":
            sell_signal = crossed_below(
                self.price_data["RSI"], self.price_data["RSI_upper_threshold"]
            )
        else:
            sell_signal = crossed_above(
                self.price_data["RSI"], self.price_data["RSI_upper_threshold"]
            )

        self.price_data["trading_positions"] = np.where(buy_signal, 1, np.nan)
        self.price_data["trading_positions"] = np.where(
            sell_signal, -1, self.price_data["trading_positions"]
        )
        if (
            self.long_threshold_exit is not None
            or self.short_threshold_exit is not None
        ):
            self.price_data["trading_positions"] = np.where(
                (
                    (self.price_data["RSI"] - self.long_threshold_exit)
                    * (self.price_data["RSI"].shift(1) - self.long_threshold_exit)
                    < 0
                ),
                0,
                self.price_data["trading_positions"],
            )
        self.price_data["trading_positions"] = (
            self.price_data["trading_positions"].ffill().fillna(0)
        )

    def _compute_buy_or_sell(self):
        self.price_data["buy_or_sell"] = (
            self.price_data["trading_positions"].diff().clip(-1, 1)
        )

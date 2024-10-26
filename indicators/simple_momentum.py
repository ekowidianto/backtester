from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd

from lib.visualisation import plot_simple_momentum

from .base import Indicator


class Indicator_Simple_Momentum(Indicator):
    def __init__(
        self,
        symbol: str,
        price_data: pd.DataFrame,
        start_date: datetime,
        position_type: Literal["long", "short", "long_short"] = "long_short",
    ):
        super().__init__(price_data, start_date, position_type)
        self.symbol = symbol
        self.price_data = price_data

    def run(self):
        self._compute_internal_workings()
        self._compute_trading_positions()
        self._compute_buy_or_sell()

    def plot(self):
        plot_simple_momentum(self.symbol, self.get_price_data())

    def _compute_internal_workings(self):
        self.price_data["Log Return"] = np.log(
            self.price_data["Adj Close"] / self.price_data["Adj Close"].shift(1)
        )

    def _compute_trading_positions(self):
        self.price_data["trading_positions"] = np.sign(self.price_data["Log Return"])
        super()._compute_trading_positions()

    def _compute_buy_or_sell(self):
        self.price_data["buy_or_sell"] = (
            self.price_data["trading_positions"].diff().clip(-1, 1)
        ).fillna(0)

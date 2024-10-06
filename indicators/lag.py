import numpy as np
import pandas as pd

from lib.visualisation import plot_lag

from .base import Indicator


class Indicator_Lag(Indicator):
    def __init__(
        self,
        symbol: str,
        price_data: pd.DataFrame,
        lag_days: int = 2,
    ):
        super().__init__(price_data)
        self.symbol = symbol
        self.price_data = price_data
        self.lag_days = lag_days
        self.lag_col_names = []
        self.ols = {"return": None, "sign": None}

    def run(self):
        self._compute_internal_workings()
        self._compute_trading_positions()
        self._compute_buy_or_sell()

    def plot(self):
        plot_lag(self.symbol, self.get_price_data())

    def _compute_internal_workings(self):
        self.price_data["Log Return"] = np.log(
            self.price_data["Adj Close"] / self.price_data["Adj Close"].shift(1)
        )
        for lag in range(1, self.lag_days + 1):
            COL = f"lag_{lag}"
            self.price_data[COL] = self.price_data["Log Return"].shift(lag)
            self.lag_col_names.append(COL)

        self.price_data.dropna(inplace=True)

        self.ols["return"] = np.linalg.lstsq(
            self.price_data[self.lag_col_names],
            self.price_data["Log Return"],
            rcond=None,
        )[0]

        self.price_data["PREDICTION_RETURN"] = np.dot(
            self.price_data[self.lag_col_names], self.ols["return"]
        )

        self.ols["sign"] = np.linalg.lstsq(
            self.price_data[self.lag_col_names],
            np.sign(self.price_data["Log Return"]),
            rcond=None,
        )[0]

        self.price_data["PREDICTION_SIGN"] = np.sign(
            np.dot(self.price_data[self.lag_col_names], self.ols["sign"])
        )

    def _compute_trading_positions(self):
        self.price_data["trading_positions"] = self.price_data["PREDICTION_SIGN"].shift(
            -1
        )

    def _compute_buy_or_sell(self):
        self.price_data["buy_or_sell"] = (
            self.price_data["trading_positions"].diff().clip(-1, 1)
        )

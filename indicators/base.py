from abc import ABC, abstractmethod

import pandas as pd


class Indicator(ABC):
    def __init__(self, price_data: pd.DataFrame):
        self.price_data = price_data.copy(deep=True)

    def get_price_data(self) -> pd.DataFrame:
        return self.price_data.copy(deep=True).dropna()

    @abstractmethod
    def run(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def _compute_internal_workings(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def _compute_trading_positions(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def _compute_buy_or_sell(self) -> pd.DataFrame:
        pass

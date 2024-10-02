from abc import ABC, abstractmethod

import pandas as pd


class Indicator(ABC):
    def __init__(self, price_data: pd.DataFrame):
        self.price_data = price_data.copy(deep=True)

    def get_price_data(self) -> pd.DataFrame:
        return self.price_data.copy(deep=True).dropna()

    def get_trading_opportunity(self) -> pd.DataFrame:
        trading_opp_dct = {"Buy": 0, "Sell": 0, "Total": 0}
        buy_or_sell = self.get_price_data()["buy_or_sell"]
        trading_opp_dct["Buy"] = (buy_or_sell == 1).sum()
        trading_opp_dct["Sell"] = (buy_or_sell == -1).sum()
        trading_opp_dct["Total"] = trading_opp_dct["Buy"] + trading_opp_dct["Sell"]
        return trading_opp_dct

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

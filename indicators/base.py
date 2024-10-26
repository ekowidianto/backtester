from abc import ABC, abstractmethod
from datetime import datetime
from typing import Literal

import numpy as np
import pandas as pd


class Indicator(ABC):
    def __init__(
        self,
        price_data: pd.DataFrame,
        start_date: datetime,
        position_type: Literal["long", "short", "long_short"] = "long_short",
    ):
        self.price_data = price_data.copy(deep=True)
        self.start_date = start_date
        self.position_type = position_type
        self.long = 1 if position_type in ["long", "long_short"] else 0
        self.short = -1 if position_type in ["short", "long_short"] else 0

    def get_price_data(self) -> pd.DataFrame:
        df = self.price_data.copy(deep=True)
        df = df[df.index.get_level_values("Date") >= self.start_date]
        return df.copy(deep=True)

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
    def _compute_internal_workings(self):
        pass

    def _compute_trading_positions(self):
        self.price_data["trading_positions"] = self.price_data[
            "trading_positions"
        ].clip(self.short, self.long)

    @abstractmethod
    def _compute_buy_or_sell(self):
        pass


class IndicatorsCombined:
    def __init__(
        self,
        indicators: list[Indicator],
        min_indicators_to_long: int,
        min_indicators_to_short: int,
    ):
        self.indicators = indicators
        self.indicators_price_data = [
            indicator.get_price_data() for indicator in indicators
        ]
        self.min_indicators_to_long = min_indicators_to_long
        self.min_indicators_to_short = min_indicators_to_short
        self.price_data = None
        self.validate_indicators()
        self.combine_strategies()

    def validate_indicators(self):
        if len(self.indicators_price_data) < 2:
            raise Exception("At least 2 indicators are required for comparison")

        all_indices_equal = all(
            df.index.equals(self.indicators_price_data[0].index)
            for df in self.indicators_price_data
        )
        if not all_indices_equal:
            raise Exception("Date indices of all indicators should be equal")

    def combine_strategies(self):
        df = self.indicators_price_data[0][["Adj Close"]]
        agg_trading_positions = sum(
            df["trading_positions"] for df in self.indicators_price_data
        )
        trading_positions = np.where(
            agg_trading_positions >= self.min_indicators_to_long,
            1,
            0,
            # np.nan,
        )
        trading_positions = np.where(
            agg_trading_positions <= -self.min_indicators_to_short,
            -1,
            trading_positions,
        )
        df["trading_positions"] = trading_positions
        df["trading_positions"].ffill(inplace=True)
        df["trading_positions"].fillna(0, inplace=True)
        df["buy_or_sell"] = df["trading_positions"].diff()
        df["buy_or_sell"].fillna(0, inplace=True)
        self.price_data = df

    def get_price_data(self) -> pd.DataFrame:
        return self.price_data.copy(deep=True)

from datetime import datetime, timedelta
from typing import Literal

import pandas as pd
from indicators import (
    Indicator_Lag,
    Indicator_MA_Crossover,
    Indicator_MACD,
    Indicator_RSI,
    Indicator_Simple_Momentum,
    Indicator_SMA_Mean_Reversion,
)
from utils import get_data


class Signals:
    IndicatorMap = {
        "MACD": Indicator_MACD,
        "MA_crossover": Indicator_MA_Crossover,
        "SMA_mean_reversion": Indicator_SMA_Mean_Reversion,
        "Lag": Indicator_Lag,
        "Simple_momentum": Indicator_Simple_Momentum,
        "RSI": Indicator_RSI,
    }

    def __init__(
        self,
        symbol: str,
        sd: datetime,
        ed: datetime,
        lookback: int = 0,
        position_type: Literal["long", "short", "long_short"] = "long_short",
    ):
        self.symbol = symbol
        self.sd = sd
        self.ed = ed
        self.lookback = lookback
        self.data = self.get_data()
        self.data_with_indicator = {}
        self.position_type = position_type

    def get_data(self) -> pd.DataFrame:
        all_data = get_data(
            [self.symbol], self.sd - timedelta(days=self.lookback), self.ed
        )
        all_data = all_data.ffill().bfill()

        all_data["Adj Ratio"] = all_data["Adj Close"] / all_data["Close"]
        all_data[["Adj Open", "Adj High", "Adj Low"]] = all_data[
            ["Open", "High", "Low"]
        ].multiply(all_data["Adj Ratio"], axis="index")

        return all_data

    def run_indicator_for(
        self,
        indicator: Literal[
            "MACD",
            "MA_crossover",
            "SMA_mean_reversion",
            "Lag",
            "Simple_momentum",
            "RSI",
        ],
        **indicator_params
    ) -> pd.DataFrame:
        indicator = self.IndicatorMap[indicator](
            self.symbol,
            self.data,
            **indicator_params,
            start_date=self.sd,
            position_type=self.position_type
        )
        indicator.run()
        self.data_with_indicator[indicator] = indicator
        return self.data_with_indicator[indicator]

    def get_indicator_for(
        self,
        indicator: Literal[
            "MACD",
            "MA_crossover",
            "SMA_mean_reversion",
            "Lag",
            "Simple_momentum",
            "RSI",
        ],
    ) -> pd.DataFrame:
        return self.data_with_indicator.get(indicator, None)

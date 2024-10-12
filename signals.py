from datetime import datetime, timedelta
from typing import Literal

import pandas as pd
from indicators import (
    Indicator_Lag,
    Indicator_MA_Crossover,
    Indicator_MACD,
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
    }

    def __init__(self, symbol: str, sd: datetime, ed: datetime, lookback: int = 0):
        self.symbol = symbol
        self.sd = sd
        self.ed = ed
        self.dates = pd.date_range(sd, ed)
        self.dates_with_lookback = pd.date_range(sd - timedelta(days=lookback), ed)
        self.data = self.get_data()
        self.data_with_indicator = {}

    def get_data(self) -> pd.DataFrame:
        all_data = get_data([self.symbol], self.sd, self.ed)
        all_data = all_data.ffill().bfill()

        all_data["Adj Ratio"] = all_data["Adj Close"] / all_data["Close"]
        all_data[["Adj Open", "Adj High", "Adj Low"]] = all_data[
            ["Open", "High", "Low"]
        ].multiply(all_data["Adj Ratio"], axis="index")

        return all_data

    def run_indicator_for(
        self,
        indicator: Literal[
            "MACD", "MA_crossover", "SMA_mean_reversion", "Lag", "Simple_momentum"
        ],
        **indicator_params
    ) -> pd.DataFrame:
        indicator = self.IndicatorMap[indicator](
            self.symbol, self.data, **indicator_params
        )
        indicator.run()
        self.data_with_indicator[indicator] = indicator
        return self.data_with_indicator[indicator]

    def get_indicator_for(
        self,
        indicator: Literal[
            "MACD", "MA_crossover", "SMA_mean_reversion", "Lag", "Simple_momentum"
        ],
    ) -> pd.DataFrame:
        return self.data_with_indicator.get(indicator, None)

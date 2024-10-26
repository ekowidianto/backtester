import numpy as np
import pandas as pd


def compute_sma(df: pd.DataFrame, period: int) -> pd.DataFrame:
    return df.rolling(period, min_periods=1, center=False).mean()


def compute_ema(df: pd.DataFrame, period: int) -> pd.DataFrame:
    df_ema = df.ewm(span=period).mean()
    return df_ema


def crossed(series1, series2, direction=None):
    if isinstance(series1, np.ndarray):
        series1 = pd.Series(series1)

    if isinstance(series2, (float, int, np.ndarray, np.integer, np.floating)):
        series2 = pd.Series(index=series1.index, data=series2)

    if direction is None or direction == "above":
        above = pd.Series((series1 > series2) & (series1.shift(1) <= series2.shift(1)))

    if direction is None or direction == "below":
        below = pd.Series((series1 < series2) & (series1.shift(1) >= series2.shift(1)))

    if direction is None:
        return above | below

    return above if direction == "above" else below


def crossed_above(series1, series2):
    return crossed(series1, series2, "above")


def crossed_below(series1, series2):
    return crossed(series1, series2, "below")

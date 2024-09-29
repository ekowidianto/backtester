from datetime import timedelta

import numpy as np
import pandas as pd
from utils import get_data


class Indicators:
    def __init__(self, symbol, sd, ed, lookback=100):
        self.symbols = [symbol]
        self.sd = sd
        self.ed = ed
        self.dates = pd.date_range(sd, ed)
        self.dates_with_lookback = pd.date_range(sd - timedelta(days=lookback), ed)
        self.get_data()

    def generate_indicators(self):
        self.indicator_dema_crossover(10, 30)  # Trend
        self.indicator_percentage_price_oscillator(12, 26)  # Trend
        self.indicator_bollinger_bands_percentage(20)  # Volatility
        self.indicator_rsi(14)  # Momentum
        self.indicator_mfi(14)  # Volume
        # self.indicator_cci(20) # Others

    def get_data(self):
        open_data = (
            get_data(self.symbols, self.dates_with_lookback, colname="Open")
            .ffill()
            .bfill()[self.symbols]
        )
        close_data = (
            get_data(self.symbols, self.dates_with_lookback, colname="Close")
            .ffill()
            .bfill()[self.symbols]
        )
        high_data = (
            get_data(self.symbols, self.dates_with_lookback, colname="High")
            .ffill()
            .bfill()[self.symbols]
        )
        low_data = (
            get_data(self.symbols, self.dates_with_lookback, colname="Low")
            .ffill()
            .bfill()[self.symbols]
        )
        volume_data = (
            get_data(self.symbols, self.dates_with_lookback, colname="Volume")
            .ffill()
            .bfill()[self.symbols]
        )
        adjusted_close_data = (
            get_data(self.symbols, self.dates_with_lookback)
            .ffill()
            .bfill()[self.symbols]
        )
        adjustment_ratio = adjusted_close_data[self.symbols] / close_data[self.symbols]

        self.adjusted_open_data = open_data * adjustment_ratio
        self.adjusted_close_data = adjusted_close_data
        self.adjusted_high_data = high_data * adjustment_ratio
        self.adjusted_low_data = low_data * adjustment_ratio
        self.volume_data = volume_data

    def indicator_dema_crossover(self, short_period=21, long_period=55):
        df_prices = self.adjusted_close_data.copy()
        df_dema_short = self.compute_dema(df_prices, short_period)
        df_dema_long = self.compute_dema(df_prices, long_period)

        df_dema_ratio = df_dema_short / df_dema_long
        df_dema_ratio = df_dema_ratio.truncate(before=self.sd, after=self.ed)
        return df_dema_ratio.iloc[:, 0].values

    def indicator_percentage_price_oscillator(self, short_period=12, long_period=26):
        df_prices = self.adjusted_close_data.copy()
        df_ema_short = self.compute_ema(df_prices, short_period)
        df_ema_long = self.compute_ema(df_prices, long_period)
        df_ppo = (df_ema_short - df_ema_long) / df_ema_long * 100
        df_signal_line = self.compute_ema(df_ppo, 9)

        df_ppo_ratio = df_ppo / df_prices
        df_ppo_ratio = df_ppo_ratio.truncate(before=self.sd, after=self.ed)
        return df_ppo_ratio.iloc[:, 0].values

    def indicator_bollinger_bands_percentage(self, period):
        df_prices = self.adjusted_close_data.copy()
        df_sma = self.compute_sma(df_prices, period)
        df_std = df_prices.rolling(period).std()
        df_upper_bound = df_sma + 2 * df_std
        df_lower_bound = df_sma - 2 * df_std
        df_BB_percentage = (df_prices - df_lower_bound) / (
            df_upper_bound - df_lower_bound
        )

        df_BB_percentage = df_BB_percentage.truncate(before=self.sd, after=self.ed)
        return df_BB_percentage.iloc[:, 0].values

    def indicator_rsi(self, period=14):
        df_prices = self.adjusted_close_data.copy()
        df_prices_change = (df_prices - df_prices.shift(1))[1:]
        df_prices_gain = df_prices_change[df_prices_change > 0].fillna(0)
        df_prices_loss = -df_prices_change[df_prices_change < 0].fillna(0)
        df_prices_gain_avg = pd.DataFrame().reindex_like(df_prices)
        df_prices_gain_avg = self.compute_sma(df_prices_gain, period)[:period]
        df_prices_gain_avg = (
            pd.concat([df_prices_gain_avg, df_prices_gain[period:]])
            .ewm(com=period - 1, adjust=False)
            .mean()
        )
        df_prices_loss_avg = pd.DataFrame().reindex_like(df_prices)
        df_prices_loss_avg = self.compute_sma(df_prices_loss, period)[:period]
        df_prices_loss_avg = (
            pd.concat([df_prices_loss_avg, df_prices_loss[period:]])
            .ewm(com=period - 1, adjust=False)
            .mean()
        )
        df_rs = df_prices_gain_avg / df_prices_loss_avg
        df_rsi = 100 - (100 / (1 + df_rs))
        df_rsi[df_rsi == np.inf] = 100

        df_rsi = df_rsi.truncate(before=self.sd, after=self.ed)
        return df_rsi.iloc[:, 0].values

    def indicator_mfi(self, period=14):
        df_prices_close = self.adjusted_close_data.copy()
        df_prices_high = self.adjusted_high_data.copy()
        df_prices_low = self.adjusted_low_data.copy()
        df_prices_typical = (df_prices_close + df_prices_high + df_prices_low) / 3
        df_volume = self.volume_data.copy()
        df_prices_typical_changes = df_prices_typical - df_prices_typical.shift(1)
        df_raw_money_flow = df_volume * df_prices_typical
        df_raw_money_flow_positive = df_raw_money_flow[
            df_prices_typical_changes > 0
        ].fillna(0)
        df_raw_money_flow_negative = df_raw_money_flow[
            df_prices_typical_changes < 0
        ].fillna(0)
        df_raw_money_flow_positive_n_period = df_raw_money_flow_positive.rolling(
            period
        ).sum()
        df_raw_money_flow_negative_n_period = df_raw_money_flow_negative.rolling(
            period
        ).sum()
        df_money_flow_ratio_n = (
            df_raw_money_flow_positive_n_period / df_raw_money_flow_negative_n_period
        )
        df_money_flow_index_n = 100 - (100 / (1 + df_money_flow_ratio_n))
        df_money_flow_index_n[df_money_flow_index_n == np.inf] = 100

        df_money_flow_index_n = df_money_flow_index_n.truncate(
            before=self.sd, after=self.ed
        )
        return df_money_flow_index_n.iloc[:, 0].values

    def indicator_cci(self, period=20):
        df_prices_close = self.adjusted_close_data.copy()
        df_prices_high = self.adjusted_high_data.copy()
        df_prices_low = self.adjusted_low_data.copy()
        df_prices_typical = (df_prices_close + df_prices_high + df_prices_low) / 3
        df_prices_typical_sma = self.compute_sma(df_prices_typical, period)
        df_mean_deviation = df_prices_typical.rolling(period).apply(
            lambda x: pd.Series(x).mad(), raw=False
        )
        df_cci = (df_prices_typical - df_prices_typical_sma) / (
            0.015 * df_mean_deviation
        )

    # Helper
    def compute_sma(self, df, period):
        return df.rolling(period).mean()

    def compute_ema(self, df, period):
        df_ema = pd.DataFrame().reindex_like(df)
        df_ema = self.compute_sma(df, period)[:period]
        df_ema = pd.concat([df_ema, df[period:]]).ewm(span=period, adjust=False).mean()
        return df_ema

    def compute_dema(self, df, days):
        df_ema_price = self.compute_ema(df, days)
        df_ema_ema_price = self.compute_ema(df_ema_price, days)
        return 2 * df_ema_price - df_ema_ema_price

import numpy as np
import pandas as pd

def make_supervised(hourly):
    frame = hourly.copy()
    idx = frame.index
    frame["hour"] = idx.hour
    frame["dayofweek"] = idx.dayofweek
    frame["month"] = idx.month
    frame["hour_sin"] = np.sin(2 * np.pi * frame["hour"] / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * frame["hour"] / 24)
    frame["dow_sin"] = np.sin(2 * np.pi * frame["dayofweek"] / 7)
    frame["dow_cos"] = np.cos(2 * np.pi * frame["dayofweek"] / 7)
    for lag in (1, 23, 24, 168):
        frame[f"lag_{lag}"] = frame["consumption_kw"].shift(lag)
    frame["rolling_mean_24"] = frame["consumption_kw"].shift(1).rolling(24).mean()
    frame["rolling_std_24"] = frame["consumption_kw"].shift(1).rolling(24).std()
    frame["target"] = frame["consumption_kw"].shift(-1)
    return frame.replace([np.inf, -np.inf], np.nan).dropna()

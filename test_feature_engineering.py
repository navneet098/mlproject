import pandas as pd
from src.feature_engineering import make_supervised

def test_feature_engineering_shapes_and_target_shift():
    idx = pd.date_range("2020-01-01", periods=24*10, freq="h")
    frame = pd.DataFrame({"consumption_kw": range(len(idx))}, index=idx)
    out = make_supervised(frame)
    assert "lag_24" in out.columns
    assert "target" in out.columns
    assert out.iloc[0]["target"] == out.iloc[0]["consumption_kw"] + 1

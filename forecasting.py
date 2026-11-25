from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.feature_engineering import make_supervised

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month",
            "lag_1", "lag_23", "lag_24", "lag_168", "rolling_mean_24", "rolling_std_24"]

def split_chronologically(data):
    n = len(data)
    a, b = int(0.70*n), int(0.85*n)
    return data.iloc[:a], data.iloc[a:b], data.iloc[b:]

def run_forecasting(hourly):
    data = make_supervised(hourly)
    train, val, test = split_chronologically(data)
    X_train, y_train = train[FEATURES], train["target"]
    X_val, y_val = val[FEATURES], val["target"]
    X_test, y_test = test[FEATURES], test["target"]

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=250, max_depth=18, min_samples_leaf=2,
            random_state=42, n_jobs=-1
        )
    }
    rows, predictions = [], {}
    # Validation data is reserved for diagnostics; final reported metrics use the untouched test period.
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        predictions[name] = pred
        rows.append({"model": name,
                     "MAE_kW": mean_absolute_error(y_test, pred),
                     "RMSE_kW": mean_squared_error(y_test, pred) ** 0.5})
        joblib.dump(model, ROOT / "models" / (name.lower().replace(" ", "_") + ".joblib"))

    baseline = test["lag_23"].to_numpy()
    predictions["Previous-day baseline"] = baseline
    rows.append({"model": "Previous-day baseline",
                 "MAE_kW": mean_absolute_error(y_test, baseline),
                 "RMSE_kW": mean_squared_error(y_test, baseline) ** 0.5})

    result = pd.DataFrame({"timestamp": test.index + pd.Timedelta(hours=1), "actual_kw": y_test.to_numpy(),
                           **{f"{k}_pred_kw": v for k, v in predictions.items()}})
    result.to_csv(ROOT / "results" / "forecast_comparison.csv", index=False)

    metrics = {r["model"]: {"MAE_kW": float(r["MAE_kW"]), "RMSE_kW": float(r["RMSE_kW"])} for r in rows}
    with open(ROOT / "results" / "metrics.json", "w") as f:
        json.dump({"forecasting_test_metrics": metrics,
                   "split": {"train": 0.70, "validation": 0.15, "test": 0.15},
                   "test_rows": int(len(test))}, f, indent=2)

    plt.figure(figsize=(12, 4))
    view = result.iloc[:min(24*14, len(result))]
    plt.plot(view["timestamp"], view["actual_kw"], label="Actual", linewidth=1)
    plt.plot(view["timestamp"], view["Random Forest_pred_kw"], label="Random Forest", linewidth=1)
    plt.xlabel("Time"); plt.ylabel("Hourly mean active power (kW)")
    plt.title("WattWise: test-period next-hour forecast")
    plt.legend(); plt.tight_layout()
    plt.savefig(ROOT / "results" / "figures" / "forecast_test.png", dpi=160)
    plt.close()
    return data, train, val, test, result

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

ROOT = Path(__file__).resolve().parents[1]

def rolling_mad_flags(series, window=24*7, threshold=3.5):
    prior = series.shift(1)
    med = prior.rolling(window, min_periods=24).median()
    mad = (prior - med).abs().rolling(window, min_periods=24).median()
    robust_z = 0.6745 * (series - med) / mad.replace(0, np.nan)
    return (robust_z.abs() > threshold).fillna(False), robust_z

def run_anomaly_detection(hourly, train, val, test, forecast_result):
    # Raw-consumption Isolation Forest is fitted only on the training period.
    raw_if = IsolationForest(n_estimators=250, contamination=0.01, random_state=42)
    raw_if.fit(train[["target"]])
    raw_test = test[["target"]]
    raw_flags = raw_if.predict(raw_test) == -1
    joblib.dump(raw_if, ROOT / "models" / "isolation_forest_raw.joblib")

    # Fit the residual detector on validation residuals, then score the held-out test residuals.
    rf_path = ROOT / "models" / "random_forest.joblib"
    rf = joblib.load(rf_path)
    val_resid = val["target"].to_numpy() - rf.predict(val[["hour_sin","hour_cos","dow_sin","dow_cos","month",
        "lag_1","lag_23","lag_24","lag_168","rolling_mean_24","rolling_std_24"]])
    test_resid = test["target"].to_numpy() - forecast_result["Random Forest_pred_kw"].to_numpy()
    resid_if = IsolationForest(n_estimators=250, contamination=0.01, random_state=42)
    resid_if.fit(val_resid.reshape(-1, 1))
    resid_flags = resid_if.predict(test_resid.reshape(-1, 1)) == -1
    joblib.dump(resid_if, ROOT / "models" / "isolation_forest_residual.joblib")

    target_index = test.index + pd.Timedelta(hours=1)
    rolling_flags_all, robust_z_all = rolling_mad_flags(hourly["consumption_kw"])
    rolling_flags = rolling_flags_all.reindex(target_index).fillna(False)
    robust_z = robust_z_all.reindex(target_index)
    out = pd.DataFrame({
        "timestamp": target_index,
        "consumption_kw": test["target"].to_numpy(),
        "rolling_mad_flag": rolling_flags.to_numpy(),
        "rolling_robust_z": robust_z.to_numpy(),
        "isolation_forest_raw_flag": raw_flags,
        "forecast_residual_kw": test_resid,
        "isolation_forest_residual_flag": resid_flags
    })
    out.to_csv(ROOT / "results" / "anomaly_results.csv", index=False)
    summary = {
        "test_rows": int(len(out)),
        "alert_rates": {
            "rolling_mad": float(out["rolling_mad_flag"].mean()),
            "isolation_forest_raw": float(out["isolation_forest_raw_flag"].mean()),
            "isolation_forest_residual": float(out["isolation_forest_residual_flag"].mean())
        },
        "ground_truth_note": "No verified fault labels are supplied; alert rates are descriptive and are not precision/recall or confirmed false-alarm rates."
    }
    import json
    metrics_path = ROOT / "results" / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["anomaly_test_summary"] = summary
    metrics_path.write_text(json.dumps(metrics, indent=2))

    plt.figure(figsize=(12, 4))
    plt.plot(out["timestamp"], out["consumption_kw"], linewidth=.7, label="Hourly consumption")
    flagged = out[out["isolation_forest_raw_flag"]]
    plt.scatter(flagged["timestamp"], flagged["consumption_kw"], s=10, label="Isolation Forest flags")
    plt.xlabel("Time"); plt.ylabel("kW"); plt.title("WattWise: candidate raw-consumption anomalies")
    plt.legend(); plt.tight_layout()
    plt.savefig(ROOT / "results" / "figures" / "anomaly_timeline.png", dpi=160)
    plt.close()
    return out

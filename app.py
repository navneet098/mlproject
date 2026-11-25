from pathlib import Path
import json
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
st.set_page_config(page_title="WattWise", layout="wide")
st.title("⚡ WattWise")
st.caption("Household electricity forecasting and unsupervised anomaly detection")
metrics_path = ROOT / "results" / "metrics.json"
forecast_path = ROOT / "results" / "forecast_comparison.csv"
anomaly_path = ROOT / "results" / "anomaly_results.csv"

if not metrics_path.exists() or not forecast_path.exists() or not anomaly_path.exists():
    st.warning("No experiment outputs found. Run `python -m src.run_pipeline` from the repository root first.")
    st.stop()

metrics = json.loads(metrics_path.read_text())
st.subheader("Forecasting test metrics")
metric_df = pd.DataFrame(metrics["forecasting_test_metrics"]).T
st.dataframe(metric_df, use_container_width=True)

forecast = pd.read_csv(forecast_path, parse_dates=["timestamp"])
st.subheader("Actual versus forecast")
cols = [c for c in forecast.columns if c.endswith("_pred_kw")]
selected = st.selectbox("Forecast model", cols, index=cols.index("Random Forest_pred_kw") if "Random Forest_pred_kw" in cols else 0)
chart = forecast.set_index("timestamp")[["actual_kw", selected]].rename(columns={selected: "prediction"})
st.line_chart(chart)

anomalies = pd.read_csv(anomaly_path, parse_dates=["timestamp"])
st.subheader("Candidate anomaly alerts")
flag_col = st.selectbox("Detection method", ["rolling_mad_flag", "isolation_forest_raw_flag", "isolation_forest_residual_flag"])
flagged = anomalies[anomalies[flag_col].astype(bool)]
st.metric("Flagged test hours", f"{len(flagged):,}", f"{len(flagged)/max(len(anomalies),1):.2%} of test hours")
st.dataframe(flagged.head(200), use_container_width=True)
st.caption("Alerts are candidates for human review. The dataset contains no verified appliance-fault labels.")

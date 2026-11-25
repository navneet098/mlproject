from pathlib import Path
import shutil
from src.data_preprocessing import load_hourly_data
from src.forecasting import run_forecasting
from src.anomaly_detection import run_anomaly_detection

ROOT = Path(__file__).resolve().parents[1]

def main():
    for folder in ("models", "results/figures"):
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    hourly = load_hourly_data()
    data, train, val, test, forecast = run_forecasting(hourly)
    anomalies = run_anomaly_detection(hourly, train, val, test, forecast)
    checkpoint_dir = ROOT / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    for model_file in (ROOT / "models").glob("*.joblib"):
        shutil.copy2(model_file, checkpoint_dir / model_file.name)
    print(f"Pipeline complete: {len(hourly):,} hourly rows; {len(test):,} test examples.")
    print("Inspect results/metrics.json, result CSV files and results/figures/.")

if __name__ == "__main__":
    main()

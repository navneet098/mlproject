from pathlib import Path
import zipfile
import requests
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
RAW_FILE = RAW_DIR / "household_power_consumption.txt"
ZIP_FILE = RAW_DIR / "household_power_consumption.txt.zip"
UCI_URL = "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip"

def download_dataset():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_FILE.exists():
        return RAW_FILE
    response = requests.get(UCI_URL, timeout=120)
    response.raise_for_status()
    ZIP_FILE.write_bytes(response.content)
    with zipfile.ZipFile(ZIP_FILE) as zf:
        member = next((n for n in zf.namelist() if n.lower().endswith("household_power_consumption.txt")), None)
        if member is None:
            raise FileNotFoundError("Expected household_power_consumption.txt inside UCI archive")
        RAW_FILE.write_bytes(zf.read(member))
    return RAW_FILE

def load_hourly_data():
    path = download_dataset()
    df = pd.read_csv(path, sep=";", na_values="?", low_memory=False)
    df["timestamp"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str),
        format="%d/%m/%Y %H:%M:%S", errors="coerce"
    )
    df["Global_active_power"] = pd.to_numeric(df["Global_active_power"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    df = df.drop_duplicates(subset="timestamp").set_index("timestamp")
    hourly = df["Global_active_power"].resample("h").mean().rename("consumption_kw").to_frame()
    hourly = hourly.dropna()
    hourly.index.name = "timestamp"
    return hourly

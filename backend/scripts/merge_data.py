import pandas as pd
from pathlib import Path

EIA = "backend/data/processed/eia/FPL_DEMAND_2019-01-01T00_2025-09-20T00.csv"             # your first file
WX  = "backend/data/processed/nasa/T2M,PRECTOT,WS2M,ALLSKY_SFC_SW_DWN,RH2M_20190101_20250920.csv"      # your second file
OUT = "backend/data/processed/merged/merged_hourly.parquet"

def load_eia(path=EIA):
    df = pd.read_csv(path)
    # keep only FPL demand rows
    df = df[(df["respondent"] == "FPL") & (df["type-name"].str.contains("Demand", case=False))]
    # parse time; EIA 'period' is end-of-interval. Make it naive UTC or local consistently.
    df["timestamp"] = pd.to_datetime(df["period"], utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    df = df.rename(columns={"value": "load_mw"}).copy()
    # EIA gives MWh/interval; for hourly, numeric value is fine as MW-equivalent.
    df["load_mw"] = pd.to_numeric(df["load_mw"], errors="coerce")
    return df[["timestamp", "load_mw"]].sort_values("timestamp").reset_index(drop=True)

def load_weather(path=WX, start="2019-01-01 00:00:00"):
    cols = {
        "ALLSKY_SFC_SW_DWN": "ghi_kwhm2",
        "RH2M": "rh",
        "T2M": "temp_c",
        "WS2M": "wind_mps",
        "PRECTOTCORR": "precip_mm",
    }
    df = pd.read_csv(path)
    df = df.rename(columns=cols)

    # if no timestamp, build one
    if "timestamp" not in df.columns:
        # assume continuous hourly data starting at 'start'
        n = len(df)
        ts = pd.date_range(start=start, periods=n, freq="H")
        df.insert(0, "timestamp", ts)

    numeric = ["ghi_kwhm2","rh","temp_c","wind_mps","precip_mm"]
    for c in numeric:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df[["timestamp"] + numeric].sort_values("timestamp").reset_index(drop=True)

def merge_and_save():
    eia = load_eia()
    wx  = load_weather()
    # align to the **start** of the hour if needed:
    # eia["timestamp"] -= pd.Timedelta(hours=1)
    df = pd.merge_asof(
        eia.sort_values("timestamp"),
        wx.sort_values("timestamp"),
        on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta("30min"),
    )
    # basic cleaning
    df = df.dropna(subset=["load_mw"]).reset_index(drop=True)
    df.to_parquet(OUT, index=False)
    print(f"Saved {len(df):,} rows -> {OUT}")

if __name__ == "__main__":
    merge_and_save()
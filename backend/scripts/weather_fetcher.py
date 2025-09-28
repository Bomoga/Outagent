## Core imports
import os 
import requests, json, time, csv
from pathlib import Path
import pandas as pd
import numpy as np
from io import StringIO

## Set 'True' for debug info
INFO_MODE = False

## Base URL for NASA POWER API
BASE_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"

## API parameter values ()
PARAMETERS = "T2M,PRECTOT,WS2M,ALLSKY_SFC_SW_DWN,RH2M"
START = "20190101"
END = "20250920"
LAT = 26.5225
LON = 81.1637

def preprocess(data, interpolate = True, max_gap_hours = 3):
    base_date = pd.to_datetime({
        "year": data["YEAR"], "month": data["MO"], "day": data["DY"]
    }, errors="coerce")

    hours = pd.to_numeric(data["HR"], errors="coerce").fillna(0).astype(int)

    dt = base_date + pd.to_timedelta(hours, unit="h")
    data.insert(0, "datetime", dt)

    data.drop(columns=["YEAR", "MO", "DY", "HR"], inplace=True)

    value_cols = [c for c in data.columns if c != "datetime"]

    for col in value_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    if "RH2M" in data.columns:
        data["RH2M"] = data["RH2M"].clip(lower=0, upper=100)

    if "ALLSKY_SFC_SW_DWN" in data.columns:
        data["ALLSKY_SFC_SW_DWN"] = data["ALLSKY_SFC_SW_DWN"].clip(lower=0)

    for wcol in ("WS2M", "WS50M"):
        if wcol in data.columns:
            data[wcol] = data[wcol].clip(lower=0)

    if "PRECTOTCORR" in data.columns:
        data["PRECTOTCORR"] = data["PRECTOTCORR"].clip(lower=0)

    data.set_index("datetime", inplace=True)
    data.sort_index(inplace=True)

    if not data.empty:
        full_index = pd.date_range(data.index.min(), data.index.max(), freq="H")
        data = data.reindex(full_index)

    if interpolate:
        data = data.interpolate(method="time", limit=max_gap_hours, limit_direction="both")

    return data

def fetch(start, end, parameters, latitude, longitude, session = None):
    s = session or requests.Session()

    params = {
        "parameters": parameters,
        "latitude": latitude,
        "longitude": longitude,
        "community": "AG",
        "start": start,                    
        "end": end,                      
        "format": "CSV"
    }

    try:
        response = s.get(BASE_URL, params = params, timeout = 30)
        response.raise_for_status()

        return pd.read_csv(StringIO(response.text), skiprows = 13)

    except Exception as e:
        print(e)
        return None

#########################################################################################################

BASE_DIR = Path(__file__).resolve().parent.parent.parent

output_path = Path(os.path.join(f"{BASE_DIR}", "backend", "data", "processed", "nasa", f"{PARAMETERS}_{START}_{END}.csv"))

if output_path.exists():
    data = pd.read_csv(output_path)
    print(f"Loaded cache from {output_path} into dataframe.")

else:
    data = fetch(START, END, PARAMETERS, LAT, LON)
    preprocess(data)

    if INFO_MODE:
        print("Test rows:", len(data))
        print(data.head())
        print(data.shape)
        print(data.columns)

    ## Load raw data into raw folder
    data.to_csv(output_path, index = False)
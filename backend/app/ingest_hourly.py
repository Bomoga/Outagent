import requests
import pandas as pd
import numpy as np
import math
import logging
import sys
from pathlib import Path
from typing import Optional
from backend.settings import get_settings

# -----------------------
# CONFIG
# -----------------------
settings = get_settings()
API_BASE = settings.api_base
EIA_API_KEY = settings.eia_api_key
LAT = settings.wx_lat
LON = settings.wx_lon

BASE_DIR     = Path(__file__).resolve().parents[2]
RAW_EIA      = BASE_DIR / "backend" / "data" / "processed" / "eia"  / "FPL_DEMAND_2019-01-01T00_2025-09-20T00.csv"
RAW_NASA     = BASE_DIR / "backend" / "data" / "processed" / "nasa" / "T2M,PRECTOT,WS2M,ALLSKY_SFC_SW_DWN,RH2M_20190101_20250920.csv"
MASTER_DIR   = BASE_DIR / "backend" / "data" / "processed" / "latest"
EIA_CSV      = MASTER_DIR / "eia_hourly_master.csv"
WX_CSV       = MASTER_DIR / "weather_hourly_master.csv"

# Raw source files that must remain unchanged.
READ_ONLY_PATHS = {RAW_EIA.resolve(), RAW_NASA.resolve()}

EIA_COLS  = ["timestamp", "load_mw"]
WX_COLS   = ["timestamp", "temp_c", "rh", "wind_mps", "precip_mm", "ghi_kwhm2"]


def migrate_raw_eia_to_master(raw_path: Path, master_path: Path) -> None:
    if master_path.exists():
        return
    if not raw_path.exists():
        _ensure_csv(master_path, EIA_COLS)
        return
    df = pd.read_csv(raw_path)
    if "period" not in df.columns or "value" not in df.columns:
        _ensure_csv(master_path, EIA_COLS)
        return

    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(df["period"], errors="coerce", utc=True).dt.tz_convert(None),
            "load_mw": pd.to_numeric(df["value"], errors="coerce"),
        }
    )
    frame = frame.dropna(subset=["timestamp", "load_mw"])
    frame = frame.sort_values("timestamp")
    frame = frame.drop_duplicates(subset=["timestamp"], keep="last")
    master_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(master_path, index=False)


def migrate_raw_nasa_to_master(raw_path: Path, master_path: Path) -> None:
    if master_path.exists():
        return  # already migrated
    if not raw_path.exists():
        _ensure_csv(master_path, WX_COLS)
        return
    df = pd.read_csv(raw_path)

    if all(c in df.columns for c in ["YEAR", "MO", "DY", "HR"]):
        base = pd.to_datetime(
            dict(year=df["YEAR"], month=df["MO"], day=df["DY"]), errors="coerce", utc=True
        )
        hr = pd.to_numeric(df["HR"], errors="coerce").fillna(0).astype(int)
        ts = (base + pd.to_timedelta(hr, unit="h")).dt.tz_convert(None)
    elif "period" in df.columns:
        ts = pd.to_datetime(df["period"], errors="coerce", utc=True).dt.tz_convert(None)
    else:
        _ensure_csv(master_path, WX_COLS)
        return

    rename = {
        "T2M": "temp_c",
        "RH2M": "rh",
        "WS2M": "wind_mps",
        "ALLSKY_SFC_SW_DWN": "ghi_kwhm2",
        "PRECTOTCORR": "precip_mm",
        "PRECTOT": "precip_mm",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    for col in ["temp_c", "rh", "wind_mps", "precip_mm", "ghi_kwhm2"]:
        if col not in df.columns:
            df[col] = np.nan

    out = pd.DataFrame(
        {
            "timestamp": ts,
            "temp_c": pd.to_numeric(df["temp_c"], errors="coerce").replace([-999, -999.0], np.nan),
            "rh": pd.to_numeric(df["rh"], errors="coerce").replace([-999, -999.0], np.nan),
            "wind_mps": pd.to_numeric(df["wind_mps"], errors="coerce").replace([-999, -999.0], np.nan),
            "precip_mm": pd.to_numeric(df["precip_mm"], errors="coerce").replace([-999, -999.0], np.nan),
            "ghi_kwhm2": pd.to_numeric(df["ghi_kwhm2"], errors="coerce").replace([-999, -999.0], np.nan),
        }
    )
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp")
    out = out.drop_duplicates(subset=["timestamp"], keep="last")
    master_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(master_path, index=False)


def _ensure_csv(path: Path, columns: list[str]) -> None:
    resolved = path.resolve()
    if resolved in READ_ONLY_PATHS:
        raise ValueError(f"Refusing to create or modify read-only dataset: {resolved}")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=columns).to_csv(path, index=False)

def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True).dt.tz_convert(None)
    return df

def _append_row(path: Path, row: dict, order_cols: list[str]) -> pd.DataFrame:
    resolved = path.resolve()
    if resolved in READ_ONLY_PATHS:
        raise ValueError(f"Refusing to modify read-only dataset: {resolved}")

    if resolved == WX_CSV.resolve():
        df = _normalize_weather_csv(path)
    else:
        df = _read_csv(path)

    df_new = pd.DataFrame([row])
    df = pd.concat([df, df_new], ignore_index=True)

    for c in order_cols:
        if c not in df.columns:
            df[c] = np.nan

    if "timestamp" in df.columns:
        ts_col = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df["timestamp"] = ts_col.dt.tz_convert(None)
    df = df.dropna(subset=["timestamp"])

    for c in order_cols:
        if c == "timestamp" or c not in df.columns:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df[order_cols]
    df = df.dropna(subset=["timestamp"])
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    df.to_csv(resolved, index=False)
    return df

def _normalize_weather_csv(path: Path) -> pd.DataFrame:
    """
    Ensure master weather CSV has:
      ['timestamp','temp_c','rh','wind_mps','precip_mm','ghi_kwhm2']
    If timestamp is missing, build it from YEAR/MO/DY/HR or 'period'.
    Convert -999 sentinels to NaN.
    """
    if not path.exists():
        return pd.DataFrame(columns=["timestamp","temp_c","rh","wind_mps","precip_mm","ghi_kwhm2"])

    df = pd.read_csv(path)

    # Build timestamp if missing
    if "timestamp" not in df.columns:
        if all(c in df.columns for c in ["YEAR","MO","DY","HR"]):
            base = pd.to_datetime(dict(year=df["YEAR"], month=df["MO"], day=df["DY"]),
                                  errors="coerce", utc=True)
            hr = pd.to_numeric(df["HR"], errors="coerce").fillna(0).astype(int)
            df["timestamp"] = (base + pd.to_timedelta(hr, unit="h")).dt.tz_convert(None)
        elif "period" in df.columns:
            df["timestamp"] = pd.to_datetime(df["period"], errors="coerce", utc=True).dt.tz_convert(None)
        else:
            return pd.DataFrame(columns=["timestamp","temp_c","rh","wind_mps","precip_mm","ghi_kwhm2"])

    # Standardize column names
    rename = {
        "T2M": "temp_c",
        "RH2M": "rh",
        "WS2M": "wind_mps",
        "ALLSKY_SFC_SW_DWN": "ghi_kwhm2",
        "PRECTOTCORR": "precip_mm",
        "PRECTOT": "precip_mm",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # Ensure expected cols exist
    for col in ["temp_c","rh","wind_mps","precip_mm","ghi_kwhm2"]:
        if col not in df.columns:
            df[col] = np.nan

    # Clean types and sentinels
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True).dt.tz_convert(None)
    for col in ["temp_c","rh","wind_mps","precip_mm","ghi_kwhm2"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").replace([-999, -999.0], np.nan)

    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df[["timestamp","temp_c","rh","wind_mps","precip_mm","ghi_kwhm2"]]

# -----------------------
# Utilities
# -----------------------
def _is_finite(x) -> bool:
    return isinstance(x, (int, float, np.floating)) and math.isfinite(float(x))

def _all_weather_valid(rec: dict) -> bool:
    keys = ("temp_c","rh","wind_mps","precip_mm","ghi_kwhm2")
    return all(_is_finite(rec.get(k, np.nan)) for k in keys)

def _clip_weather(rec: dict) -> dict:
    out = dict(rec)
    if "rh" in out and _is_finite(out["rh"]):
        out["rh"] = float(np.clip(out["rh"], 0.0, 100.0))
    for k in ("wind_mps","precip_mm","ghi_kwhm2"):
        if k in out and _is_finite(out[k]):
            out[k] = float(max(out[k], 0.0))
    if "temp_c" in out and _is_finite(out["temp_c"]):
        out["temp_c"] = float(out["temp_c"])  # allow < 0
    return out

def _hourly_interp(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("_hourly_interp requires a DatetimeIndex")

    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    full = pd.date_range(df.index.min(), df.index.max(), freq="h")
    df = df.reindex(full)
    df = df.interpolate(method="time", limit=3, limit_direction="both").ffill().bfill()
    return df

# -----------------------
# Demand (EIA)
# -----------------------
def fetch_eia_latest() -> tuple[pd.Timestamp, float]:
    url = (
        "https://api.eia.gov/v2/electricity/rto/region-data/data/"
        f"?api_key={EIA_API_KEY}"
        "&frequency=hourly"
        "&facets[respondent][]=FPL"
        "&facets[type][]=D"
        "&data[0]=value"
        "&sort[0][column]=period&sort[0][direction]=desc"
        "&offset=0&length=1"
    )
    r = requests.get(url, timeout=25)
    r.raise_for_status()
    rows = r.json()["response"]["data"]
    if not rows:
        raise RuntimeError("EIA returned no rows")
    ts = pd.to_datetime(rows[0]["period"], utc=True).tz_convert(None)
    return ts, float(rows[0]["value"])

# -----------------------
# Weather (Open-Meteo primary, NASA fallback)
# -----------------------
def fetch_openmeteo_window(ts: pd.Timestamp, lat: float, lon: float) -> pd.DataFrame | None:
    start = (ts - pd.Timedelta(hours=47)).strftime("%Y-%m-%dT%H:00")
    end   = ts.strftime("%Y-%m-%dT%H:00")
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation,shortwave_radiation"
        "&windspeed_unit=ms&precipitation_unit=mm&timezone=UTC"
        f"&start_hour={start}&end_hour={end}"
    )
    r = requests.get(url, timeout=25)
    if not r.ok:
        return None
    h = r.json().get("hourly")
    if not h:
        return None
    idx = pd.to_datetime(h["time"], utc=True).tz_convert(None)
    df = pd.DataFrame({
        "temp_c": h["temperature_2m"],
        "rh": h["relative_humidity_2m"],
        "wind_mps": h["wind_speed_10m"],
        "precip_mm": h["precipitation"],
        "ghi_kwhm2": (np.array(h["shortwave_radiation"], dtype=float) / 1000.0),
    }, index=idx).astype(float)
    return df

def fetch_nasa_window(ts: pd.Timestamp, lat: float, lon: float) -> pd.DataFrame | None:
    day0 = ts.strftime("%Y%m%d")
    day1 = (ts - pd.Timedelta(days=1)).strftime("%Y%m%d")
    url = (
        "https://power.larc.nasa.gov/api/temporal/hourly/point"
        "?parameters=T2M,PRECTOTCORR,WS2M,ALLSKY_SFC_SW_DWN,RH2M"
        f"&start={day1}&end={day0}"
        f"&latitude={lat}&longitude={lon}"
        "&community=AG&format=JSON"
    )
    r = requests.get(url, timeout=30)
    if not r.ok:
        return None
    p = r.json().get("properties", {}).get("parameter", {})
    if not p:
        return None
    rows = []
    for d in (day1, day0):
        base = pd.to_datetime(d, utc=True).tz_convert(None)
        for hr in range(24):
            t = base + pd.Timedelta(hours=hr)
            rows.append({
                "timestamp": t,
                "temp_c": float(p.get("T2M", {}).get(str(hr), np.nan)),
                "rh": float(p.get("RH2M", {}).get(str(hr), np.nan)),
                "wind_mps": float(p.get("WS2M", {}).get(str(hr), np.nan)),
                "precip_mm": float(p.get("PRECTOTCORR", {}).get(str(hr), np.nan)),
                "ghi_kwhm2": float(p.get("ALLSKY_SFC_SW_DWN", {}).get(str(hr), np.nan)),
            })
    df = pd.DataFrame(rows).set_index("timestamp").sort_index()
    df = df.replace([-999, -999.0], np.nan)
    return df

# -----------------------
# Weather at target hour (using master CSV + APIs)
# -----------------------
def interpolate_weather_at(ts: pd.Timestamp) -> dict:
    # Load/normalize last ~72h from master CSV
    wx_hist = _normalize_weather_csv(WX_CSV)
    if not wx_hist.empty:
        wx_hist = wx_hist[wx_hist["timestamp"] >= (ts - pd.Timedelta(hours=72))]
        wx_hist = wx_hist.set_index("timestamp")[["temp_c","rh","wind_mps","precip_mm","ghi_kwhm2"]].astype(float)
    else:
        wx_hist = None

    # Pull fresh window (Open-Meteo → NASA fallback)
    df_api = fetch_openmeteo_window(ts, LAT, LON)
    if df_api is None:
        df_api = fetch_nasa_window(ts, LAT, LON)

    parts = [x for x in [wx_hist, df_api] if x is not None and not x.empty]
    if not parts:
        return {}

    df = pd.concat(parts).sort_index()
    df = df.replace([-999, -999.0], np.nan)
    df = _hourly_interp(df)

    # pick the row at or before ts
    if ts < df.index.min():
        pick = df.index.min()
    elif ts > df.index.max():
        pick = df.index.max()
    else:
        pick = df.index[df.index.get_indexer([ts], method="pad")][0]

    rec = df.loc[pick].to_dict()
    rec = _clip_weather(rec)

    # ensure finite
    for k, v in rec.items():
        if not _is_finite(v):
            return {}
    return rec

# -----------------------
# Main
# -----------------------
LOGGER = logging.getLogger(__name__)

_DIRECT_INGEST_CACHE: Optional[tuple] = None
_DIRECT_INGEST_DISABLED = False


def _try_ingest_via_app(payload: dict) -> Optional[bool]:
    global _DIRECT_INGEST_CACHE, _DIRECT_INGEST_DISABLED
    if _DIRECT_INGEST_DISABLED:
        return None
    if 'backend.app.main' not in sys.modules:
        _DIRECT_INGEST_DISABLED = True
        return None
    if _DIRECT_INGEST_CACHE is None:
        try:
            from backend.app.main import Observation, ingest_hour
        except Exception:
            _DIRECT_INGEST_DISABLED = True
            LOGGER.debug('backend.app.main unavailable for direct ingest; falling back to HTTP.', exc_info=True)
            return None
        _DIRECT_INGEST_CACHE = (Observation, ingest_hour)
    Observation, ingest_hour = _DIRECT_INGEST_CACHE
    try:
        if hasattr(Observation, 'model_validate'):
            obs = Observation.model_validate(payload)  # type: ignore[attr-defined)
        elif hasattr(Observation, 'parse_obj'):
            obs = Observation.parse_obj(payload)  # type: ignore[attr-defined)
        else:
            obs = Observation(**payload)
    except Exception:
        LOGGER.exception('Observation payload failed validation for in-process ingest.')
        return False
    ingest_hour(obs)
    return True


def ingest_once() -> bool:
    """Run the hourly ingest pipeline once. Returns True when data was appended."""

    migrate_raw_eia_to_master(RAW_EIA, EIA_CSV)
    migrate_raw_nasa_to_master(RAW_NASA, WX_CSV)

    _ensure_csv(EIA_CSV, EIA_COLS)
    _ensure_csv(WX_CSV, WX_COLS)

    ts, load_mw = fetch_eia_latest()

    wx = interpolate_weather_at(ts)
    if not wx or not _all_weather_valid(wx):
        LOGGER.warning("Weather unavailable or invalid for %s; skipping ingest.", ts)
        return False

    eia_row = {"timestamp": ts.isoformat(), "load_mw": float(load_mw)}
    wx_row = {
        "timestamp": ts.isoformat(),
        "temp_c": float(wx["temp_c"]),
        "rh": float(wx["rh"]),
        "wind_mps": float(wx["wind_mps"]),
        "precip_mm": float(wx["precip_mm"]),
        "ghi_kwhm2": float(wx["ghi_kwhm2"]),
    }

    _append_row(EIA_CSV, eia_row, EIA_COLS)
    _append_row(WX_CSV, wx_row, WX_COLS)

    payload = {
        "timestamp": ts.isoformat(),
        "load_mw": float(load_mw),
        "temp_c": wx_row["temp_c"],
        "rh": wx_row["rh"],
        "wind_mps": wx_row["wind_mps"],
        "precip_mm": wx_row["precip_mm"],
        "ghi_kwhm2": wx_row["ghi_kwhm2"],
    }

    response = requests.post(f"{API_BASE}/ingest/hour", json=payload, timeout=20)
    response.raise_for_status()
    LOGGER.info("Ingested hourly observation for %s", ts.isoformat())
    return True


def main() -> None:
    success = ingest_once()
    if success:
        print("Ingest completed successfully.")
    else:
        print("Ingest skipped due to invalid weather data.")


if __name__ == "__main__":
    main()



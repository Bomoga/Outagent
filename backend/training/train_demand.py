# backend/scripts/train_demand.py

import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import log_evaluation, early_stopping
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from joblib import dump

# === Paths ===
DATA = Path("backend/data/processed/merged/merged_hourly.parquet")
OUT = Path("backend/models/demand")
OUT.mkdir(parents=True, exist_ok=True)

# === Horizons ===
HORIZONS = list(range(1, 13))  # predict 1 to 12 hours ahead

# === Feature Engineering ===
def add_time_feats(df: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(df["timestamp"])
    df["hour"] = ts.dt.hour
    df["dow"] = ts.dt.dayofweek
    df["month"] = ts.dt.month
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    return df

def add_load_feats(df: pd.DataFrame) -> pd.DataFrame:
    y = "load_mw"
    df["load_t1"] = df[y].shift(1)
    df["load_t24"] = df[y].shift(24)
    df["load_t168"] = df[y].shift(168)
    df["roll_mean_24"] = df[y].rolling(24).mean()
    df["roll_max_24"] = df[y].rolling(24).max()
    df["roll_std_24"] = df[y].rolling(24).std()
    return df

FEATURES = [
    "load_t1","load_t24","load_t168",
    "roll_mean_24","roll_max_24","roll_std_24",
    "temp_c","rh","wind_mps","precip_mm","ghi_kwhm2",
    "hour","dow","month","is_weekend"
]

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

# === Main Training Loop ===
def main():
    print("Loading merged data...")
    df = pd.read_parquet(DATA).sort_values("timestamp").reset_index(drop=True)
    df = add_time_feats(add_load_feats(df))

    print(f"Training models for {len(df):,} hours of data")

    tscv = TimeSeriesSplit(n_splits=4)

    for h in HORIZONS:
        target = f"y_h{h}"
        df[target] = df["load_mw"].shift(-h)

        use = df.dropna(subset=FEATURES + [target])
        X, y = use[FEATURES], use[target]

        best, best_rmse = None, 1e9
        # cross-validation
        for tr, va in tscv.split(X):
            m = lgb.LGBMRegressor(
                n_estimators=1200,
                learning_rate=0.03,
                num_leaves=64,
                subsample=0.9,
                colsample_bytree=0.9
            )
            m.fit(
                X.iloc[tr], y.iloc[tr],
                eval_set=[(X.iloc[va], y.iloc[va])],
                eval_metric="l2",
                callbacks=[log_evaluation(period=0)]  # 0 = no logs
            )
            yhat = m.predict(X.iloc[va])
            score = rmse(y.iloc[va], yhat)
            if score < best_rmse:
                best, best_rmse = m, score

        # refit on full data
        best.fit(X, y, callbacks=[log_evaluation(period=0)])
        dump({"model": best, "features": FEATURES}, OUT / f"load_plus{h}h.joblib")

        print(f"[+{h:02}h] RMSE={best_rmse:.2f}  MAE={mean_absolute_error(y, best.predict(X)):.2f}")

    print(f"✅ Models saved to {OUT}")

if __name__ == "__main__":
    main()
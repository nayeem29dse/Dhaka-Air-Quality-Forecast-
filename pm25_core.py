"""Shared code for the Dhaka PM2.5 forecaster. Used by update_data.py and app.py."""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from lightgbm import LGBMRegressor

ROOT = Path(__file__).resolve().parent
HISTORY_PATH = ROOT / "data" / "pm25_history.csv"
LOG_PATH = ROOT / "data" / "forecast_log.csv"
MODEL_PATH = ROOT / "models" / "pm25_models.joblib"
RANGES_PATH = ROOT / "models" / "pm25_intervals_final.csv"

LAT, LON, TZ = 23.81, 90.41, "Asia/Dhaka"
AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

H = 24                 # forecast horizon (hours)
LAGS = [1, 2, 3, 6, 12, 24, 48]
MIN_HISTORY = 72       # complete hourly values needed before a forecast
RETRAIN_DAYS = 7

WINDOWS = ["Morning", "Office peak", "Afternoon peak", "Night"]
WINDOW_LABEL = {"Morning": "05:00-08:00", "Office peak": "08:00-11:00",
                "Afternoon peak": "16:00-20:00", "Night": "21:00-05:00"}

LGBM_PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=50,
                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                   random_state=42, verbose=-1)


# ---------- time and windows ----------
def now_dhaka():
    """Current Dhaka time as a naive timestamp (servers run in UTC, so never use the local clock)."""
    return pd.Timestamp.now(tz=TZ).tz_localize(None)


def window(h):
    if 5 <= h < 8:
        return "Morning"
    if 8 <= h < 11:
        return "Office peak"
    if 16 <= h < 20:
        return "Afternoon peak"
    if h >= 21 or h < 5:
        return "Night"
    return "Other"


# ---------- data ----------
def fetch_range(start_date, end_date, retries=4):
    """Hourly PM2.5 for Dhaka from start_date to end_date (inclusive), up to the last complete hour."""
    params = {"latitude": LAT, "longitude": LON, "hourly": "pm2_5",
              "start_date": str(start_date), "end_date": str(end_date), "timezone": TZ}
    for attempt in range(retries):
        r = requests.get(AQ_URL, params=params, timeout=60)
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(5 * 2 ** attempt)
            continue
        r.raise_for_status()
        h = r.json()["hourly"]
        s = pd.Series(h["pm2_5"], index=pd.to_datetime(h["time"]), dtype=float, name="pm2_5")
        s.index.name = "time"
        last_complete = now_dhaka().floor("h") - pd.Timedelta(hours=1)
        return s.loc[:last_complete].asfreq("h")
    raise RuntimeError("Open-Meteo still failing after retries")


def load_history(path=HISTORY_PATH):
    return pd.read_csv(path, index_col="time", parse_dates=True)["pm2_5"].asfreq("h")


def save_history(y, path=HISTORY_PATH):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    y.rename("pm2_5").rename_axis("time").to_csv(path)


def load_ranges(path=RANGES_PATH):
    return pd.read_csv(path).set_index("window")


# ---------- features and models ----------
def make_features(y):
    X = pd.DataFrame(index=y.index)
    X["y_t"] = y
    for k in LAGS:
        X[f"lag_{k}"] = y.shift(k)
    X["roll_24"] = y.rolling(24).mean()
    X["roll_72"] = y.rolling(72).mean()
    X["hour"] = y.index.hour
    X["month"] = y.index.month
    return X


def add_sh(X, y, h):
    """Same-hour-yesterday features for the horizon-h model (values at t+h-24 and t+h-48)."""
    return X.assign(same_hour_yday=y.shift(24 - h), same_hour_2d=y.shift(48 - h))


def train_models(y, first_origin=None):
    """Fit the 24 horizon models. first_origin=None uses all data (final models);
    otherwise it applies the backtest's cutoff, which lets the code be tested against the notebook."""
    X = make_features(y)
    models, cols = {}, None
    for h in range(1, H + 1):
        Xh = add_sh(X, y, h)
        train = pd.concat([Xh, y.shift(-h).rename("target")], axis=1)
        if first_origin is not None:
            train = train.loc[:first_origin - pd.Timedelta(hours=h + 1)]
        train = train.dropna()
        cols = list(Xh.columns)
        m = LGBMRegressor(**LGBM_PARAMS)
        m.fit(train[cols], train["target"])
        models[h] = m
    return {"models": models, "feature_cols": cols,
            "trained_through": str(y.dropna().index[-1]), "H": H}


# ---------- forecasting ----------
def _fmt_run(g):
    a, b = g["time"].iloc[0], g["time"].iloc[-1]
    if a == b:
        return f"{a:%a %H:%M}"
    end = f"{b:%H:%M}" if a.date() == b.date() else f"{b:%a %H:%M}"
    return f"{a:%a %H:%M}-{end}"


def forecast_now(y_recent, bundle, ranges):
    """y_recent: hourly PM2.5 ending at the last complete hour. Returns hourly and per-window forecasts."""
    y = y_recent.asfreq("h").iloc[-240:]
    if len(y) < MIN_HISTORY or y.iloc[-MIN_HISTORY:].isna().any():
        raise ValueError(f"need {MIN_HISTORY} complete hourly values ending at the last observed hour")
    t = y.index[-1]
    x_t = make_features(y).iloc[[-1]]

    fc = np.empty(H)
    for h in range(1, H + 1):
        xh = x_t.assign(same_hour_yday=y.iloc[-1 - (24 - h)],
                        same_hour_2d=y.iloc[-1 - (48 - h)])
        fc[h - 1] = bundle["models"][h].predict(xh[bundle["feature_cols"]])[0]
    fc = np.clip(fc, 0, None)

    times = t + pd.to_timedelta(np.arange(1, H + 1), unit="h")
    hourly = pd.DataFrame({"time": times, "window": [window(x.hour) for x in times], "forecast": fc})

    rows = []
    for w in WINDOWS:
        sub = hourly[hourly.window == w]
        mean = sub["forecast"].mean()
        brk = sub["time"].diff().gt(pd.Timedelta(hours=1)).cumsum()   # split non-contiguous pieces
        rows.append({"window": w, "forecast": mean,
                     "low": mean * ranges.loc[w, "lo"], "high": mean * ranges.loc[w, "hi"],
                     "hours": len(sub),
                     "covers": " + ".join(_fmt_run(g) for _, g in sub.groupby(brk))})
    return {"as_of": t, "hourly": hourly, "windows": pd.DataFrame(rows).set_index("window")}


# ---------- live track record ----------
def score_log(log, y):
    """Compare logged forecasts with what happened. log: as_of, window, forecast, low, high.
    Only forecasts whose window hours are all observed are scored."""
    y = y.asfreq("h")
    out = []
    for r in log.itertuples():
        t = pd.Timestamp(r.as_of)
        times = t + pd.to_timedelta(np.arange(1, H + 1), unit="h")
        sel = [x for x in times if window(x.hour) == r.window]
        if not sel or sel[-1] > y.index[-1]:
            continue
        actual = y.reindex(sel)
        naive = y.reindex([x - pd.Timedelta(hours=24) for x in sel])   # seasonal naive: same hours yesterday
        if actual.isna().any() or naive.isna().any():
            continue
        out.append({"as_of": t, "window": r.window, "forecast": r.forecast, "low": r.low,
                    "high": r.high, "actual": actual.mean(), "seasonal_naive": naive.mean()})
    s = pd.DataFrame(out)
    if len(s):
        s["abs_err"] = (s["actual"] - s["forecast"]).abs()
        s["abs_err_naive"] = (s["actual"] - s["seasonal_naive"]).abs()
        s["covered"] = (s["actual"] >= s["low"]) & (s["actual"] <= s["high"])
    return s

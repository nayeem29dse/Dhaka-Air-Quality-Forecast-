"""Refresh the stored history, retrain when the models are a week old, and log a forecast.

    python update_data.py              # normal run (what GitHub Actions does)
    python update_data.py --retrain    # force a retrain
"""
import argparse
import sys

import joblib
import pandas as pd

import pm25_core as core


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrain", action="store_true", help="retrain even if the models are recent")
    args = ap.parse_args(argv)

    # 1. Fetch from 3 days before the last stored hour (to pick up revisions) up to now.
    hist = core.load_history()
    last_stored = hist.dropna().index[-1]
    new = core.fetch_range((last_stored - pd.Timedelta(days=3)).date(), core.now_dhaka().date())
    if new.dropna().empty:
        sys.exit("API returned no data")

    both = new.dropna().index.intersection(hist.dropna().index)
    diff = (new.loc[both] - hist.loc[both]).abs()
    print(f"stored through {last_stored} | fetched {new.index[0]} -> {new.index[-1]} | "
          f"overlap {len(both)} h, mean revision {diff.mean():.3f}, max {diff.max():.2f}")

    # 2. New values win; stored values fill any hour the API returned as null.
    y = new.combine_first(hist).asfreq("h")
    core.save_history(y)

    n_missing = int(y.isna().sum())
    recent_gap = int(y.iloc[-core.MIN_HISTORY:].isna().sum())
    print(f"history: {len(y)} rows, {n_missing} missing, {recent_gap} missing in the last "
          f"{core.MIN_HISTORY} h")

    # 3. Retrain if the models are missing or a week old.
    bundle = joblib.load(core.MODEL_PATH) if core.MODEL_PATH.exists() else None
    age = None if bundle is None else y.dropna().index[-1] - pd.Timestamp(bundle["trained_through"])
    if args.retrain or bundle is None or age >= pd.Timedelta(days=core.RETRAIN_DAYS):
        print("training 24 models ...")
        bundle = core.train_models(y)
        core.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, core.MODEL_PATH, compress=3)
        print(f"saved models trained through {bundle['trained_through']} "
              f"({core.MODEL_PATH.stat().st_size / 1e6:.1f} MB)")

    # 4. Forecast and append to the log (refuse if the last 72 h are not complete).
    if recent_gap:
        sys.exit(f"{recent_gap} missing hours in the last {core.MIN_HISTORY}: "
                 f"history saved, forecast not logged")
    res = core.forecast_now(y, bundle, core.load_ranges())
    rows = res["windows"].reset_index()[["window", "forecast", "low", "high", "hours", "covers"]].round(2)
    rows.insert(0, "as_of", res["as_of"].strftime("%Y-%m-%d %H:%M:%S"))
    if core.LOG_PATH.exists():
        rows = pd.concat([pd.read_csv(core.LOG_PATH), rows])
    rows.drop_duplicates(["as_of", "window"], keep="first").to_csv(core.LOG_PATH, index=False)
    print(f"forecast as of {res['as_of']}")
    print(res["windows"].round(1).to_string())


if __name__ == "__main__":
    main()

import joblib
import pandas as pd
import streamlit as st

import pm25_core as core

st.set_page_config(page_title="Dhaka Air Quality(pm2.5) forecast", layout="wide")

# US EPA PM2.5 breakpoints (2024 revision), applied to each window's average.
BANDS = [(9.0, "Good", "🟢"), (35.4, "Moderate", "🟡"),
         (55.4, "Unhealthy for sensitive groups", "🟠"), (125.4, "Unhealthy", "🔴"),
         (225.4, "Very unhealthy", "🟣"), (float("inf"), "Hazardous", "🟤")]


def band(v):
    for limit, name, dot in BANDS:
        if v <= limit:
            return dot, name


@st.cache_resource
def load_models():
    return joblib.load(core.MODEL_PATH), core.load_ranges()


@st.cache_data(ttl=900)
def load_live():
    today = core.now_dhaka()
    return core.fetch_range((today - pd.Timedelta(days=8)).date(), today.date())


@st.cache_data(ttl=3600)
def load_track_record():
    if not core.LOG_PATH.exists():
        return pd.DataFrame()
    return core.score_log(pd.read_csv(core.LOG_PATH), core.load_history())


bundle, ranges = load_models()
live = load_live()
try:
    res = core.forecast_now(live, bundle, ranges)
except ValueError as e:
    st.error(f"Not enough recent data to forecast right now ({e}). Try again in a few minutes.")
    st.stop()

st.title("Dhaka PM2.5 forecast")
age_h = (core.now_dhaka() - res["as_of"]).total_seconds() / 3600
st.caption(f"Based on data up to {res['as_of']:%a %d %b, %H:%M} Dhaka time. "
           f"Each card is the average over that window within the next 24 hours.")
if age_h > 4:
    st.warning(f"The latest data is {age_h:.0f} hours old, so these forecasts may be out of date.")

cols = st.columns(4)
for col, w in zip(cols, core.WINDOWS):
    r = res["windows"].loc[w]
    dot, name = band(r["forecast"])
    low_name, high_name = band(r["low"])[1], band(r["high"])[1]
    with col, st.container(border=True):
        st.caption(f"{w} · {core.WINDOW_LABEL[w]}")
        st.metric("PM2.5 (µg/m³)", f"{r['forecast']:.0f}")
        st.write(f"{dot} {name}")
        st.write(f"Likely range: **{r['low']:.0f} to {r['high']:.0f}**")
        if low_name != high_name:
            st.caption(f"Range spans {low_name.lower()} to {high_name.lower()}")
        st.caption(f"Hours: {r['covers']}")

st.subheader("Next 24 hours")
obs = live.iloc[-48:].rename("Observed")
fc = res["hourly"].set_index("time")["forecast"].rename("Forecast")
st.line_chart(pd.concat([obs, fc], axis=1), y_label="PM2.5 (µg/m³)")

with st.expander("Live track record"):
    scored = load_track_record()
    if scored.empty:
        st.write("No forecasts have been scored yet. Results appear once a logged forecast's "
                 "window hours have all been observed.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Forecasts scored", len(scored))
        c2.metric("MAE, this model", f"{scored['abs_err'].mean():.1f}")
        c3.metric("MAE, same as yesterday", f"{scored['abs_err_naive'].mean():.1f}")
        c4.metric("Actual inside range", f"{scored['covered'].mean():.0%}")
        by_window = scored.groupby("window")[["abs_err", "abs_err_naive", "covered"]].mean().round(2)
        by_window.columns = ["MAE model", "MAE yesterday", "In range"]
        st.dataframe(by_window.reindex(core.WINDOWS))
        st.caption("Every logged forecast is scored once its window has been observed, "
                   "with no selection. The range targets 80%.")

with st.expander("How to read this, and what it can't do"):
    st.markdown(
        "- **Data is modeled, not measured.** PM2.5 comes from the CAMS atmospheric model via "
        "Open-Meteo for the Dhaka grid cell, not from a ground sensor, and the same value applies "
        "across the city.\n"
        "- **Ranges are wide on purpose.** They are the 10th to 90th percentile of past forecast "
        "errors, scaled to the forecast. In backtests they held about 80% of the time overall "
        "but only about 64% in the monsoon-to-winter transition weeks of September.\n"
        "- **Accuracy fades with lead time.** Windows within about 8 hours are noticeably better "
        "than 'same as yesterday'. Beyond about 12 hours the model is barely better.\n"
        "- **Winter mornings run low.** In held-out December weeks the model under-forecast the "
        "morning windows by about 6 µg/m³ on average.\n"
        "- Colors use the US EPA PM2.5 categories on window averages; they are a guide, not an "
        "official index for Dhaka.")

st.caption("Air quality data by [Open-Meteo.com](https://open-meteo.com/) (CAMS model data, CC BY 4.0). "
           "Forecast model: LightGBM, one model per forecast hour.")

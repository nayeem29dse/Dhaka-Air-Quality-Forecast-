# Dhaka PM2.5 Forecast

**Live app: [aq-dhaka.streamlit.app](https://aq-dhaka.streamlit.app/)**

A time series project that forecasts fine particulate matter (PM2.5) in Dhaka for the next 24 hours and reports it for four parts of the day: **Morning (05-08)**, **Office peak (08-11)**, **Afternoon peak (16-20)** and **Night (21-05)**. Each forecast comes with a likely range, a US EPA air quality category, and a live track record that scores past forecasts against what actually happened.

## What the app shows

- Four cards, one per time window, with the forecast average, the 80% range, the AQI category, and the exact hours the window covers.
- A 24-hour curve: the last 48 hours observed, followed by the forecast.
- A live track record: every logged forecast is scored against the observed values and compared with "same as yesterday". No forecasts are filtered out.

## Data

- **Source:** [Open-Meteo Air Quality API](https://open-meteo.com/en/docs/air-quality-api), hourly `pm2_5` for Dhaka (23.81, 90.41), from October 2022 onward (about 35,000 hourly rows, four winters).
- **Important:** these values come from the CAMS atmospheric model on a coarse grid. They are not ground sensor readings, and the same value applies across the whole city. A station-level map would need OpenAQ station data, but the Dhaka stations I checked do not have enough history to train on.
- **Free tier limits:** 600 calls per minute and 10,000 per day, far above what this project uses. Non-commercial use only.

## Method

1. **Exploration.** Strong daily cycle (autocorrelation peaks at lags 24, 48, 72), right-skewed values, a clear yearly cycle with January the worst month and July to August the cleanest, and no separate weekly pattern.
2. **Baselines.** Naive (repeat the last value) and seasonal naive (same hour yesterday).
3. **Model.** One LightGBM regressor for each forecast hour (24 models). Features: lags 1, 2, 3, 6, 12, 24, 48, rolling means over 24 and 72 hours, hour, month, and the values at the same clock hour yesterday and the day before.
4. **Validation.** Rolling backtests with each model trained only on data that ends before the test week (including a check that the training targets cannot overlap the test period). Weeks were chosen from every season, with forecasts issued at 00:00, 06:00, 12:00 and 18:00.
5. **Ranges.** The 10th and 90th percentile of `actual / forecast` per window, from 22 backtest weeks. Ratios are used instead of fixed offsets so the range widens when pollution is high.
6. **Windows.** For each forecast, the hourly predictions inside a window are averaged. The Night window wraps midnight, and a window can combine hours from two different days, which the app shows under "Hours".

## Results

All numbers are mean absolute error (MAE) in µg/m³ of the window average. Lower is better.

**Six held-out weeks** (December 2025 and October 2026), which were not used to tune the model or the ranges, except that I picked the final model after comparing variants on them:

| Window | Same as yesterday | LightGBM |
|---|---:|---:|
| Morning | 21.17 | 16.26 |
| Office peak | 19.70 | 16.32 |
| Afternoon peak | 10.58 | 10.58 |
| Night | 20.07 | 17.52 |
| **All** | **17.88** | **15.17 (-15%)** |

- On 16 earlier weeks spanning all seasons, the hourly forecasts beat seasonal naive in every week, by about 30%. That figure was optimistic, because the model variants were compared on those same weeks. The held-out gain of about 15% is the more honest estimate.
- **The gain shrinks with lead time.** When a window is within about 8 hours, the model beats "same as yesterday" by 20% to 45%. Beyond about 12 hours it is barely better.
- **The Afternoon peak window is a tie** with the baseline on held-out weeks.

## Limitations

- **The data is modeled, not measured**, so the app cannot show differences between neighborhoods.
- **Ranges are wide.** A forecast of 100 might show 65 to 140. That reflects real uncertainty 6 to 23 hours ahead.
- **Coverage varies by season.** The 80% range held about 80% of the time overall, but only about 64% in the September weeks when monsoon air gives way to winter air, and it is wider than needed in January.
- **The model runs low in winter mornings.** On held-out December weeks, the Morning and Office peak windows were under-forecast by about 6 µg/m³ on average.
- **Four winters is a short history,** and winters differ a lot (the 2023-24 winter was far milder than 2024-25).
- **AQI colors use the US EPA PM2.5 breakpoints** applied to window averages. They are a guide, not an official index for Bangladesh.

## Project structure

```
├── app.py                      # Streamlit app
├── pm25_core.py                # features, training, forecasting, scoring
├── update_data.py              # data refresh, weekly retrain, forecast log
├── requirements.txt
├── .github/workflows/update.yml
├── data/
│   ├── pm25_history.csv        # hourly PM2.5 history
│   └── forecast_log.csv        # every forecast issued, for the track record
└── models/
    ├── pm25_models.joblib      # the 24 LightGBM models
    └── pm25_intervals_final.csv  # range multipliers per window
```

## Automation

A GitHub Actions workflow runs at 00:10, 06:10, 12:10 and 18:10 Dhaka time. Each run:

1. Fetches the last 3 days from Open-Meteo, overwriting stored values with any revisions and filling new hours.
2. Retrains the 24 models if they are 7 days old (a manual run can force this).
3. Logs the forecast for all four windows, unless any of the last 72 hours is missing, in which case the run fails and nothing is logged.
4. Commits the updated data, models and log, which redeploys the app.

The app itself re-fetches the latest 8 days every 15 minutes, so the forecast always starts from the most recent hour, even between workflow runs.

## Run it locally

```bash
pip install -r requirements.txt
python update_data.py        # first run trains the models
streamlit run app.py
```

`update_data.py` expects `data/pm25_history.csv` (columns `time`, `pm2_5`, hourly) and `models/pm25_intervals_final.csv` (columns `window`, `lo`, `hi`).

## Next steps

- Recalibrate the ranges on a rolling window of recent weeks once the live log has enough data.
- Add weather variables (wind, humidity, rain) as features, which should help with the pollution episodes the model currently misses.
- Add a station layer from OpenAQ if Dhaka stations build up enough history.

## Credits

Air quality data by [Open-Meteo.com](https://open-meteo.com/), based on CAMS (Copernicus Atmosphere Monitoring Service) model data, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Author
**Mehedee Hasan Nyeem**
**Data Science Student**

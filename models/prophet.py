import pandas as pd

# If this import fails, try: from fbprophet import Prophet
from prophet import Prophet


def prophet_snow(df, date_col="Date", hs_col="HS_after_gapfill"):
    """
    Fit a Prophet model on HS_after_gapfill and forecast from 2018-11-01 to 2019-05-31.
    Returns a dataframe indexed by Date with an extra column 'HS_prophet'.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()

    # Regular daily series with forward-fill (Prophet expects regular freq)
    y = df[hs_col].astype(float).asfreq("D").ffill()

    # Prophet expects columns: ds (datetime), y (target)
    tmp = y.to_frame(name="y")
    tmp["ds"] = tmp.index

    # Train: up to 2018-10-31 (inclusive)
    train = tmp[tmp["ds"] <= pd.Timestamp("2018-10-31")]

    # Define Prophet model
    m = Prophet(
        yearly_seasonality=True,   # snow is strongly yearly-seasonal
        weekly_seasonality=False,  # can be turned on if you want
        daily_seasonality=False,
    )
    # You *can* add a custom seasonality if you want to tune:
    # m.add_seasonality(name="winter_cycle", period=365.25, fourier_order=5)

    # Fit
    m.fit(train[["ds", "y"]])

    # Build future dataframe up to 2019-05-31
    last_train_date = train["ds"].max()
    end_test_date = pd.Timestamp("2019-05-31")
    periods = (end_test_date - last_train_date).days

    future = m.make_future_dataframe(freq="D", periods=periods)
    forecast = m.predict(future)

    # Extract only the test window
    mask_test = (forecast["ds"] >= pd.Timestamp("2018-11-01")) & (
        forecast["ds"] <= end_test_date
    )
    forecast_test = forecast.loc[mask_test, ["ds", "yhat"]]

    # Align with your original df slice
    out = df.loc["2018-11-01":"2019-05-31"].copy()
    pred_series = (
        forecast_test.set_index("ds")["yhat"]
        .reindex(out.index)  # ensure exact date alignment
    )

    out["HS_prophet"] = pred_series
    return out

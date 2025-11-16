import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


def arima(df, p, d, q):
    date_col = "Date"
    hs_col = "HS_after_gapfill"
    out_col = "HS_arima"

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()

    # Regular daily series + fill missing values
    y = df[hs_col].astype(float).asfreq("D").ffill()

    # Train/test split
    train = y.loc[: "2018-10-31"]
    test_index = pd.date_range("2018-11-01", "2019-05-31", freq="D")

    # Build and fit ARIMA model
    model = ARIMA(train, order=(p, d, q))
    fitted = model.fit()

    # Forecast over the test window
    fc = fitted.get_forecast(steps=len(test_index))
    preds = pd.Series(fc.predicted_mean.values, index=test_index)

    # Create aligned output
    out = df.loc["2018-11-01":"2019-05-31"].copy()
    out[out_col] = preds
    return out

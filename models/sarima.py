import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

def sarima_forecast(df):
    """
    Returns the 2018-2019 season with a Seasonal ARIMA forecast:
    HS_sarima = mean HS of the same (month, day) across all previous years.
    :param df: df to update
    :return:
    """
    date_col = "Date"
    hs_col = "HS_after_gapfill"
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()


    train = df[df.index < "2018-11-01"]
    test  = df[(df.index >= "2018-11-01") & (df.index <= "2019-05-31")]
    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 365)

    model = SARIMAX(
        train[hs_col],
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    fitted = model.fit(disp=False)

    # Forecast on the test range
    preds = fitted.predict(start=test.index[0], end=test.index[-1])

    # Assemble output
    out = test.copy()
    out["HS_sarima"] = preds

    return out

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

def sarima(df, heaviness):
    date_col = "Date"
    hs_col = "HS_after_gapfill"
    out_col = "HS_sarima"

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()

    y = df[hs_col].astype(float).asfreq("D").ffill()

    train = y.loc[: "2018-10-31"]
    test_index = pd.date_range("2018-11-01", "2019-05-31", freq="D")


    order = (1, 1, 1) #events, treznds, errors
    seasonal_order = (1, 1, 1, 212) #THIS IS 212 FOR 1 NOV TO 31 MAY
    if heaviness == 1:
        seasonal_order = (1, 1, 1, 30)
    elif heaviness == 2:
        seasonal_order = (1, 1, 1, 7)
    elif heaviness == 3:
        seasonal_order = (0, 0, 0, 0)

    model = SARIMAX(
        train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    fitted = model.fit(disp=False)

    fc = fitted.get_forecast(steps=len(test_index))
    preds = pd.Series(fc.predicted_mean.values, index=test_index)

    out = df.loc["2018-11-01":"2019-05-31"].copy()
    out[out_col] = preds
    return out

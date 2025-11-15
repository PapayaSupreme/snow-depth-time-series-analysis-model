import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

def sarima(df, is_light):
    date_col = "Date"
    hs_col = "HS_after_gapfill"
    out_col = "HS_sarima"

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()

    train = df[df.index < "2018-11-01"]
    test = df[(df.index >= "2018-11-01") & (df.index <= "2019-05-31")]


    order = (1, 1, 1) #events, treznds, errors
    seasonal_order = (1, 1, 1, 212) #THIS IS 212 FOR 1 NOV TO 31 MAY
    if is_light:
        seasonal_order = (0, 0, 0, 0)

    y_train = train[hs_col].astype(float).copy()
    y_train.index = pd.RangeIndex(len(y_train))

    model = SARIMAX(
        y_train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    fitted = model.fit(disp=False)

    n_test = len(test)
    forecast_res = fitted.get_forecast(steps=n_test)
    preds_values = forecast_res.predicted_mean.values
    preds = pd.Series(preds_values, index=test.index)

    out = test.copy()
    out[out_col] = preds
    return out

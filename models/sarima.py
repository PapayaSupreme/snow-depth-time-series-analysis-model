import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

def sarima(df, date_col="Date", hs_col="HS_after_gapfill"):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()

    df = df[(df.index.month >= 11) | (df.index.month <= 5)]

    train = df[df.index < "2018-11-01"]
    test = df[(df.index >= "2018-11-01") & (df.index <= "2019-05-31")]


    order = (1, 1, 1)
    seasonal_order = (1, 1, 1, 212) #THIS IS 212 FOR 1 NOV TO 31 MAY

    model = SARIMAX(
        train[hs_col],
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    fitted = model.fit(disp=False)

    # ---- THIS IS THE IMPORTANT PART ----
    start_idx = len(train)
    end_idx = len(train) + len(test) - 1

    preds = fitted.predict(start=start_idx, end=end_idx)

    # Align predictions with test dates
    preds = pd.Series(preds, index=test.index)

    out = test.copy()
    out["HS_sarima"] = preds
    return out


def accuracy(df, col):
    """
    Computes the average error of the naive model
    :param df: pandas df
    :param col: the column to compute the accuracy
    :return: (float) mean absolute error between real and expected value
    """
    return (df["HS_after_gapfill"] - df[col]).abs().mean()
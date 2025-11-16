import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from utils.accuracy import accuracy, normalization

def rolling_seasonal_cv_arima(df, p, d, q, min_train_seasons=10):
    """
    Rolling seasonal cross-validation for ARIMA(p,d,q).

    :param df: (pandas dataframe) contains at least "Date" and "HS_after_gapfill
    :param p: (int) AR iterations
    :param d: (int) I iterations
    :param q: (int) MA iterations
    :param min_train_seasons: (int) number of seasons trained without rolling cv

    :return results_df: (pandas dataframe) per-season MAE and season_year
    :return global_mae: mean of per-season normalized MAE
    """
    date_col = "Date"
    hs_col = "HS_after_gapfill"

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()

    y = df[hs_col].astype(float).asfreq("D").ffill().bfill()

    m = y.index.month
    season_mask = (m >= 11) | (m <= 5)
    y = y[season_mask]

    months = y.index.month

    season_year = pd.Series(y.index.year, index=y.index, name="season_year").astype(int)
    season_year[months <= 5] -= 1

    seasons = sorted(season_year.unique())

    maes = []
    season_list = []

    for i in range(min_train_seasons, len(seasons)):
        val_season = seasons[i]
        train_seasons = seasons[:i]

        train_mask = season_year.isin(train_seasons)
        y_train = y[train_mask]

        val_mask = season_year == val_season
        val_index = y.index[val_mask]

        if len(y_train) == 0 or len(val_index) == 0:
            continue

        preds = arima_predict(y_train, val_index, p, d, q)

        if preds.isna().all():
            continue

        val_df = pd.DataFrame({
            "HS_after_gapfill": y[val_index],
            "HS_arima": preds
        })

        mae = accuracy(val_df, "HS_arima")

        season_mean = val_df["HS_after_gapfill"].mean()
        norm_mae = normalization(mae, season_mean)

        maes.append(norm_mae)
        season_list.append(val_season)

    results_df = pd.DataFrame({
        "season_year": season_list,
        "mae": maes,
    })

    global_mae = results_df["mae"].mean() if not results_df.empty else float("nan")
    return results_df, global_mae

def arima_predict(train_series, val_index, p, d, q):
    """
    Fit ARIMA(p,d,q)
     on `train_series` and forecast over `val_index`.
    :param p: (int) AR iterations
    :param d: (int) I iterations
    :param q: (int) MA iterations

    :returns: Series of predictions indexed by val_index. Returns NaN series if fit fails.
    """
    try:
        y_train = pd.Series(
            pd.to_numeric(train_series, errors="coerce").values,
            index=pd.RangeIndex(len(train_series))
        )

        y_train = y_train.ffill().bfill()

        model = ARIMA(y_train, order=(p, d, q))
        fitted = model.fit()

        fc = fitted.get_forecast(steps=len(val_index))
        preds = pd.Series(fc.predicted_mean.values, index=val_index)
        return preds
    except Exception:
        return pd.Series([float("nan")] * len(val_index), index=val_index)

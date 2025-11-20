import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

def rolling_seasonal_arima(df, p, d, q, is_whole, min_train_seasons=10):
    """
    Rolling seasonal cross-validation for ARIMA(p,d,q).

    :param df: (pandas dataframe) contains at least "Date" and "HS_after_gapfill
    :param p: (int) AR iterations
    :param d: (int) I iteration
    :param q: (int) MA iterations
    :param min_train_seasons: (int) number of seasons trained without rolling cv

    :return results_df: (pandas dataframe) per-season MAE and season_year
    :return global_mae: mean of per-season normalized MAE
    :return global_season_mean: mean of season depth
    :return global_predicted_mean: mean of predicted depth
    :return global_pct_error: pct error regarding global vals
    """
    date_col = "Date"
    hs_col = "HS_after_gapfill"

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()

    y = df[hs_col].astype(float).asfreq("D").ffill()

    if not is_whole:
        m = y.index.month
        season_mask = (m >= 11) | (m <= 5)
        y = y[season_mask]

    months = y.index.month

    season_year = pd.Series(y.index.year, index=y.index, name="season_year").astype(int)

    if not is_whole:
        season_year[months <= 5] -= 1

    seasons = sorted(season_year.unique())

    maes = []
    nmaes = []
    season_list = []
    season_means = []
    predicted_means = []
    pct_errors = []

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

        y_true = pd.to_numeric(val_df["HS_after_gapfill"], errors="coerce")
        y_pred = pd.to_numeric(val_df["HS_arima"], errors="coerce")
        mae = (y_true - y_pred).abs().mean()
        season_mean = val_df["HS_after_gapfill"].mean()
        predicted_mean = val_df["HS_arima"].mean()
        pct_error = ((predicted_mean - season_mean) / season_mean) * 100.0
        nmae = mae / season_mean if season_mean != 0 else float("nan")


        maes.append(mae)
        nmaes.append(nmae)
        season_list.append(val_season)
        season_means.append(season_mean)
        predicted_means.append(predicted_mean)
        pct_errors.append(pct_error)


    results_df = pd.DataFrame({
        "season_year": season_list,
        "mae": maes,
        "nmae": nmaes,
        "season_mean": season_means,
        "predicted_mean": predicted_means,
        "pct_error": pct_errors,
    })

    global_mae = results_df["mae"].mean() if not results_df.empty else float("nan")
    global_season_mean = results_df["season_mean"].mean() if not results_df.empty else float("nan")
    global_predicted_mean = results_df["predicted_mean"].mean() if not results_df.empty else float("nan")
    global_pct_error = results_df["pct_error"].mean() if not results_df.empty else float("nan")
    global_nmae = results_df["nmae"].mean() if not results_df.empty else float("nan")

    return (results_df, global_mae, global_nmae, global_season_mean,
            global_predicted_mean, global_pct_error)

def arima_predict(train_series, val_index, p, d, q):
    """
    Fit ARIMA(p,d,q)
     on `train_series` and forecast over `val_index`.

    :param train_series: (Series) training series
    :param val_index: (Index) index for predictions
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

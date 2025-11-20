import pandas as pd
from prophet import Prophet

def rolling_seasonal_prophet(
    df: pd.DataFrame,
    min_train_seasons = 10,
    yearly_seasonality: bool = True,
    weekly_seasonality: bool = False,
    daily_seasonality: bool = False,
    **prophet_kwargs,
):
    """
    Rolling seasonal cross-validation using Prophet.

    Same seasonal split logic and return structure as rolling_seasonal_sarima.

    :param df: pandas DataFrame with at least "Date" and "HS_after_gapfill"
    :param min_train_seasons: (int) number of seasons trained without rolling CV
    :param yearly_seasonality: passed to Prophet(...)
    :param weekly_seasonality: passed to Prophet(...)
    :param daily_seasonality: passed to Prophet(...)
    :param prophet_kwargs: any extra kwargs forwarded to Prophet(...)

    :return results_df: (DataFrame) per-season metrics:
        ["season_year", "mae", "nmae", "season_mean", "predicted_mean", "pct_error"]
    :return global_mae:      mean of per-season MAE
    :return global_nmae:     mean of per-season NMAE
    :return global_season_mean: mean of true season means
    :return global_predicted_mean: mean of predicted season means
    :return global_pct_error: average pct error of season means
    """
    date_col = "Date"
    hs_col = "HS_after_gapfill"

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()

    y = df[hs_col].astype(float).asfreq("D").ffill()

    m = y.index.month
    season_mask = (m >= 11) | (m <= 5)
    y = y[season_mask].dropna()

    months = y.index.month
    season_year = pd.Series(
        y.index.year, index=y.index, name="season_year"
    ).astype(int)
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
        val_mask = (season_year == val_season)

        y_train = y[train_mask]
        val_index = y.index[val_mask]

        if len(y_train) == 0 or len(val_index) == 0:
            continue

        preds = prophet_predict(
            y_train,
            val_index,
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,
            **prophet_kwargs,
        )

        if preds.isna().all():
            continue

        val_df = pd.DataFrame(
            {
                "HS_after_gapfill": y[val_index],
                "HS_prophet": preds,
            }
        )

        y_true = pd.to_numeric(val_df["HS_after_gapfill"], errors="coerce")
        y_pred = pd.to_numeric(val_df["HS_prophet"], errors="coerce")
        mae = (y_true - y_pred).abs().mean()
        season_mean = val_df["HS_after_gapfill"].mean()
        predicted_mean = val_df["HS_prophet"].mean()
        pct_error = ((predicted_mean - season_mean) / season_mean) * 100.0
        nmae = mae / season_mean if season_mean != 0 else float("nan")

        maes.append(mae)
        nmaes.append(nmae)
        season_list.append(val_season)
        season_means.append(season_mean)
        predicted_means.append(predicted_mean)
        pct_errors.append(pct_error)

    results_df = pd.DataFrame(
        {
            "season_year": season_list,
            "mae": maes,
            "nmae": nmaes,
            "season_mean": season_means,
            "predicted_mean": predicted_means,
            "pct_error": pct_errors,
        }
    )

    global_mae = results_df["mae"].mean() if not results_df.empty else float("nan")
    global_season_mean = (
        results_df["season_mean"].mean() if not results_df.empty else float("nan")
    )
    global_predicted_mean = (
        results_df["predicted_mean"].mean() if not results_df.empty else float("nan")
    )
    global_pct_error = (
        results_df["pct_error"].mean() if not results_df.empty else float("nan")
    )
    global_nmae = (
        results_df["nmae"].mean() if not results_df.empty else float("nan")
    )

    return (
        results_df,
        global_mae,
        global_nmae,
        global_season_mean,
        global_predicted_mean,
        global_pct_error,
    )


def prophet_predict(
    train_series: pd.Series,
    val_index: pd.DatetimeIndex,
    yearly_seasonality: bool = True,
    weekly_seasonality: bool = False,
    daily_seasonality: bool = False,
    **prophet_kwargs,
) -> pd.Series:
    """
    Fit Prophet on `train_series` and forecast over `val_index`.

    :param train_series: (Series) training series, index = DatetimeIndex
    :param val_index: (DatetimeIndex) dates to predict on
    :param yearly_seasonality: passed to Prophet(...)
    :param weekly_seasonality: passed to Prophet(...)
    :param daily_seasonality: passed to Prophet(...)
    :param prophet_kwargs: extra kwargs forwarded to Prophet(...)

    :returns: Series of predictions indexed by val_index.
              Returns NaN series if fit fails.
    """
    try:
        # Prepare training data for Prophet
        train_df = pd.DataFrame({
            "ds": pd.to_datetime(train_series.index),
            "y": pd.to_numeric(train_series.values, errors="coerce"),
        })

        train_df["ds"] = pd.to_datetime(train_df["ds"])

        model = Prophet(
            changepoint_prior_scale=0.1,
            yearly_seasonality=12,
            seasonality_mode="multiplicative",
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,
            **prophet_kwargs,
        )

        model.fit(train_df)

        future = pd.DataFrame({"ds": pd.to_datetime(val_index)})
        forecast = model.predict(future)

        preds = pd.Series(forecast["yhat"].values, index=val_index)
        return preds

    except Exception:
        return pd.Series([float("nan")] * len(val_index), index=val_index)

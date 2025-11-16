import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from utils.accuracy import accuracy, normalization


def rolling_seasonal_cv_sarima(
    df,
    p, d, q,
    P, D, Q, s,
    simple_diff,
    min_train_seasons=10
):
    """
    Rolling seasonal cross-validation for SARIMA.

    :param df: pandas DataFrame, has at least 'Date' and 'HS_after_gapfill'
    :param p,d,q: non-seasonal ARIMA order
    :param P,D,Q,s: seasonal order (P,D,Q,s)
    :param min_train_seasons: number of initial seasons used only for training

    :return results_df: per-season (normalized) MAE and season_year
    :return global_mae: mean of per-season normalized MAE
    """
    date_col = "Date"
    hs_col = "HS_after_gapfill"

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()

    # Regular daily series + fill missing values
    y = df[hs_col].astype(float).asfreq("D").ffill().bfill()

    # Keep only Nov–May (snow season)
    m = y.index.month
    season_mask = (m >= 11) | (m <= 5)
    y = y[season_mask]

    # Compute season_year: Nov–Dec -> year, Jan–May -> year-1
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
        val_mask = (season_year == val_season)

        y_train = y[train_mask]
        val_index = y.index[val_mask]

        if len(y_train) == 0 or len(val_index) == 0:
            continue

        preds = sarima_predict(y_train, val_index, p, d, q, P, D, Q, s, simple_diff)

        if preds.isna().all():
            continue

        val_df = pd.DataFrame({
            "HS_after_gapfill": y[val_index],
            "HS_sarima": preds
        })

        # Raw MAE using your existing util
        mae = accuracy(val_df, "HS_sarima")

        # Normalized by seasonal mean depth
        season_mean = val_df["HS_after_gapfill"].mean()
        norm_mae = noemalization(mae, season_mean)

        maes.append(norm_mae)
        season_list.append(val_season)

    results_df = pd.DataFrame({
        "season_year": season_list,
        "normalized_mae": maes,
    })

    global_mae = results_df["normalized_mae"].mean() if not results_df.empty else float("nan")
    return results_df, global_mae


def sarima_predict(train_series, val_index, p, d, q, P, D, Q, s, simple_diff):
    """
    Fit SARIMA on `train_series` and forecast over `val_index`.
    Returns Series indexed by val_index. NaN series if fit fails.
    """
    try:
        y_train = pd.Series(
            pd.to_numeric(train_series, errors="coerce").values,
            index=pd.RangeIndex(len(train_series))
        )
        y_train = y_train.ffill().bfill()

        model = SARIMAX(
            y_train,
            order=(p, d, q),
            seasonal_order=(P, D, Q, s),
            enforce_stationarity=False,
            enforce_invertibility=False,
            simple_differencing=simple_diff
        )
        fitted = model.fit(disp=False)

        fc = fitted.get_forecast(steps=len(val_index))
        preds = pd.Series(fc.predicted_mean.values, index=val_index)
        return preds

    except Exception:
        return pd.Series([float("nan")] * len(val_index), index=val_index)

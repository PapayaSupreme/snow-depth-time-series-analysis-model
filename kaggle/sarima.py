# de-comment line below in kaggle
# !pip install statsmodels pandas --quiet

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

def main():
    filenames = ["col_de_porte_daily.txt", "les2alpes_daily.txt",
                 "serre_chevalier_daily.txt", "tignes_daily.txt"]

    dfs = {}
    for filename in filenames:
        dfs[filename] = pd.read_csv("/kaggle/input/stations/" + filename)

    print("NOTE: ARIMA PARAMETERS ARE (p = 1, d = 1, q = 1)")
    p, d, q = 1, 0, 0
    P, D, Q = 1, 1, 1
    S = 212
    simple_diff = False
    for k in dfs:
        results, mae, nmae, season, predicted, pct_error = rolling_seasonal_sarima(
            dfs[k],
            p, d, q,
            P, D, Q, S,
            simple_diff,
            min_train_seasons=10
        )

        print(f"=== {k} SARIMA({p}, {d}, {q})({P}, {D}, {Q}, {S}) Rolling Validation ===")
        print(results.tail(5))
        pct_error = ((predicted - season) / season) * 100.0
        print("Global: MAE:", mae.round(3), "NMAE:", nmae.round(3), "mean:", season.round(3), "predicted:",
                          predicted.round(3), "%:", pct_error.round(3))
        print()

def rolling_seasonal_sarima(df,
                               p, d, q,
                               P, D, Q, S,
                               simple_diff,
                               min_train_seasons):
    """
    Rolling seasonal cross-validation for SARIMA(p, d, q)(P, D, Q, S).

    :param df: (pandas dataframe) contains at least "Date" and "HS_after_gapfill
    :param p: (int) AR iterations
    :param d: (int) I iterations
    :param q: (int) MA iterations
    :param P: (int) AR iterations
    :param D: (int) I iterations
    :param Q: (int) MA iterations
    :param S: (int) seasonal grid size
    :param simple_diff: (boolean) Use simple differenciating
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

    months = y.index.month
    season_year = pd.Series(y.index.year, index=y.index, name="season_year").astype(int)

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

        preds = sarima_predict(y_train, val_index, p, d, q, P, D, Q, S, simple_diff)

        if preds.isna().all():
            continue

        val_df = pd.DataFrame({
            "HS_after_gapfill": y[val_index],
            "HS_sarima": preds
        })

        y_true = pd.to_numeric(val_df["HS_after_gapfill"], errors="coerce")
        y_pred = pd.to_numeric(val_df["HS_sarima"], errors="coerce")
        mae = (y_true - y_pred).abs().mean()
        season_mean = val_df["HS_after_gapfill"].mean()
        predicted_mean = val_df["HS_sarima"].mean()
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


def sarima_predict(train_series, val_index, p, d, q, P, D, Q, S, simple_diff):
    """
    Fit SARIMA(p, d, q)(P, D, Q, S)
     on `train_series` and forecast over `val_index`.

    :param p: (int) AR iterations
    :param d: (int) I iterations
    :param q: (int) MA iterations
    :param P: (int) AR iterations
    :param D: (int) I iterations
    :param Q: (int) MA iterations
    :param S: (int) seasonal grid size
    :param simple_diff: (boolean) Use simple differenciating

    :returns: Series of predictions indexed by val_index. Returns NaN series if fit fails.
    """
    try:
        y_train = pd.to_numeric(train_series, errors="coerce").ffill().dropna()

        if not isinstance(y_train.index, pd.DatetimeIndex):
            y_train.index = pd.to_datetime(train_series.index)
        y_train = y_train.asfreq('D').ffill().dropna()

        model = SARIMAX(
            y_train,
            order=(p, d, q),
            seasonal_order=(P, D, Q, S),
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


if __name__ == "__main__":
    main()

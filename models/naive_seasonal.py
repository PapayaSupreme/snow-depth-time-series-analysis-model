import pandas as pd


def rolling_naive_seasonal(df: pd.DataFrame, min_train_seasons: int):
    """
    Rolling seasonal cross-validation for a naive seasonal baseline.
    For each validation season, the forecast for a given (month, day) is the
    mean HS of the same (month, day) across all previous seasons.

    :param df: (pandas dataframe) contains at least "Date" and "HS_after_gapfill
    :param min_train_seasons: (int) number of seasons trained without rolling cv

    :return results_df: per-season normalized MAE and season_year
    :return global_mae: mean of per-season normalized MAE
    :return global_season_mean: mean of season depth
    :return global_predicted_mean: mean of predicted depth
    :return global_pct_error: pct error regarding global vals
    """
    date_col = "Date"
    hs_col = "HS_after_gapfill"

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    df["year"] = df[date_col].dt.year
    df["month"] = df[date_col].dt.month
    df["day"] = df[date_col].dt.day

    season_mask = (df["month"] >= 11) | (df["month"] <= 5)
    df = df[season_mask]

    df["season_year"] = df["year"].astype(int)
    df.loc[df["month"] <= 5, "season_year"] -= 1

    seasons = sorted(df["season_year"].unique())

    maes = []
    nmaes = []
    season_list = []
    season_means = []
    predicted_means = []
    pct_errors = []

    for i in range(min_train_seasons, len(seasons)):
        val_season = seasons[i]
        train_seasons = seasons[:i]

        prev = df[df["season_year"].isin(train_seasons)]
        curr = df[df["season_year"] == val_season].copy()

        if prev.empty or curr.empty:
            continue

        day_means = (
            prev.groupby(["month", "day"])[hs_col]
            .mean()
            .rename("HS_naive")
        )

        curr = curr.merge(day_means, on=["month", "day"], how="left")

        if curr["HS_naive"].isna().all():
            continue

        val_df = curr[[hs_col, "HS_naive"]]


        y_true = pd.to_numeric(val_df["HS_after_gapfill"], errors="coerce")
        y_pred = pd.to_numeric(val_df["HS_naive"], errors="coerce")
        mae = (y_true - y_pred).abs().mean()
        season_mean = val_df["HS_after_gapfill"].mean()
        predicted_mean = val_df["HS_naive"].mean()
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

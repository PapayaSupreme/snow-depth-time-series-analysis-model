import pandas as pd
from utils.accuracy import accuracy, normalization


def rolling_naive_seasonal(df: pd.DataFrame, min_train_seasons: int):
    """
    Rolling seasonal cross-validation for a naive seasonal baseline.
    For each validation season, the forecast for a given (month, day) is the
    mean HS of the same (month, day) across all previous seasons.

    :param df: (pandas dataframe) contains at least "Date" and "HS_after_gapfill
    :param min_train_seasons: (int) number of seasons trained without rolling cv

    :return results_df: per-season normalized MAE and season_year
    :return global_mae: mean of per-season normalized MAE
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
    season_list = []

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

        mae = accuracy(val_df, "HS_naive")

        mean_depth = curr[hs_col].mean()
        norm_mae = normalization(mae, mean_depth)

        maes.append(norm_mae)
        season_list.append(val_season)

    results_df = pd.DataFrame({
        "season_year": season_list,
        "normalized_mae": maes,
    })

    global_mae = results_df["normalized_mae"].mean() if not results_df.empty else float("nan")
    return results_df, global_mae

import pandas as pd

def naive_seasonal(df: pd.DataFrame):
    """
    Returns the 2018-2019 season with a naive forecast:
    HS_naive = mean HS of the same (month, day) across all previous years.
    :param: df: df to update
    :return: df with the HS_naive column added
    """
    date_col = "Date"
    hs_col = "HS_after_gapfill"
    out_col = "HS_naive"
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    df["year"]  = df[date_col].dt.year
    df["month"] = df[date_col].dt.month
    df["day"]   = df[date_col].dt.day

    is_target = (
        ((df["year"] == 2018) & (df["month"] >= 11)) |
        ((df["year"] == 2019) & (df["month"] <= 5))
    )
    curr = df[is_target].copy()

    prev = df[df["year"] < 2018].copy()

    curr = curr.sort_values(date_col).reset_index(drop=True)
    prev = prev.sort_values(date_col).reset_index(drop=True)

    curr[out_col] = float("nan")

    for i in range(len(curr)):
        m = curr.loc[i, "month"]
        d = curr.loc[i, "day"]

        same_day_prev = prev[(prev["month"] == m) & (prev["day"] == d)]

        if not same_day_prev.empty:
            curr.loc[i, out_col] = same_day_prev[hs_col].mean()
        else:
            curr.loc[i, out_col] = float("nan")

    return curr

def naive_accuracy(df_naive):
    """
    Computes the average error of the naive model
    :param df_naive: pandas df containing the naive model column
    :return: (float) mean absolute error between real and expected value
    """
    return (df_naive["HS_after_gapfill"] - df_naive["HS_naive"]).abs().mean()
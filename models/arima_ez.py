import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


def arima_train_test_split(df, p, d, q, test_ratio=0.2):
    """
    Train-test split (80/20) ARIMA(p,d,q) evaluation.

    :param df: (pandas dataframe) contains at least "Date" and "HS_after_gapfill
    :param p: (int) AR iterations
    :param d: (int) I iteration
    :param q: (int) MA iterations
    :param test_ratio: (float) train test ratio

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

    y = df[hs_col].astype(float).asfreq("D").ffill().bfill()

    m = y.index.month
    season_mask = (m >= 11) | (m <= 5)
    y = y[season_mask]

    split_idx = int(len(y) * (1 - test_ratio))

    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    if len(y_train) == 0 or len(y_test) == 0:
        raise ValueError("Train or test set is empty. Check the dataset or ratio.")

    try:
        model = ARIMA(y_train, order=(p, d, q))
        fitted = model.fit()
    except Exception as e:
        raise RuntimeError(f"ARIMA failed to fit: {e}")

    try:
        fc = fitted.get_forecast(steps=len(y_test))
        preds = pd.Series(fc.predicted_mean.values, index=y_test.index)
    except Exception as e:
        raise RuntimeError(f"Forecasting failed: {e}")

    y_true = y_test
    y_pred = preds

    mae = (y_true - y_pred).abs().mean()
    season_mean = y_true.mean()
    predicted_mean = y_pred.mean()
    pct_error = ((predicted_mean - season_mean) / season_mean) * 100.0
    nmae = mae / season_mean if season_mean != 0 else float("nan")

    results_df = pd.DataFrame({
        "date": y_test.index,
        "true": y_true.values,
        "pred": y_pred.values,
        "abs_error": (y_true - y_pred).abs().values
    })

    return results_df, mae, nmae, season_mean, predicted_mean, pct_error

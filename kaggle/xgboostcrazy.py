import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import warnings
import torch

warnings.filterwarnings('ignore')

STATION_FILENAMES = ["tignes_daily.txt", "les2alpes_daily.txt",
                     "serre_chevalier_daily.txt", "col_de_porte_daily.txt"]
TARGET_COL = "HS_after_gapfill"
TEST_YEARS = 3

def create_features(df):
    df = df.copy()

    doy = df.index.dayofyear
    is_leap = df.index.is_leap_year

    df['adjusted_doy'] = np.where(
        (is_leap) & (doy > 60),
        doy - 1,
        doy
    )

    df['day_sin'] = np.sin(2 * np.pi * df['adjusted_doy'] / 365.0)
    df['day_cos'] = np.cos(2 * np.pi * df['adjusted_doy'] / 365.0)

    df['month'] = df.index.month

    df['lag_365'] = df[TARGET_COL].shift(365)
    df['lag_1']   = df[TARGET_COL].shift(1)
    df['lag_7']   = df[TARGET_COL].shift(7)

    return df

def train_and_predict_recursive(df):
    df = df.dropna().copy()

    unique_years = df.index.year.unique()
    split_year = unique_years[-TEST_YEARS]

    train = df[df.index.year < split_year].copy()
    test  = df[df.index.year >= split_year].copy()

    features = ['adjusted_doy', 'month', 'day_sin', 'day_cos', 'lag_1', 'lag_7', 'lag_365']
    target = TARGET_COL

    print(f"Training XGBoost on data before {split_year}...")

    reg = xgb.XGBRegressor(
        n_estimators=3000,
        learning_rate=0.01,
        max_depth=10,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=1,
        reg_lambda=1.0,
        objective='reg:squarederror',
        tree_method='hist',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )


    val_split_idx = int(len(train) * 0.9)
    X_train = train[features].iloc[:val_split_idx]
    y_train = train[target].iloc[:val_split_idx]
    X_val   = train[features].iloc[val_split_idx:]
    y_val   = train[target].iloc[val_split_idx:]

    reg.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='mae',
        early_stopping_rounds=50,
        verbose=False
    )

    test_predictions = []
    current_test_df = test.copy()

    lag_1_idx = current_test_df.columns.get_loc('lag_1')
    lag_7_idx = current_test_df.columns.get_loc('lag_7')

    for i in range(len(current_test_df)):
        row = current_test_df.iloc[[i]][features]

        pred_val = reg.predict(row)[0]
        pred_val = max(0, pred_val)

        test_predictions.append(pred_val)

        if i + 1 < len(current_test_df):
            current_test_df.iloc[i + 1, lag_1_idx] = pred_val

        if i + 7 < len(current_test_df):
            current_test_df.iloc[i + 7, lag_7_idx] = pred_val

    test['prediction'] = test_predictions
    mae = mean_absolute_error(test[target], test['prediction'])
    nmae = mae / test[target].mean()
    print(f"\nFinal MAE : {mae:.2f} cm")
    print(f"Final NMAE: {nmae:.3f}")

    return test

def main():
    for STATION_FILENAME in STATION_FILENAMES:
        print(f"\n===== {STATION_FILENAME} =====")
        path = "/kaggle/input/stations/" + STATION_FILENAME
        try:
            df = pd.read_csv(path)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date').sort_index()

            df = df[[TARGET_COL]]
            df[TARGET_COL] = df[TARGET_COL].fillna(0)

        except Exception as e:
            print(f"Error loading file {STATION_FILENAME}: {e}")
            continue

        df_features = create_features(df)
        results = train_and_predict_recursive(df_features)

        plt.figure(figsize=(15, 6))
        plt.plot(results.index, results[TARGET_COL], label='Actual', color='black', alpha=0.5, linewidth=1)
        plt.plot(results.index, results['prediction'], label='XGBoost Forecast', color='#007acc', alpha=0.8, linewidth=1.5)
        plt.title(f'XGBoost Recursive Forecast for {STATION_FILENAME}')
        plt.ylabel('Snow Depth (cm)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

if __name__ == "__main__":
    main()

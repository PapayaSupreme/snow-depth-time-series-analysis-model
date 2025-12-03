import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, List
from torch.utils.data import TensorDataset, DataLoader
import sys

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Status: Running on {DEVICE}")


class SeasonalGRU(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, output_dim: int, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.gru(x)
        last_out = self.dropout(out[:, -1, :])
        return self.fc(last_out)


def _create_features(dates: pd.DatetimeIndex, is_whole: bool = False) -> np.ndarray:
    """Create cyclical time features efficiently."""
    doy = dates.dayofyear

    if is_whole:
        day_sin = np.sin(2 * np.pi * doy / 365.25)
        day_cos = np.cos(2 * np.pi * doy / 365.25)
    else:
        seasonal_day = np.where(
            doy >= 305,
            doy - 305,  # Nov-Dec: 0 to 60
            doy + 60  # Jan-May: 61 to ~212
        )
        day_sin = np.sin(2 * np.pi * seasonal_day / 213.0)
        day_cos = np.cos(2 * np.pi * seasonal_day / 213.0)

    dow = dates.dayofweek
    dow_sin = np.sin(2 * np.pi * dow / 7)
    dow_cos = np.cos(2 * np.pi * dow / 7)

    return np.column_stack([day_sin, day_cos, dow_sin, dow_cos])


def _make_sequences(series: np.ndarray, features: np.ndarray,
                    season_years: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    step = max(1, seq_len // 2)

    for i in range(0, len(series) - seq_len, step):
        start_season = season_years[i]
        end_season = season_years[i + seq_len]

        # Only create window if it resides wholly within one season
        if start_season == end_season:
            X.append(np.column_stack([series[i:i + seq_len], features[i:i + seq_len]]))
            y.append(series[i + seq_len])

    return np.array(X), np.array(y)


def gru_predict(
        train_series: pd.Series,
        train_season_years: pd.Series,
        val_index: pd.DatetimeIndex,
        is_whole: bool = False,
        seq_len: int = 90,
        hidden_dim: int = 64,
        num_layers: int = 2,
        epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int = 16,
        patience: int = 10
) -> pd.Series:
    y_train = pd.to_numeric(train_series, errors="coerce").ffill().bfill()
    values = y_train.values.astype(float)

    if len(values) <= seq_len + 20:
        return pd.Series([np.nan] * len(val_index), index=val_index)

    mean, std = values.mean(), values.std()
    std = std if std > 0 else 1.0
    values_norm = (values - mean) / std

    train_features = _create_features(y_train.index, is_whole)
    season_years_arr = train_season_years.values

    X, y = _make_sequences(values_norm, train_features, season_years_arr, seq_len)

    if len(X) < 10:
        return pd.Series([np.nan] * len(val_index), index=val_index)

    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train_seq, y_val_seq = y[:split_idx], y[split_idx:]

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
    y_train_t = torch.tensor(y_train_seq, dtype=torch.float32, device=DEVICE).unsqueeze(1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=DEVICE)
    y_val_t = torch.tensor(y_val_seq, dtype=torch.float32, device=DEVICE).unsqueeze(1)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=batch_size,
        shuffle=True
    )

    input_dim = X.shape[2]
    model = SeasonalGRU(input_dim, hidden_dim, num_layers, 1).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            if len(X_val_t) > 0:
                val_preds = model(X_val_t)
                val_loss = criterion(val_preds, y_val_t).item()
            else:
                val_loss = 0

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    preds_norm = []

    zero_norm = (0 - mean) / std
    current_window_values = np.full(seq_len, zero_norm)

    start_date = val_index[0]
    context_dates = pd.date_range(end=start_date, periods=seq_len + 1, freq='D')[:-1]
    full_dates = context_dates.union(val_index)

    full_features = _create_features(full_dates, is_whole)

    with torch.no_grad():
        for i in range(len(val_index)):
            current_features_window = full_features[i: i + seq_len]

            inp_data = np.column_stack([current_window_values, current_features_window])
            inp = torch.tensor(inp_data, dtype=torch.float32, device=DEVICE).unsqueeze(0)

            pred_norm = model(inp).item()
            preds_norm.append(pred_norm)

            current_window_values = np.append(current_window_values[1:], pred_norm)

    preds = np.array(preds_norm) * std + mean
    preds = np.maximum(preds, 0)

    return pd.Series(preds, index=val_index)


def rolling_seasonal_gru(
        df: pd.DataFrame,
        is_whole: bool,
        seq_len: int = 90,
        hidden_dim: int = 64,
        num_layers: int = 2,
        epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int = 16,
        patience: int = 10,
        min_train_seasons: int = 10,
):
    date_col = "Date"
    hs_col = "HS_after_gapfill"

    df = df.copy()
    try:
        df[date_col] = pd.to_datetime(df[date_col])
    except Exception as e:
        print(f"Error parsing date column: {e}")
        return pd.DataFrame(), 0, 0, 0, 0, 0

    df = df.set_index(date_col).sort_index()

    df[hs_col] = pd.to_numeric(df[hs_col], errors='coerce')

    y = df[hs_col].astype(float).asfreq("D").ffill().fillna(0)

    if not is_whole:
        m = y.index.month
        season_mask = (m >= 11) | (m <= 5)
        y = y[season_mask]

    months = y.index.month
    season_year = pd.Series(y.index.year, index=y.index, name="season_year").astype(int)
    if not is_whole:
        season_year[months >= 11] += 1

    seasons = sorted(season_year.unique())
    results = []

    print(f"  > Found {len(seasons)} seasons. Training rolling window...")

    for i in range(min_train_seasons, len(seasons)):
        val_season = seasons[i]
        train_seasons = seasons[:i]

        train_mask = season_year.isin(train_seasons)
        y_train = y[train_mask]
        train_years_slice = season_year[train_mask]

        val_mask = season_year == val_season
        val_index = y.index[val_mask]

        if len(y_train) == 0 or len(val_index) == 0:
            continue

        try:
            preds = gru_predict(
                y_train, train_years_slice, val_index, is_whole,
                seq_len, hidden_dim, num_layers, epochs, lr, batch_size, patience
            )
        except Exception as e:
            print(f"  ! Error on season {val_season}: {e}")
            continue

        if preds.isna().all():
            continue

        y_true = y[val_index]
        mae = (y_true - preds).abs().mean()

        results.append({
            "season_year": val_season,
            "mae": mae,
            "season_mean": y_true.mean(),
            "predicted_mean": preds.mean(),
        })

    results_df = pd.DataFrame(results)
    if results_df.empty:
        return results_df, 0, 0, 0, 0, 0

    total_mae = results_df["mae"].sum()
    total_mean = results_df["season_mean"].sum()
    nmae = total_mae / total_mean if total_mean > 0 else 0.0

    return (
        results_df,
        results_df["mae"].mean(),
        nmae,
        results_df["season_mean"].mean(),
        results_df["predicted_mean"].mean(),
        0
    )


def main():
    filenames = ["col_de_porte_daily.txt", "les2alpes_daily.txt",
                 "serre_chevalier_daily.txt", "tignes_daily.txt"]

    dfs = {}

    print("--- Loading Data ---")
    for filename in filenames:
        try:
            path = "/kaggle/input/stations/" + filename
            dfs[filename] = pd.read_csv(path, sep=',')
            print(f"Loaded {filename}")
        except Exception as e:
            print(f"Failed to load {filename}: {e}")

    for k, df in dfs.items():
        print(f"\n=== Processing {k} ===")
        res, mae, nmae, s_mean, p_mean, pct = rolling_seasonal_gru(
            df,
            is_whole=False,
            seq_len=45,
            hidden_dim=64,
            num_layers=2,
            epochs=150,
            patience=12,
            min_train_seasons=10
        )
        print(f"Final Results for {k}: MAE: {mae:.2f} cm | Weighted NMAE: {nmae * 100:.1f}%")


if __name__ == "__main__":
    main()
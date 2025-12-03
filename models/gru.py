import pandas as pd
import numpy as np
from typing import Tuple

def _ensure_torch():
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import TensorDataset, DataLoader
        return torch, nn, TensorDataset, DataLoader
    except (ImportError, OSError) as e:
        raise RuntimeError(f"PyTorch backend unavailable: {e}")

class SeasonalGRU:
    """Lazy Torch GRU wrapper to avoid import-time failures."""
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, output_dim: int, dropout: float = 0.2, device=None):
        torch, nn, _, _ = _ensure_torch()
        class _Impl(nn.Module):
            def __init__(self):
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

        self.torch = torch
        self.nn = nn
        self.model = _Impl().to(device)

    def __call__(self, x):
        return self.model(x)

    def parameters(self):
        return self.model.parameters()

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state):
        self.model.load_state_dict(state)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

def _create_features(dates: pd.DatetimeIndex, is_whole: bool = False) -> np.ndarray:
    if is_whole:
        day_of_year = dates.dayofyear
        day_sin = np.sin(2 * np.pi * day_of_year / 365.25)
        day_cos = np.cos(2 * np.pi * day_of_year / 365.25)
    else:
        doy = dates.dayofyear
        seasonal_day = np.where(doy >= 305, doy - 305, doy + 60)
        day_sin = np.sin(2 * np.pi * seasonal_day / 213.0)
        day_cos = np.cos(2 * np.pi * seasonal_day / 213.0)
    dow_sin = np.sin(2 * np.pi * dates.dayofweek / 7)
    dow_cos = np.cos(2 * np.pi * dates.dayofweek / 7)
    return np.column_stack([day_sin, day_cos, dow_sin, dow_cos])

def _make_sequences(series: np.ndarray, features: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(0, len(series) - seq_len, max(1, seq_len // 2)):
        if i + seq_len >= len(series):
            break
        X.append(np.column_stack([series[i:i + seq_len], features[i:i + seq_len]]))
        y.append(series[i + seq_len])
    return np.array(X), np.array(y)

def gru_predict(
    train_series: pd.Series,
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
    torch, nn, TensorDataset, DataLoader = _ensure_torch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    y_train = pd.to_numeric(train_series, errors="coerce").ffill().bfill()
    values = y_train.values.astype(float)
    if len(values) <= seq_len + 20:
        return pd.Series([float("nan")] * len(val_index), index=val_index)

    mean, std = values.mean(), values.std() or 1.0
    values_norm = (values - mean) / std

    train_features = _create_features(y_train.index, is_whole)
    X, y = _make_sequences(values_norm, train_features, seq_len)
    if len(X) < 10:
        return pd.Series([float("nan")] * len(val_index), index=val_index)

    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train_seq, y_val_seq = y[:split_idx], y[split_idx:]

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train_seq, dtype=torch.float32, device=device).unsqueeze(1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val_seq, dtype=torch.float32, device=device).unsqueeze(1)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
    input_dim = X.shape[2]
    model = SeasonalGRU(input_dim, hidden_dim, num_layers, 1, device=device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    best_val, patience_ctr, best_state = float("inf"), 0, None
    for _ in range(epochs):
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
            val_preds = model(X_val_t)
            val_loss = criterion(val_preds, y_val_t).item()
        if val_loss < best_val:
            best_val, best_state, patience_ctr = val_loss, model.state_dict(), 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    preds_norm = []
    last_values = values_norm[-seq_len:].copy()
    last_dates = y_train.index[-seq_len:]
    with torch.no_grad():
        for target_date in val_index:
            current_features = _create_features(last_dates, is_whole)
            inp = np.column_stack([last_values, current_features])
            inp_t = torch.tensor(inp, dtype=torch.float32, device=device).unsqueeze(0)
            pred_norm = model(inp_t).item()
            preds_norm.append(pred_norm)
            last_values = np.append(last_values[1:], pred_norm)
            last_dates = pd.DatetimeIndex(list(last_dates[1:]) + [target_date])
    preds = np.array(preds_norm) * std + mean
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
    _ensure_torch()  # fail fast if backend missing

    date_col = "Date"
    hs_col = "HS_after_gapfill"
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col])
    d = d.set_index(date_col).sort_index()
    y = d[hs_col].astype(float).asfreq("D").ffill()

    if not is_whole:
        m = y.index.month
        y = y[(m >= 11) | (m <= 5)]

    months = y.index.month
    season_year = pd.Series(y.index.year, index=y.index, name="season_year").astype(int)
    if not is_whole:
        season_year[months <= 5] -= 1

    seasons = sorted(season_year.unique())
    results = []
    for i in range(min_train_seasons, len(seasons)):
        val_season = seasons[i]
        y_train = y[season_year.isin(seasons[:i])]
        val_index = y.index[season_year == val_season]
        if len(y_train) == 0 or len(val_index) == 0:
            continue
        preds = gru_predict(y_train, val_index, is_whole, seq_len, hidden_dim, num_layers, epochs, lr, batch_size, patience)
        if preds.isna().all():
            continue
        y_true = y[val_index]
        mae = (y_true - preds).abs().mean()
        season_mean = y_true.mean()
        predicted_mean = preds.mean()
        pct_error = ((predicted_mean - season_mean) / season_mean) * 100.0
        nmae = mae / season_mean if season_mean != 0 else float("nan")
        results.append({
            "season_year": val_season,
            "mae": mae,
            "nmae": nmae,
            "season_mean": season_mean,
            "predicted_mean": predicted_mean,
            "pct_error": pct_error
        })
    results_df = pd.DataFrame(results)
    return (
        results_df,
        results_df["mae"].mean(),
        results_df["nmae"].mean(),
        results_df["season_mean"].mean(),
        results_df["predicted_mean"].mean(),
        results_df["pct_error"].mean()
    )

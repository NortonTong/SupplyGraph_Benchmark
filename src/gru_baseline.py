import numpy as np
import pandas as pd
from math import sqrt
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error

from config.config import PROC_DIR, LAG_WINDOWS


# ================= EarlyStopping =================

class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def step(self, metric: float) -> bool:
        # metric: val_rmse (càng nhỏ càng tốt)
        score = -metric

        if self.best_score is None:
            self.best_score = score
            self.counter = 0
            return False

        if score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        else:
            self.best_score = score
            self.counter = 0

        return False


# ================= Dataset & Model =================

class GRUDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class GRURegressor(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, h_n = self.gru(x)          # h_n: (num_layers, batch, hidden)
        last_hidden = h_n[-1]           # (batch, hidden)
        y_hat = self.fc(last_hidden).squeeze(-1)
        return y_hat


# ================= Data utils =================

def load_baseline(horizon: int, lag_window: int | None = None) -> pd.DataFrame:
    if lag_window is None:
        lag_window = LAG_WINDOWS[0]

    path = PROC_DIR / "baseline" / f"gru_ready_h{horizon}_lag{lag_window}.parquet"
    print(f"Loading GRU-ready baseline from {path}")
    df = pd.read_parquet(path)

    num_cols = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    df[num_cols] = df[num_cols].fillna(0.0)

    return df

def build_sequences(
    df: pd.DataFrame,
    window: int,
    split: str = "train",
):
    df_split = df[df["split"] == split].copy()
    df_split = df_split.sort_values(["node_id", "day"])

    drop_cols = [
        "target",
        "split",
        "node_id",
        "node_index",
        "date",
        "day",
    ]

    candidate_cols = [c for c in df_split.columns if c not in drop_cols]
    numeric_cols = df_split[candidate_cols].select_dtypes(
        include=["number", "bool"]
    ).columns.tolist()
    feature_cols = numeric_cols

    # Log NaN theo cột trước dropna
    na_counts = df_split[feature_cols + ["target"]].isna().sum()
    print(f"\n[DEBUG] {split} NaN counts BEFORE dropna:")
    print(na_counts[na_counts > 0])

    # Bỏ các hàng có NaN trong feature hoặc target (sau khi fill 0, bước này chủ yếu là safety)
    df_split = df_split.dropna(subset=feature_cols + ["target"])

    # Log lại sau dropna
    na_counts_after = df_split[feature_cols + ["target"]].isna().sum()
    print(f"[DEBUG] {split} NaN counts AFTER dropna (should be empty):")
    print(na_counts_after[na_counts_after > 0])

    X_list, y_list = [], []

    for node_id, grp in df_split.groupby("node_id"):
        grp = grp.reset_index(drop=True)
        values = grp[feature_cols].astype(float).values
        targets = grp["target"].astype(float).values

        if len(grp) <= window:
            continue

        for t in range(window, len(grp)):
            X_seq = values[t - window : t, :]
            y_t = targets[t]
            X_list.append(X_seq)
            y_list.append(y_t)

    if not X_list:
        return None, None, feature_cols

    X = np.stack(X_list, axis=0)
    y = np.asarray(y_list)

    print("DEBUG build_sequences:", split,
          "| X NaN:", np.isnan(X).sum(),
          "X inf:", np.isinf(X).sum(),
          "| y NaN:", np.isnan(y).sum(),
          "y inf:", np.isinf(y).sum())

    return X, y, feature_cols


def make_dataloaders(
    df: pd.DataFrame,
    window: int,
    batch_size: int,
):
    X_train, y_train, feature_cols = build_sequences(
        df, window=window, split="train"
    )
    X_val, y_val, _ = build_sequences(df, window=window, split="val")
    X_test, y_test, _ = build_sequences(df, window=window, split="test")

    train_ds = GRUDataset(X_train, y_train)
    val_ds = GRUDataset(X_val, y_val)
    test_ds = GRUDataset(X_test, y_test)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    seq_len = X_train.shape[1]
    input_size = X_train.shape[2]

    return (
        train_loader,
        val_loader,
        test_loader,
        feature_cols,
        (seq_len, input_size),
    )


# ================= Training =================

def train_one_gru(
    horizon: int,
    lag_window: int,
    window: int,
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.2,
    batch_size: int = 256,
    n_epochs: int = 200,
    lr: float = 1e-3,
    device: str | None = None,
    tag: str = "GRU",
    patience: int = 20,
    min_delta: float = 0.0,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    df = load_baseline(horizon=horizon, lag_window=lag_window)

    (
        train_loader,
        val_loader,
        test_loader,
        feature_cols,
        (seq_len, input_size),
    ) = make_dataloaders(
        df=df,
        window=window,
        batch_size=batch_size,
    )

    model = GRURegressor(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_rmse = float("inf")
    best_state = None
    early_stopper = EarlyStopping(patience=patience, min_delta=min_delta)

    print(
        f"\n=== Training GRU H{horizon} lag{lag_window} ({tag}) "
        f"seq_len={seq_len}, n_features={input_size} ==="
    )

    for epoch in range(1, n_epochs + 1):
        # ---------- Train ----------
        model.train()
        train_losses = []

        for Xb, yb in train_loader:
            Xb = Xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            y_pred = model(Xb)
            loss = criterion(y_pred, yb)
            loss.backward()

            # optional: gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_losses.append(loss.item())

        # ---------- Validation ----------
        model.eval()
        val_preds, val_trues = [], []

        with torch.no_grad():
            for i, (Xb, yb) in enumerate(val_loader):
                X_np = Xb.numpy()
                nan_mask = np.isnan(X_np)
                if nan_mask.any():
                    nan_per_feature = nan_mask.sum(axis=(0, 1))
                    print(f"[VAL BATCH {i}] X NaN per feature (index:count):")
                    for j, c in enumerate(feature_cols):
                        if nan_per_feature[j] > 0:
                            print(f"  {c}: {nan_per_feature[j]} NaNs")

                Xb = Xb.to(device)
                yb = yb.to(device)
                y_pred = model(Xb)

                val_preds.append(y_pred.cpu().numpy())
                val_trues.append(yb.cpu().numpy())

        val_preds = np.concatenate(val_preds)
        val_trues = np.concatenate(val_trues)

        # Lọc NaN/inf ở output (safety)
        mask = np.isfinite(val_preds) & np.isfinite(val_trues)
        val_preds = val_preds[mask]
        val_trues = val_trues[mask]

        if len(val_trues) == 0:
            print("Warning: no valid validation samples after filtering NaNs.")
            return model, {
                "horizon": horizon,
                "lag_window": lag_window,
                "tag": tag,
                "val_rmse": np.nan,
                "test_rmse": np.nan,
                "test_mae": np.nan,
                "n_features": input_size,
                "seq_len": seq_len,
            }

        val_rmse = sqrt(mean_squared_error(val_trues, val_preds))
        val_mae = mean_absolute_error(val_trues, val_preds)

        print(
            f"Epoch {epoch:03d} | "
            f"TrainLoss={np.mean(train_losses):.4f} | "
            f"ValRMSE={val_rmse:.4f} | ValMAE={val_mae:.4f}"
        )

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = model.state_dict()

        if early_stopper.step(val_rmse):
            print(f"Early stopping triggered at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # ---------- Test ----------
    model.eval()
    test_preds, test_trues = [], []
    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb = Xb.to(device)
            y_pred = model(Xb)
            test_preds.append(y_pred.cpu().numpy())
            test_trues.append(yb.numpy())

    test_preds = np.concatenate(test_preds)
    test_trues = np.concatenate(test_trues)

    mask = np.isfinite(test_preds) & np.isfinite(test_trues)
    test_preds = test_preds[mask]
    test_trues = test_trues[mask]

    if len(test_trues) == 0:
        print("Warning: no valid test samples after filtering NaNs.")
        test_rmse = np.nan
        test_mae = np.nan
    else:
        test_rmse = sqrt(mean_squared_error(test_trues, test_preds))
        test_mae = mean_absolute_error(test_trues, test_preds)

    print(
        f"\n[GRU][H{horizon}][lag{lag_window}][{tag}] "
        f"Test RMSE={test_rmse:.4f}, MAE={test_mae:.4f}"
    )

    # Lưu model
    out_dir = PROC_DIR / "models" / "gru"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"gru_h{horizon}_lag{lag_window}_{tag}.pth"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "horizon": horizon,
            "lag_window": lag_window,
            "window": window,
            "input_size": input_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "feature_cols": feature_cols,
        },
        model_path,
    )
    print(f"Saved GRU model to {model_path}")

    info = {
        "horizon": horizon,
        "lag_window": lag_window,
        "tag": tag,
        "val_rmse": best_val_rmse,
        "test_rmse": test_rmse,
        "test_mae": test_mae,
        "n_features": input_size,
        "seq_len": seq_len,
    }

    return model, info

def rolling_origin_evaluation_gru(
    horizon: int,
    lag_window: int,
    window: int,
    origins: list[int],
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.2,
    batch_size: int = 256,
    n_epochs: int = 200,
    lr: float = 1e-3,
    device: str | None = None,
    tag_prefix: str = "gru_rolling",
    patience: int = 20,
    min_delta: float = 0.0,
):
    """
    Rolling-origin evaluation: với mỗi origin T, train GRU trên day <= T,
    predict day = T + horizon, tính MAE/RMSE, rồi tổng hợp.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    df_full = load_baseline(horizon=horizon, lag_window=lag_window)
    errors = []

    for T in origins:
        print(f"\n=== GRU Rolling origin at day {T}, H={horizon}, lag={lag_window} ===")

        # Train: day <= T, Test: day == T + horizon
        df_train = df_full[df_full["day"] <= T].copy()
        df_test  = df_full[df_full["day"] == T + horizon].copy()

        if df_test.empty:
            print(f"No test samples for origin {T} (day={T+horizon}). Skipping.")
            continue

        # Build sequences riêng cho train & test (không dùng cột split)
        def build_seq_for_df(df_sub: pd.DataFrame, split_name: str):
            # set split tạm để reuse build_sequences
            df_tmp = df_sub.copy()
            df_tmp["split"] = split_name
            X, y, feature_cols = build_sequences(df_tmp, window=window, split=split_name)
            return X, y, feature_cols

        X_train, y_train, feature_cols = build_seq_for_df(df_train, "train")
        X_test,  y_test,  _            = build_seq_for_df(df_test,  "test")
        if X_train is None or y_train is None:
            print(f"  Skip origin {T}: no train sequences.")
            continue
        if X_test is None or y_test is None:
            print(f"  Skip origin {T}: no test sequences (not enough history for window={window}).")
            continue
        train_ds = GRUDataset(X_train, y_train)
        test_ds  = GRUDataset(X_test, y_test)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
        test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        model = GRURegressor(
            input_size=X_train.shape[2],
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
        ).to(device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        early_stopper = EarlyStopping(patience=patience, min_delta=min_delta)

        # Train cho origin này
        for epoch in range(1, n_epochs + 1):
            model.train()
            train_losses = []
            for Xb, yb in train_loader:
                Xb = Xb.to(device)
                yb = yb.to(device)
                optimizer.zero_grad()
                y_pred = model(Xb)
                loss = criterion(y_pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_losses.append(loss.item())

            # simple early stopping trên train loss (vì không có val riêng)
            mean_loss = np.mean(train_losses)
            if early_stopper.step(mean_loss):
                print(f"  Early stop at epoch {epoch}, train_loss={mean_loss:.4f}")
                break

        # Evaluate on test (day T + horizon)
        model.eval()
        test_preds, test_trues = [], []
        with torch.no_grad():
            for Xb, yb in test_loader:
                Xb = Xb.to(device)
                y_pred = model(Xb)
                test_preds.append(y_pred.cpu().numpy())
                test_trues.append(yb.numpy())

        test_preds = np.concatenate(test_preds)
        test_trues = np.concatenate(test_trues)

        mask = np.isfinite(test_preds) & np.isfinite(test_trues)
        test_preds = test_preds[mask]
        test_trues = test_trues[mask]

        if len(test_trues) == 0:
            print("  Warning: no valid test samples after filtering NaNs.")
            continue

        mae = mean_absolute_error(test_trues, test_preds)
        rmse = sqrt(mean_squared_error(test_trues, test_preds))

        print(f"  MAE  : {mae:.4f}")
        print(f"  RMSE : {rmse:.4f}")

        errors.append({"origin_day": T, "MAE": mae, "RMSE": rmse})

    if errors:
        df_err = pd.DataFrame(errors)
        print("\n=== GRU Rolling-origin summary ===")
        print(df_err)
        print("\nAverage over origins:")
        print(df_err.mean(numeric_only=True))

        out_dir = PROC_DIR / "predictions" / "gru" / "rolling"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"gru_rolling_H{horizon}_lag{lag_window}.csv"
        df_err.to_csv(out_path, index=False)
        print(f"\nSaved GRU rolling-origin results to {out_path}")
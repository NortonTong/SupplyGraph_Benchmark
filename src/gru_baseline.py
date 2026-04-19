import numpy as np
import pandas as pd
from math import sqrt
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error

from config.config import PROC_DIR

# ====================== Metrics ======================

def mape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.abs(y_true) > eps
    if mask.sum() == 0:
        return np.nan
    return float(
        np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0
    )

def smape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) + eps
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom) * 100.0)

# ====================== Target transform ======================

TARGET_TYPES = ["raw", "log1p"]

def transform_target(y: np.ndarray, target_type: str) -> np.ndarray:
    if target_type == "raw":
        return y
    elif target_type == "log1p":
        # y >= 0, log1p ổn cho zero-inflated demand / outlier.[web:49][web:51]
        return np.log1p(np.clip(y, a_min=0.0, a_max=None))
    else:
        raise ValueError(f"Unknown target_type={target_type}")

def inverse_transform_target(y_hat: np.ndarray, target_type: str) -> np.ndarray:
    if target_type == "raw":
        y = y_hat
    elif target_type == "log1p":
        y = np.expm1(y_hat)
    else:
        raise ValueError(f"Unknown target_type={target_type}")
    # đảm bảo không âm
    return np.maximum(y, 0.0)

# ====================== EarlyStopping ======================

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

# ====================== Dataset & Model ======================

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

# ====================== Data utils ======================

def load_baseline(
    horizon: int,
    temporal_type: str = "unit",
) -> pd.DataFrame:
    """
    Dùng GRU base no-graph: baseline/gru_sequence_{temporal_type}.parquet
    Horizon=7 => target gốc là 'target' (y_h7 trong preprocessing).
    """
    assert horizon == 7, "GRU preprocessing hiện chỉ hỗ trợ H=7."

    path = PROC_DIR / "baseline" / f"gru_sequence_{temporal_type}.parquet"
    print(f"Loading GRU base from {path}")
    df = pd.read_parquet(path)

    df = df.copy()
    df["target"] = df["target"].astype(float)

    num_cols = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    df[num_cols] = df[num_cols].fillna(0.0)

    return df

def build_sequences(
    df: pd.DataFrame,
    window: int,
    split: str = "train",
):
    """
    Tạo sequence (X, y) cho một split (train/val/test):
    - X: (num_samples, window, num_features)
    - y: (num_samples,)
    """
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

    # Debug NaN trước dropna
    na_counts = df_split[feature_cols + ["target"]].isna().sum()
    print(f"\n[DEBUG] {split} NaN counts BEFORE dropna:")
    print(na_counts[na_counts > 0])

    df_split = df_split.dropna(subset=feature_cols + ["target"])

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

    print(
        "DEBUG build_sequences:",
        split,
        "| X NaN:", np.isnan(X).sum(),
        "X inf:", np.isinf(X).sum(),
        "| y NaN:", np.isnan(y).sum(),
        "y inf:", np.isinf(y).sum(),
    )

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

# ====================== Training (fixed split) ======================

def train_one_gru(
    horizon: int,
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
    temporal_type: str = "unit",
    target_type: str = "raw",
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    df = load_baseline(
        horizon=horizon,
        temporal_type=temporal_type,
    )

    # Transform target trước khi build sequence (model học trên scale y')
    df = df.copy()
    df["target"] = transform_target(df["target"].values, target_type)

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
        f"\n=== Training GRU Baseline 2 H{horizon}, window={window}, target_type={target_type} "
        f"({tag}, {temporal_type}) seq_len={seq_len}, n_features={input_size} ==="
    )

    for epoch in range(1, n_epochs + 1):
        # ---------- Train ----------
        model.train()
        train_losses = []
        train_trues, train_preds = [], []

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
            train_trues.append(yb.detach().cpu().numpy())
            train_preds.append(y_pred.detach().cpu().numpy())

        train_trues = np.concatenate(train_trues)
        train_preds = np.concatenate(train_preds)

        # inverse về scale gốc + clip
        train_trues_inv = inverse_transform_target(train_trues, target_type)
        train_preds_inv = inverse_transform_target(train_preds, target_type)

        train_mask = np.isfinite(train_trues_inv) & np.isfinite(train_preds_inv)
        train_trues_inv = train_trues_inv[train_mask]
        train_preds_inv = train_preds_inv[train_mask]

        if len(train_trues_inv) == 0:
            train_rmse = np.nan
            train_mae = np.nan
            train_mape_val = np.nan
            train_smape_val = np.nan
        else:
            train_rmse = sqrt(mean_squared_error(train_trues_inv, train_preds_inv))
            train_mae = mean_absolute_error(train_trues_inv, train_preds_inv)
            train_mape_val = mape(train_trues_inv, train_preds_inv)
            train_smape_val = smape(train_trues_inv, train_preds_inv)

        # ---------- Validation ----------
        model.eval()
        val_preds, val_trues = [], []

        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb = Xb.to(device)
                yb = yb.to(device)
                y_pred = model(Xb)
                val_preds.append(y_pred.cpu().numpy())
                val_trues.append(yb.cpu().numpy())

        val_preds = np.concatenate(val_preds)
        val_trues = np.concatenate(val_trues)

        val_trues_inv = inverse_transform_target(val_trues, target_type)
        val_preds_inv = inverse_transform_target(val_preds, target_type)

        mask = np.isfinite(val_preds_inv) & np.isfinite(val_trues_inv)
        val_preds_inv = val_preds_inv[mask]
        val_trues_inv = val_trues_inv[mask]

        if len(val_trues_inv) == 0:
            print("Warning: no valid validation samples after filtering NaNs.")
            val_rmse = np.nan
            val_mae = np.nan
            val_mape_val = np.nan
            val_smape_val = np.nan
        else:
            val_rmse = sqrt(mean_squared_error(val_trues_inv, val_preds_inv))
            val_mae = mean_absolute_error(val_trues_inv, val_preds_inv)
            val_mape_val = mape(val_trues_inv, val_preds_inv)
            val_smape_val = smape(val_trues_inv, val_preds_inv)

        print(
            f"Epoch {epoch:03d} | "
            f"TrainLoss={np.mean(train_losses):.4f} | "
            f"TrainRMSE={train_rmse:.4f} | TrainMAE={train_mae:.4f} | "
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

    test_trues_inv = inverse_transform_target(test_trues, target_type)
    test_preds_inv = inverse_transform_target(test_preds, target_type)

    mask = np.isfinite(test_preds_inv) & np.isfinite(test_trues_inv)
    test_preds_inv = test_preds_inv[mask]
    test_trues_inv = test_trues_inv[mask]

    if len(test_trues_inv) == 0:
        print("Warning: no valid test samples after filtering NaNs.")
        test_rmse = np.nan
        test_mae = np.nan
        test_mape_val = np.nan
        test_smape_val = np.nan
    else:
        test_rmse = sqrt(mean_squared_error(test_trues_inv, test_preds_inv))
        test_mae = mean_absolute_error(test_trues_inv, test_preds_inv)
        test_mape_val = mape(test_trues_inv, test_preds_inv)
        test_smape_val = smape(test_trues_inv, test_preds_inv)

    print(
        f"\n[GRU Baseline 2][H{horizon}][window={window}][{target_type}][{tag}][{temporal_type}] "
        f"Test RMSE={test_rmse:.4f}, MAE={test_mae:.4f}, "
        f"MAPE={test_mape_val:.4f}, sMAPE={test_smape_val:.4f}"
    )

    # Lưu model
    out_dir_model = PROC_DIR / "models" / "gru" / "baseline_2"
    out_dir_model.mkdir(parents=True, exist_ok=True)
    model_path = out_dir_model / f"gru_baseline2_h{horizon}_w{window}_{target_type}_{temporal_type}.pth"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "horizon": horizon,
            "window": window,
            "input_size": input_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "feature_cols": feature_cols,
            "temporal_type": temporal_type,
            "target_type": target_type,
        },
        model_path,
    )
    print(f"Saved GRU model to {model_path}")

    # Lưu test predictions (scale gốc)
    out_dir_pred = PROC_DIR / "predictions" / "baseline_2" / "gru" / "csv" / temporal_type
    out_dir_pred.mkdir(parents=True, exist_ok=True)
    df_test_pred = pd.DataFrame({"y_true": test_trues_inv, "y_pred": test_preds_inv})
    out_pred_file = out_dir_pred / f"gru_baseline2_h{horizon}_w{window}_{target_type}_{temporal_type}_test_predictions.csv"
    df_test_pred.to_csv(out_pred_file, index=False)
    print(f"Saved GRU test predictions to {out_pred_file}")

    info = {
        "horizon": horizon,
        "window": window,
        "tag": tag,
        "temporal_type": temporal_type,
        "target_type": target_type,
        "train_rmse": train_rmse,
        "train_mae": train_mae,
        "train_mape": train_mape_val,
        "train_smape": train_smape_val,
        "val_rmse": best_val_rmse,
        "val_mae": val_mae,
        "val_mape": val_mape_val,
        "val_smape": val_smape_val,
        "test_rmse": test_rmse,
        "test_mae": test_mae,
        "test_mape": test_mape_val,
        "test_smape": test_smape_val,
        "n_features": input_size,
        "seq_len": seq_len,
    }

    return model, info

# ====================== Rolling-origin GRU ======================

def rolling_origin_evaluation_gru(
    horizon: int,
    window: int,
    origins: list[int],
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.2,
    batch_size: int = 256,
    n_epochs: int = 200,
    lr: float = 1e-3,
    device: str | None = None,
    tag_prefix: str = "gru_baseline2_rolling",
    patience: int = 20,
    min_delta: float = 0.0,
    temporal_type: str = "unit",
    target_type: str = "raw",  # bạn có thể dùng "raw" hoặc "log1p"
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    df_full = load_baseline(
        horizon=horizon,
        temporal_type=temporal_type,
    )

    # transform target để train (như fixed-split)
    df_full = df_full.copy()
    df_full["target"] = transform_target(df_full["target"].values, target_type)

    errors = []

    for T in origins:
        print(f"\n=== GRU Baseline 2 Rolling origin at day {T}, H={horizon}, "
              f"window={window}, temporal_type={temporal_type}, target_type={target_type} ===")

        df_train = df_full[df_full["day"] <= T].copy()
        df_test  = df_full[df_full["day"] == T + horizon].copy()

        if df_test.empty:
            print(f"No test samples for origin {T} (day={T+horizon}). Skipping.")
            continue

        def build_seq_for_df(df_sub: pd.DataFrame, split_name: str):
            df_tmp = df_sub.copy()
            df_tmp["split"] = split_name
            X, y, feature_cols = build_sequences(df_tmp, window=window, split=split_name)
            return X, y, feature_cols

        X_train, y_train, feature_cols = build_seq_for_df(df_train, "train")
        X_test,  y_test,  _           = build_seq_for_df(df_test,  "test")

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

            mean_loss = np.mean(train_losses)
            if early_stopper.step(mean_loss):
                print(f"  Early stop at epoch {epoch}, train_loss={mean_loss:.4f}")
                break

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

        # inverse về scale gốc + clip
        test_trues_inv = inverse_transform_target(test_trues, target_type)
        test_preds_inv = inverse_transform_target(test_preds, target_type)

        mask = np.isfinite(test_preds_inv) & np.isfinite(test_trues_inv)
        test_preds_inv = test_preds_inv[mask]
        test_trues_inv = test_trues_inv[mask]

        if len(test_trues_inv) == 0:
            print("  Warning: no valid test samples after filtering NaNs.")
            continue

        mae = mean_absolute_error(test_trues_inv, test_preds_inv)
        rmse = sqrt(mean_squared_error(test_trues_inv, test_preds_inv))

        print(f"  MAE  : {mae:.4f}")
        print(f"  RMSE : {rmse:.4f}")

        errors.append({"origin_day": T, "MAE": mae, "RMSE": rmse})

    if errors:
        df_err = pd.DataFrame(errors)
        print("\n=== GRU Baseline 2 Rolling-origin summary ===")
        print(df_err)
        print("\nAverage over origins:")
        print(df_err.mean(numeric_only=True))

        out_dir = PROC_DIR / "predictions" / "baseline_2" / "gru" / "rolling"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"gru_baseline2_rolling_H{horizon}_w{window}_{target_type}_{temporal_type}.csv"
        df_err.to_csv(out_path, index=False)
        print(f"\nSaved GRU rolling-origin results to {out_path}")
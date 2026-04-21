import numpy as np
import pandas as pd
from math import sqrt
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

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
        is_softplus: bool = False,
        is_log1p: bool = False,
    ):
        """
        - is_softplus=True:  y_hat = softplus(head)        (>= 0, smooth)
        - is_log1p=True:    head ~ log1p(y), y_hat = expm1(head).clamp_min(0)
        - cả 2 False:       y_hat = head (raw)
        Không được bật đồng thời cả 2.
        """
        super().__init__()
        if is_softplus and is_log1p:
            raise ValueError("Only one of is_softplus / is_log1p can be True.")

        self.is_softplus = is_softplus
        self.is_log1p = is_log1p

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
        if self.is_softplus:
            self.softplus = nn.Softplus()

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, h_n = self.gru(x)       # h_n: (num_layers, batch, hidden)
        last_hidden = h_n[-1]        # (batch, hidden)
        head = self.fc(last_hidden).squeeze(-1)  # (batch,)

        if self.is_softplus:
            y_hat = self.softplus(head)
        elif self.is_log1p:
            y_hat = torch.expm1(head).clamp_min(0.0)
        else:
            y_hat = head

        return y_hat


# ====================== Data utils ======================


def load_baseline(
    horizon: int,
    window: int,
    temporal_type: str = "unit",

) -> pd.DataFrame:
    """
    Dùng GRU base no-graph: baseline/gru_sequence_{temporal_type}.parquet
    Horizon=7 => target gốc là 'target' (y_h7 trong preprocessing).
    """
    assert horizon == 7, "GRU preprocessing hiện chỉ hỗ trợ H=7."

    path = PROC_DIR / "baseline" / "gru" / f"gru_sequence_h{horizon}_L{window}_{temporal_type}.parquet"
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
    - meta: DataFrame (num_samples, [node_id, date])
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

    X_list, y_list = [], []
    meta_list = []

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
            meta_list.append(
                {
                    "node_id": grp.loc[t, "node_id"],
                    "date": grp.loc[t, "date"],
                }
            )

    if not X_list:
        return None, None, feature_cols, None

    X = np.stack(X_list, axis=0)
    y = np.asarray(y_list)
    meta = pd.DataFrame(meta_list)

    return X, y, feature_cols, meta


def make_dataloaders(
    df: pd.DataFrame,
    window: int,
    batch_size: int,
):
    X_train, y_train, feature_cols, meta_train = build_sequences(
        df, window=window, split="train"
    )
    X_val, y_val, _, meta_val = build_sequences(df, window=window, split="val")
    X_test, y_test, _, meta_test = build_sequences(df, window=window, split="test")

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
        meta_train,
        meta_val,
        meta_test,
    )


# ====================== Plot predictions per product ======================


def plot_predictions_per_product_gru(
    df_test: pd.DataFrame,
    out_dir: Path,
    horizon: int,
    window: int,
    temporal_type: str,
    transform_tag: str,
    max_plots: int | None = None,
) -> None:
    """
    Vẽ y_true vs y_pred theo ngày cho từng sản phẩm (node_id) trên test split.
    Lưu 1 file .png / sản phẩm vào out_dir, prefix gru_baseline2_*.
    """
    df_plot = df_test.copy()
    unique_nodes = df_plot["node_id"].unique()
    if max_plots is not None:
        unique_nodes = unique_nodes[:max_plots]

    for node in unique_nodes:
        sub = df_plot[df_plot["node_id"] == node].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("date")

        plt.figure(figsize=(10, 4))
        plt.plot(sub["date"], sub["y_true"], label="True", marker="o", linewidth=1)
        plt.plot(sub["date"], sub["y_pred"], label="Pred", marker="x", linewidth=1)
        plt.title(
            f"GRU Baseline 2 (H={horizon}, w={window}, {temporal_type}, {transform_tag}) - node_id={node}"
        )
        plt.xlabel("Date")
        plt.ylabel("Sales order")
        plt.legend()
        plt.tight_layout()

        fname = (
            out_dir
            / f"gru_baseline2_h{horizon}_w{window}_{transform_tag}_{temporal_type}_node_{node}.png"
        )
        plt.savefig(fname, dpi=150)
        plt.close()


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
    is_softplus: bool = False,
    is_log1p: bool = False,
):
    """
    - Không còn target_type ngoài model.
    - is_softplus / is_log1p điều khiển output layer trong GRURegressor.
    - Loss + metrics luôn tính trên scale output của model.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    df = load_baseline(
        horizon=horizon,
        temporal_type=temporal_type,
        window = window,
    )

    (
        train_loader,
        val_loader,
        test_loader,
        feature_cols,
        (seq_len, input_size),
        meta_train,
        meta_val,
        meta_test,
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
        is_softplus=is_softplus,
        is_log1p=is_log1p,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_rmse = float("inf")
    best_state = None
    early_stopper = EarlyStopping(patience=patience, min_delta=min_delta)

    transform_tag = "raw"
    if is_softplus:
        transform_tag = "softplus"
    elif is_log1p:
        transform_tag = "log1p-head"

    print(
        f"\n=== Training GRU Baseline 2 H{horizon}, window={window}, "
        f"output={transform_tag} ({tag}, {temporal_type}) "
        f"seq_len={seq_len}, n_features={input_size} ==="
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

        train_mask = np.isfinite(train_trues) & np.isfinite(train_preds)
        train_trues = train_trues[train_mask]
        train_preds = train_preds[train_mask]

        if len(train_trues) == 0:
            train_rmse = np.nan
            train_mae = np.nan
            train_mape_val = np.nan
            train_smape_val = np.nan
        else:
            train_rmse = sqrt(mean_squared_error(train_trues, train_preds))
            train_mae = mean_absolute_error(train_trues, train_preds)
            train_mape_val = mape(train_trues, train_preds)
            train_smape_val = smape(train_trues, train_preds)

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

        mask = np.isfinite(val_preds) & np.isfinite(val_trues)
        val_preds = val_preds[mask]
        val_trues = val_trues[mask]

        if len(val_trues) == 0:
            print("Warning: no valid validation samples after filtering NaNs.")
            val_rmse = np.nan
            val_mae = np.nan
            val_mape_val = np.nan
            val_smape_val = np.nan
        else:
            val_rmse = sqrt(mean_squared_error(val_trues, val_preds))
            val_mae = mean_absolute_error(val_trues, val_preds)
            val_mape_val = mape(val_trues, val_preds)
            val_smape_val = smape(val_trues, val_preds)

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
            yb = yb.to(device)
            y_pred = model(Xb)
            test_preds.append(y_pred.cpu().numpy())
            test_trues.append(yb.cpu().numpy())

    test_preds = np.concatenate(test_preds)
    test_trues = np.concatenate(test_trues)

    mask = np.isfinite(test_preds) & np.isfinite(test_trues)
    test_preds = test_preds[mask]
    test_trues = test_trues[mask]
    meta_test = meta_test.iloc[mask].reset_index(drop=True)

    if len(test_trues) == 0:
        print("Warning: no valid test samples after filtering NaNs.")
        test_rmse = np.nan
        test_mae = np.nan
        test_mape_val = np.nan
        test_smape_val = np.nan
    else:
        test_rmse = sqrt(mean_squared_error(test_trues, test_preds))
        test_mae = mean_absolute_error(test_trues, test_preds)
        test_mape_val = mape(test_trues, test_preds)
        test_smape_val = smape(test_trues, test_preds)

    print(
        f"\n[GRU Baseline 2][H{horizon}][window={window}][{transform_tag}][{tag}][{temporal_type}] "
        f"Test RMSE={test_rmse:.4f}, MAE={test_mae:.4f}, "
        f"MAPE={test_mape_val:.4f}, sMAPE={test_smape_val:.4f}"
    )

    # ===== Lưu model theo folder transform_tag =====
    out_dir_model = PROC_DIR / "models" / "gru" / "baseline_2" / transform_tag
    out_dir_model.mkdir(parents=True, exist_ok=True)
    model_path = out_dir_model / f"gru_baseline2_h{horizon}_w{window}_{temporal_type}.pth"
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
            "output_transform": transform_tag,
        },
        model_path,
    )
    print(f"Saved GRU model to {model_path}")

    # ===== Lưu test predictions theo folder transform_tag =====
    out_dir_pred = (
        PROC_DIR
        / "predictions"
        / "baseline_2"
        / "gru"
        / "csv"
        / transform_tag
        / temporal_type
    )
    out_dir_pred.mkdir(parents=True, exist_ok=True)
    df_test_pred = meta_test.copy()
    df_test_pred["y_true"] = test_trues
    df_test_pred["y_pred"] = test_preds
    out_pred_file = out_dir_pred / f"gru_baseline2_h{horizon}_w{window}_test_predictions.csv"
    df_test_pred.to_csv(out_pred_file, index=False)
    print(f"Saved GRU test predictions to {out_pred_file}")

    # ===== Lưu plots theo folder transform_tag =====
    plot_dir = (
        PROC_DIR
        / "predictions"
        / "baseline_2"
        / "gru"
        / "plots_gru"
        / transform_tag
        / f"h{horizon}_w{window}"
        / temporal_type
    )
    plot_dir.mkdir(parents=True, exist_ok=True)

    plot_predictions_per_product_gru(
        df_test=df_test_pred,
        out_dir=plot_dir,
        horizon=horizon,
        window=window,
        temporal_type=temporal_type,
        transform_tag=transform_tag,
        max_plots=None,
    )
    print(f"Saved per-product GRU prediction plots to {plot_dir}")

    info = {
        "horizon": horizon,
        "window": window,
        "tag": tag,
        "temporal_type": temporal_type,
        "output_transform": transform_tag,
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
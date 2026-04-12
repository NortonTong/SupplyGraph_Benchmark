# train_tgcn.py

import re
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

from config.config import PROC_DIR
from tgcn_model import TGCN


GNN_DIR = PROC_DIR / "gnn"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

L_HISTORY = 30  # số bước lịch sử dùng cho mỗi sample


class TGCNDataset(Dataset):
    """
    Dataset cho T-GCN:
      - X: [T_all, N, F], Y: [T_all, N], days: [T_all], splits: [T_all] (str)
      - với mỗi t >= L_HISTORY-1, tạo sample nếu splits[t] = split_name:
          x_seq = X[t-L+1 : t+1]
          y     = Y[t + horizon]
    """
    def __init__(self, X, Y, days, splits, horizon, L, split_name):
        super().__init__()
        self.samples = []
        self.L = L
        T_all = X.shape[0]

        for t in range(L - 1, T_all - horizon):
            if splits[t] != split_name:
                continue
            x_seq = X[t - L + 1 : t + 1]  # [L, N, F]
            y_t = Y[t + horizon]          # [N]
            if torch.all(torch.isnan(y_t)):
                continue
            day_val = days[t + horizon]
            self.samples.append((x_seq, y_t, day_val))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x_seq, y, day_val = self.samples[idx]
        return {
            "x_seq": x_seq,
            "y": y,
            "day": day_val,
        }


def build_tgcn_datasets(edge_types: list[str], horizon: int):
    """
    Build dataset T-GCN cho 1 horizon, với multi-view edge_types.
    Edge multi-view được union như train_gnn_ablation.
    """
    pkg = torch.load(GNN_DIR / "gnn_data_encoded.pt", weights_only=False)
    X = pkg["X"].clone()   # [T, N, F]
    if horizon == 1:
        Y = pkg["Y_h1"]
    elif horizon == 7:
        Y = pkg["Y_h7"]
    else:
        raise ValueError(f"Unsupported horizon: {horizon}")

    days = pkg["days"]           # [T]
    splits = pkg["split"]        # np.array[str] shape [T]

    # normalize feature theo train days (giống train_gnn)
    train_mask_t = (splits == "train")
    X_train = X[train_mask_t]   # [T_train, N, F]
    X_train_flat = X_train.reshape(-1, X.shape[-1])

    mask = ~torch.isnan(X_train_flat)
    counts = mask.sum(dim=0).clamp(min=1)
    X_zero = torch.where(mask, X_train_flat, torch.zeros_like(X_train_flat))

    mean = X_zero.sum(dim=0) / counts
    diff = torch.where(mask, X_train_flat - mean, torch.zeros_like(X_train_flat))
    var = (diff * diff).sum(dim=0) / counts
    std = torch.sqrt(var) + 1e-6

    X = (X - mean) / std
# sau khi X = (X - mean) / std
    X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    print("[DEBUG] X stats after norm+nan_to_num:",
      float(torch.isnan(X).sum()), "NaNs")
    print("[DEBUG] Y stats:", float(torch.isnan(Y).sum()), "NaNs total")
    # load edges multi-view và union
    edges_path = GNN_DIR / "edge_index_views.pt"
    if not edges_path.exists():
        raise FileNotFoundError(
            f"{edges_path} not found. Run build_edge_index.py first."
        )
    edges_all = torch.load(edges_path)

    edge_list = []
    for et in edge_types:
        if et not in edges_all:
            raise KeyError(f"edge_type '{et}' not found in edge_index_views.pt")
        edge_list.append(edges_all[et].long())

    edge_cat = torch.cat(edge_list, dim=1)
    edge_index = torch.unique(edge_cat, dim=1)

    T_all, N, Fdim = X.shape
    print(
        f"[DEBUG] T-GCN multi-view {edge_types}, "
        f"T={T_all}, N={N}, F={Fdim}, "
        f"edge_min={int(edge_index.min())}, edge_max={int(edge_index.max())}, "
        f"edges={edge_index.size(1)}"
    )
    assert int(edge_index.max()) < N, "edge_index has node index out of range for X"

    # Tạo dataset train/val/test
    ds_train = TGCNDataset(X, Y, days, splits, horizon=horizon, L=L_HISTORY, split_name="train")
    ds_val   = TGCNDataset(X, Y, days, splits, horizon=horizon, L=L_HISTORY, split_name="val")
    ds_test  = TGCNDataset(X, Y, days, splits, horizon=horizon, L=L_HISTORY, split_name="test")

    return ds_train, ds_val, ds_test, edge_index


def train_epoch_tgcn(model, loader, edge_index, optimizer):
    model.train()
    total_loss, count = 0.0, 0
    edge_index = edge_index.to(DEVICE)

    for batch in loader:
        x_seq = batch["x_seq"].to(DEVICE)  # [B, L, N, F]
        y = batch["y"].to(DEVICE)          # [B, N] hoặc [N]

        # ép batch_size=1 -> [L, N, F] và [N]
        if x_seq.dim() == 4:
            x_seq = x_seq.squeeze(0)
        if y.dim() == 2:
            y = y.squeeze(0)

        optimizer.zero_grad()
        y_hat = model(x_seq, edge_index)   # [N]
        mask = ~torch.isnan(y)
        if mask.sum() == 0:
            continue
        loss = F.mse_loss(y_hat[mask], y[mask])
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        count += 1

    return total_loss / max(count, 1)

@torch.no_grad()
@torch.no_grad()
def eval_epoch_tgcn(model, loader, edge_index):
    model.eval()
    edge_index = edge_index.to(DEVICE)
    all_true, all_pred = [], []

    for batch in loader:
        x_seq = batch["x_seq"].to(DEVICE)  # [B,L,N,F]
        y = batch["y"].to(DEVICE)          # [B,N] or [N]

        if x_seq.dim() == 4:
            x_seq = x_seq.squeeze(0)
        if y.dim() == 2:
            y = y.squeeze(0)

        y_hat = model(x_seq, edge_index)
        mask = ~torch.isnan(y)
        y_true = y[mask]
        y_pred = y_hat[mask]
        if y_true.numel() == 0:
            continue
        all_true.append(y_true.cpu())
        all_pred.append(y_pred.cpu())

    if not all_true:
        return float("nan"), float("nan")
    y_true = torch.cat(all_true)
    y_pred = torch.cat(all_pred)
    mae = torch.mean(torch.abs(y_pred - y_true)).item()
    rmse = torch.sqrt(torch.mean((y_pred - y_true) ** 2)).item()
    return mae, rmse

@torch.no_grad()
def predict_and_save_tgcn(model, loader, edge_index, horizon: int, variant_tag: str):
    model.eval()
    edge_index = edge_index.to(DEVICE)
    rows = []

    for batch in loader:
        x_seq = batch["x_seq"].to(DEVICE)   # [B,L,N,F]
        y = batch["y"]                      # [B,N] or [N]
        day_val = int(batch["day"].item())

        if x_seq.dim() == 4:
            x_seq = x_seq.squeeze(0)
        if y.dim() == 2:
            y = y.squeeze(0)

        y_hat = model(x_seq, edge_index).cpu()  # [N]

        y_true = y.numpy()
        y_pred = y_hat.numpy()
        for node_idx, (yt, yp) in enumerate(zip(y_true, y_pred)):
            rows.append(
                {
                    "day": day_val,
                    "node_index": node_idx,
                    "y_true": float(yt),
                    "y_pred": float(yp),
                }
            )

    df_pred = pd.DataFrame(rows)
    out_dir = PROC_DIR / "predictions_tgcn" / variant_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"tgcn_h{horizon}_test_predictions.csv"
    df_pred.to_csv(out_path, index=False)
    print(f"Saved T-GCN predictions to {out_path}")


def plot_history(history, tag: str, horizon: int):
    out_dir = PROC_DIR / "predictions_tgcn" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    epochs = history["epoch"]
    train_loss = history["train_loss"]
    val_mae = history["val_mae"]
    val_rmse = history["val_rmse"]

    # train loss
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_loss, label="train_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(f"{tag} H{horizon} - train loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{tag}_H{horizon}_train_loss.png")
    plt.close()

    # val metrics
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, val_mae, label="val_mae")
    plt.plot(epochs, val_rmse, label="val_rmse")
    plt.xlabel("epoch")
    plt.ylabel("metric")
    plt.title(f"{tag} H{horizon} - val metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{tag}_H{horizon}_val_metrics.png")
    plt.close()


def train_one_tgcn_variant(horizon: int, edge_types: list[str], tag: str):
    ds_train, ds_val, ds_test, edge_index = build_tgcn_datasets(edge_types, horizon=horizon)

    train_loader = DataLoader(ds_train, batch_size=1, shuffle=True)
    val_loader   = DataLoader(ds_val, batch_size=1, shuffle=False)
    test_loader  = DataLoader(ds_test, batch_size=1, shuffle=False)

    # lấy in_channels từ 1 sample train
    sample = ds_train[0]
    in_channels = sample["x_seq"].shape[-1]

    model = TGCN(
        in_channels=in_channels,
        gcn_hidden=64,
        gru_hidden=64,
        horizon=horizon,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    best_rmse, best_state = float("inf"), None
    patience, wait = 20, 0
    max_epochs = 400

    history = {"epoch": [], "train_loss": [], "val_mae": [], "val_rmse": []}

    for epoch in range(1, max_epochs + 1):
        train_loss = train_epoch_tgcn(model, train_loader, edge_index, optimizer)
        val_mae, val_rmse = eval_epoch_tgcn(model, val_loader, edge_index)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_mae"].append(val_mae)
        history["val_rmse"].append(val_rmse)

        improved = val_rmse < best_rmse - 1e-3
        if improved:
            best_rmse = val_rmse
            best_state = model.state_dict()
            wait = 0
        else:
            wait += 1

        if epoch % 20 == 0 or improved:
            print(
                f"[T-GCN][H{horizon}][{tag}] Epoch {epoch} "
                f"train_loss={train_loss:.4f} val_mae={val_mae:.4f} val_rmse={val_rmse:.4f}"
            )

        if wait >= patience:
            print(
                f"[T-GCN][H{horizon}][{tag}] Early stopping at epoch {epoch}, "
                f"best_val_rmse={best_rmse:.4f}"
            )
            break

    plot_history(history, tag=tag, horizon=horizon)

    if best_state is not None:
        model.load_state_dict(best_state)

    # eval trên test
    mae_test, rmse_test = eval_epoch_tgcn(model, test_loader, edge_index)
    print(f"[T-GCN][H{horizon}][{tag}] Test MAE={mae_test:.4f} RMSE={rmse_test:.4f}")
    predict_and_save_tgcn(model, test_loader, edge_index, horizon=horizon, variant_tag=tag)


def summarize_tgcn_predictions():
    base_dir = PROC_DIR / "predictions_tgcn"
    rows = []

    for csv_path in base_dir.rglob("*.csv"):
        name = csv_path.name
        if "summary" in name:
            continue

        # path: predictions_tgcn/{tag}/tgcn_h{h}_test_predictions.csv
        parts = csv_path.relative_to(base_dir).parts
        tag = parts[0] if len(parts) > 1 else None

        h_match = re.search(r"h(\d+)", name)
        horizon = int(h_match.group(1)) if h_match else None

        df = pd.read_csv(csv_path)
        if not {"y_true", "y_pred"}.issubset(df.columns):
            print(f"Skip {csv_path}, missing y_true/y_pred")
            continue

        mae = mean_absolute_error(df["y_true"], df["y_pred"])
        rmse = root_mean_squared_error(df["y_true"], df["y_pred"])

        rows.append(
            {
                "model_type": "tgcn",
                "tag": tag,
                "horizon": horizon,
                "n_samples": len(df),
                "MAE": mae,
                "RMSE": rmse,
                "file": str(csv_path.relative_to(PROC_DIR)),
            }
        )

    if not rows:
        print("No T-GCN prediction files found.")
        return

    df_sum = pd.DataFrame(rows)
    df_sum = df_sum.sort_values(["horizon", "tag"])
    out_path = PROC_DIR / "predictions_tgcn" / "summary_tgcn.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_sum.to_csv(out_path, index=False)
    print(f"Saved T-GCN summary to {out_path}")


def main():
    # Ablation settings giống GNN/XGB baseline_2
    tgcn_settings = {
        "TGCN_plant": ["plant"],
        "TGCN_group": ["product_group"],
        "TGCN_subgroup": ["sub_group"],
        "TGCN_storage": ["storage_location"],
        "TGCN_plant_group": ["plant", "product_group"],
        "TGCN_plant_subgroup": ["plant", "sub_group"],
        "TGCN_plant_storage": ["plant", "storage_location"],
        "TGCN_all4": ["plant", "product_group", "sub_group", "storage_location"],
    }

    for h in [1, 7]:
        for tag, ets in tgcn_settings.items():
            train_one_tgcn_variant(horizon=h, edge_types=ets, tag=tag)

    summarize_tgcn_predictions()


if __name__ == "__main__":
    main()
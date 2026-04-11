# train_gnn.py
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from config.config import PROC_DIR
from gnn_model import GCNNodeRegressor, GINNodeRegressor

GNN_DIR = PROC_DIR / "gnn"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_dataset(horizon: int, edge_type: str = "plant"):
    # dùng file encoded chung
    pkg = torch.load(GNN_DIR / "gnn_data_encoded.pt", weights_only=False)

    X = pkg["X"].clone()           # [T, N, F]
    if horizon == 1:
        Y = pkg["Y_h1"]           # [T, N]
    elif horizon == 7:
        Y = pkg["Y_h7"]
    else:
        raise ValueError(f"Unsupported horizon: {horizon}")

    days = pkg["days"]             # [T]
    splits = pkg["split"]          # np.array[str] shape [T]
    feature_cols = pkg["feature_cols"]

    # --- normalize feature theo train days ---
    train_mask_t = (splits == "train")
    X_train = X[train_mask_t]      # [T_train, N, F]
    X_train_flat = X_train.reshape(-1, X.shape[-1])
    mask = ~torch.isnan(X_train_flat)
    counts = mask.sum(dim=0).clamp(min=1)
    X_zero = torch.where(mask, X_train_flat, torch.zeros_like(X_train_flat))

    mean = X_zero.sum(dim=0) / counts

    diff = torch.where(mask, X_train_flat - mean, torch.zeros_like(X_train_flat))
    var = (diff * diff).sum(dim=0) / counts
    std = torch.sqrt(var) + 1e-6

    X = (X - mean) / std

    # --- load & chỉnh edge_index cho khớp với N ---
    edges_path = GNN_DIR / "edge_index_views.pt"
    if not edges_path.exists():
        raise FileNotFoundError(
            f"{edges_path} not found. Run build_edge_index_views.py first."
        )
    edges = torch.load(edges_path)
    edge_index_full = edges[edge_type].long()   # [2, E]

    T, N, Fdim = X.shape

    # 1) shift về 0-based nếu min_idx > 0
    min_idx = int(edge_index_full.min())
    edge_index_shifted = edge_index_full - min_idx

    # 2) giữ cạnh có index trong [0, N-1]
    mask_valid = (edge_index_shifted >= 0) & (edge_index_shifted < N)
    mask_valid = mask_valid.all(dim=0)          # [E]
    edge_index = edge_index_shifted[:, mask_valid]

    print(
        f"[DEBUG] edge_type={edge_type} "
        f"N={N}, "
        f"orig_min={int(edge_index_full.min())}, orig_max={int(edge_index_full.max())}, "
        f"edges_before={edge_index_full.size(1)}, edges_after={edge_index.size(1)}"
    )

    data_train, data_val, data_test = [], [], []

    for t in range(T):
        x_t = X[t]          # [N, F]
        y_t = Y[t]          # [N]

        # đảm bảo không có NaN trong x
        x_t = torch.nan_to_num(x_t, nan=0.0, posinf=0.0, neginf=0.0)

        # giữ NaN trong y để mask khi tính loss
        if torch.all(torch.isnan(y_t)):
            continue

        data = Data(x=x_t, edge_index=edge_index, y=y_t)
        data.day = days[t]
        split = splits[t]
        if split == "train":
            data_train.append(data)
        elif split == "val":
            data_val.append(data)
        else:
            data_test.append(data)

    return data_train, data_val, data_test, feature_cols


def train_epoch(model, loader, optimizer):
    model.train()
    total_loss, count = 0.0, 0
    for data in loader:
        data = data.to(DEVICE)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)   # [N]
        mask = ~torch.isnan(data.y)
        if mask.sum() == 0:
            continue
        loss = F.mse_loss(out[mask], data.y[mask])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        count += 1
    return total_loss / max(count, 1)


@torch.no_grad()
def eval_epoch(model, loader):
    model.eval()
    all_true, all_pred = [], []
    for data in loader:
        data = data.to(DEVICE)
        out = model(data.x, data.edge_index)
        mask = ~torch.isnan(data.y)
        y_true = data.y[mask]
        y_pred = out[mask]
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
def predict_and_save(model, loader, horizon: int, variant_tag: str):
    model.eval()
    rows = []
    for data in loader:
        data = data.to(DEVICE)
        out = model(data.x, data.edge_index)
        y_true = data.y.cpu().numpy()
        y_pred = out.cpu().numpy()
        day_val = int(data.day.item())
        for node_idx, (yt, yp) in enumerate(zip(y_true, y_pred)):
            rows.append(
                {"day": day_val, "node_index": node_idx, "y_true": float(yt), "y_pred": float(yp)}
            )
    df_pred = pd.DataFrame(rows)
    out_dir = PROC_DIR / "predictions_gnn" / variant_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    df_pred.to_csv(out_dir / f"gnn_h{horizon}_test_predictions.csv", index=False)
    print(f"Saved predictions to {out_dir}/gnn_h{horizon}_test_predictions.csv")

def plot_history(history, tag: str, horizon: int):
    """
    history: dict với key 'epoch', 'train_loss', 'val_mae', 'val_rmse'
    """
    out_dir = PROC_DIR / "predictions_gnn" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    epochs = history["epoch"]
    train_loss = history["train_loss"]
    val_mae = history["val_mae"]
    val_rmse = history["val_rmse"]

    # Plot train loss
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_loss, label="train_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(f"{tag} H{horizon} - train loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{tag}_H{horizon}_train_loss.png")
    plt.close()

    # Plot val MAE & RMSE
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

def train_one_model(h: int, edge_type: str, model_cls, tag: str):
    train_set, val_set, test_set, feature_cols = build_dataset(h, edge_type=edge_type)

    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    in_channels = train_set[0].x.shape[1]

    model = model_cls(in_channels=in_channels, hidden_channels=64, num_layers=3).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    best_rmse, best_state = float("inf"), None
    patience, wait = 20, 0
    max_epochs = 300

    history = {"epoch": [], "train_loss": [], "val_mae": [], "val_rmse": []}

    for epoch in range(1, max_epochs + 1):
        train_loss = train_epoch(model, train_loader, opt)
        val_mae, val_rmse = eval_epoch(model, val_loader)

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
                f"[H{h}][{tag}] Epoch {epoch} "
                f"train_loss={train_loss:.4f} val_mae={val_mae:.4f} val_rmse={val_rmse:.4f}"
            )

        if wait >= patience:
            print(f"[H{h}][{tag}] Early stopping at epoch {epoch}, best_val_rmse={best_rmse:.4f}")
            break

    # vẽ curve
    plot_history(history, tag=tag, horizon=h)

    if best_state is not None:
        model.load_state_dict(best_state)
    test_mae, test_rmse = eval_epoch(model, test_loader)
    print(f"[H{h}][{tag}] Test MAE={test_mae:.4f} RMSE={test_rmse:.4f}")
    predict_and_save(model, test_loader, h, variant_tag=tag)

def main():
    for h in [1, 7]:
        # GCN
        train_one_model(h, edge_type="plant", model_cls=GCNNodeRegressor, tag="GCN_plant")
        # GIN
        train_one_model(h, edge_type="plant", model_cls=GINNodeRegressor, tag="GIN_plant")


if __name__ == "__main__":
    main()
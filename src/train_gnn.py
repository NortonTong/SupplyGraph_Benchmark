# train_gnn.py
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROC_DIR = DATA_DIR / "processed"
GNN_DIR = PROC_DIR / "gnn"

from gnn_model import GCNNodeRegressor, GINNodeRegressor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_dataset(horizon: int, edge_type: str = "plant"):
    # dùng file encoded chung
    pkg = torch.load(GNN_DIR / "gnn_data_encoded.pt", weights_only = False)

    X = pkg["X"]           # [T, N, F]
    if horizon == 1:
        Y = pkg["Y_h1"]    # [T, N]
    elif horizon == 7:
        Y = pkg["Y_h7"]
    else:
        raise ValueError(f"Unsupported horizon: {horizon}")

    days = pkg["days"]     # [T]
    splits = pkg["split"]  # np.array[str] shape [T]
    feature_cols = pkg["feature_cols"]

    # nếu bạn có edge_index_views.pt (graph theo node_index) thì load,
    # nếu chưa làm graph cho GNN thì bạn có thể dùng fully-connected hoặc adjacency riêng.
    edges_path = GNN_DIR / "edge_index_views.pt"
    if edges_path.exists():
        edges = torch.load(edges_path)
        edge_index = edges[edge_type]    # [2, E]
    else:
        edge_index = None  # bạn có thể tự build sau

    T, N, Fdim = X.shape
    data_train, data_val, data_test = [], [], []

    for t in range(T):
        x_t = X[t]      # [N, F]
        y_t = Y[t]      # [N]
        mask = ~torch.isnan(y_t)
        if mask.sum() == 0:
            continue

        # nếu chưa có graph, chỉ cần set edge_index=None,
        # nhưng GCN/GIN cần edge_index, nên bạn nên build graph trước.
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
        loss = F.mse_loss(out[mask], data.y[mask])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        count += 1
    return total_loss / max(count, 1)


@torch.no_grad()
def eval_epoch(model, loader):
    model.eval()
    mae_list, rmse_list = [], []
    for data in loader:
        data = data.to(DEVICE)
        out = model(data.x, data.edge_index)
        mask = ~torch.isnan(data.y)
        y_true = data.y[mask]
        y_pred = out[mask]
        if y_true.numel() == 0:
            continue
        mae = torch.mean(torch.abs(y_pred - y_true)).item()
        rmse = torch.sqrt(torch.mean((y_pred - y_true) ** 2)).item()
        mae_list.append(mae)
        rmse_list.append(rmse)
    mae = float(np.mean(mae_list)) if mae_list else float("nan")
    rmse = float(np.mean(rmse_list)) if rmse_list else float("nan")
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


def main():
    for h in [1, 7]:
        train_set, val_set, test_set, feature_cols = build_dataset(h, edge_type="plant")

        train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

        in_channels = train_set[0].x.shape[1]

        # --- GCN ---
        gcn = GCNNodeRegressor(in_channels=in_channels, hidden_channels=64, num_layers=3).to(DEVICE)
        opt = torch.optim.Adam(gcn.parameters(), lr=1e-3, weight_decay=1e-5)
        best_rmse, best_state = float("inf"), None

        for epoch in range(1, 101):
            train_loss = train_epoch(gcn, train_loader, opt)
            val_mae, val_rmse = eval_epoch(gcn, val_loader)
            if val_rmse < best_rmse:
                best_rmse = val_rmse
                best_state = gcn.state_dict()
            if epoch % 10 == 0:
                print(f"[H{h}] GCN Epoch {epoch} train_loss={train_loss:.4f} val_mae={val_mae:.4f} val_rmse={val_rmse:.4f}")

        if best_state is not None:
            gcn.load_state_dict(best_state)
        test_mae, test_rmse = eval_epoch(gcn, test_loader)
        print(f"[H{h}] GCN Test MAE={test_mae:.4f} RMSE={test_rmse:.4f}")
        predict_and_save(gcn, test_loader, h, variant_tag="GCN_plant")

        # --- GIN ---
        gin = GINNodeRegressor(in_channels=in_channels, hidden_channels=64, num_layers=3).to(DEVICE)
        opt = torch.optim.Adam(gin.parameters(), lr=1e-3, weight_decay=1e-5)
        best_rmse, best_state = float("inf"), None

        for epoch in range(1, 101):
            train_loss = train_epoch(gin, train_loader, opt)
            val_mae, val_rmse = eval_epoch(gin, val_loader)
            if val_rmse < best_rmse:
                best_rmse = val_rmse
                best_state = gin.state_dict()
            if epoch % 10 == 0:
                print(f"[H{h}] GIN Epoch {epoch} train_loss={train_loss:.4f} val_mae={val_mae:.4f} val_rmse={val_rmse:.4f}")

        if best_state is not None:
            gin.load_state_dict(best_state)
        test_mae, test_rmse = eval_epoch(gin, test_loader)
        print(f"[H{h}] GIN Test MAE={test_mae:.4f} RMSE={test_rmse:.4f}")
        predict_and_save(gin, test_loader, h, variant_tag="GIN_plant")


if __name__ == "__main__":
    main()
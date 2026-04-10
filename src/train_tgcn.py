import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
from pathlib import Path
import numpy as np
import pandas as pd

from tgcn_model import TGCN, MultiViewTGCN

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROC_DIR = DATA_DIR / "processed"
GNN_DIR = PROC_DIR / "gnn"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TGCNSeqDataset(Dataset):
    def __init__(self, X, Y, splits, seq_len, pre_len, split_name):
        self.X = X
        self.Y = Y
        self.seq_len = seq_len
        self.pre_len = pre_len
        self.split_name = split_name

        T = X.shape[0]
        idx_list = []
        for t in range(T - seq_len - pre_len + 1):
            tgt_day_idx = t + seq_len + pre_len - 1
            if splits[tgt_day_idx] == split_name:
                idx_list.append(t)
        self.start_indices = idx_list

    def __len__(self):
        return len(self.start_indices)

    def __getitem__(self, idx):
        t0 = self.start_indices[idx]
        t1 = t0 + self.seq_len
        t2 = t1 + self.pre_len
        x = self.X[t0:t1]
        y = self.Y[t1:t2]
        return x, y

def load_base_pkg():
    pkg = torch.load(GNN_DIR / "gnn_data_encoded.pt", weights_only=False)
    return pkg


def build_tgcn_datasets(horizon: int, seq_len: int, pre_len: int):
    pkg = torch.load(GNN_DIR / "gnn_data_encoded.pt", weights_only=False)
    X = pkg["X"]
    if horizon == 1:
        Y = pkg["Y_h1"]
    elif horizon == 7:
        Y = pkg["Y_h7"]
    else:
        raise ValueError(f"Unsupported horizon: {horizon}")

    splits = pkg["split"]
    feature_cols = pkg["feature_cols"]

    # load tất cả view
    edges = torch.load(GNN_DIR / "edge_index_views.pt", weights_only=False)
    edge_index_dict = {
        "plant": edges["plant"],
        "product_group": edges["product_group"],
        "sub_group": edges["sub_group"],
        "storage_location": edges["storage_location"],
    }

    X = X.float()
    Y = Y.float()

    train_ds = TGCNSeqDataset(X, Y, splits, seq_len, pre_len, split_name="train")
    val_ds   = TGCNSeqDataset(X, Y, splits, seq_len, pre_len, split_name="val")
    test_ds  = TGCNSeqDataset(X, Y, splits, seq_len, pre_len, split_name="test")
    print(
        f"[TGCN H{horizon}] seq_len={seq_len}, pre_len={pre_len} | "
        f"train_samples={len(train_ds)}, val_samples={len(val_ds)}, test_samples={len(test_ds)}"
    )
    return train_ds, val_ds, test_ds, edge_index_dict, feature_cols, pkg
# tiếp trong train_tgcn.py
def train_epoch_tgcn(model, loader, edge_index_dict, optimizer):
    model.train()
    total_loss, count = 0.0, 0
    for X_seq, Y_seq in loader:
        # X_seq: [B, L, N, F], Y_seq: [B, pre_len, N]
        X_seq = X_seq.to(DEVICE)
        Y_seq = Y_seq.to(DEVICE)
        optimizer.zero_grad()
        out = model(X_seq, edge_index_dict)      # [B, pre_len, N]
        mask = ~torch.isnan(Y_seq)
        loss = torch.mean((out[mask] - Y_seq[mask]) ** 2)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        count += 1
    return total_loss / max(count, 1)


@torch.no_grad()
def eval_epoch_tgcn(model, loader, edge_index_dict):
    model.eval()
    mae_list, rmse_list = [], []
    for X_seq, Y_seq in loader:
        X_seq = X_seq.to(DEVICE)
        Y_seq = Y_seq.to(DEVICE)
        out = model(X_seq, edge_index_dict)      # [B, pre_len, N]
        mask = ~torch.isnan(Y_seq)
        if mask.sum() == 0:
            continue
        y_true = Y_seq[mask]
        y_pred = out[mask]
        mae = torch.mean(torch.abs(y_pred - y_true)).item()
        rmse = torch.sqrt(torch.mean((y_pred - y_true) ** 2)).item()
        mae_list.append(mae)
        rmse_list.append(rmse)
    mae = float(np.mean(mae_list)) if mae_list else float("nan")
    rmse = float(np.mean(rmse_list)) if rmse_list else float("nan")
    return mae, rmse


@torch.no_grad()
def predict_and_save_tgcn(model, loader, edge_index, horizon, variant_tag, pkg, seq_len, pre_len):
    """
    Ghi lại prediction giống format cũ: day, node_index, y_true, y_pred cho ngày test.
    Với pre_len>1, ở đây demo lấy đúng bước horizon tương ứng:
    - horizon=1: lấy bước 0
    - horizon=7: lấy bước 6 (giả sử pre_len>=7)
    """
    model.eval()
    days = pkg["days"].numpy()      # [T]
    node_index_arr = pkg["node_index"].numpy()  # [N]
    splits = pkg["split"]           # [T]

    rows = []
    step_idx = 0 if horizon == 1 else (pre_len - 1)

    # lặp lại logic index để map từ sequence -> day
    T = pkg["X"].shape[0]
    start_indices = []
    for t in range(T - seq_len - pre_len + 1):
        tgt_day_idx = t + seq_len + pre_len - 1
        if splits[tgt_day_idx] == "test":
            start_indices.append(t)

    for (X_seq, Y_seq), t0 in zip(loader, start_indices):
        X_seq = X_seq.to(DEVICE)
        Y_seq = Y_seq.to(DEVICE)
        out = model(X_seq, edge_index)          # [B, pre_len, N]
        out = out.cpu().numpy()
        Y_seq = Y_seq.cpu().numpy()

        B, P, N = out.shape
        assert B == 1
        t_target = t0 + seq_len + step_idx
        day_val = int(days[t_target])

        y_pred = out[0, step_idx]   # [N]
        y_true = Y_seq[0, step_idx] # [N]

        for pos in range(N):
            rows.append(
                {
                    "day": day_val,
                    "node_index": int(node_index_arr[pos]),
                    "y_true": float(y_true[pos]),
                    "y_pred": float(y_pred[pos]),
                }
            )

    df_pred = pd.DataFrame(rows)
    out_dir = PROC_DIR / "predictions_gnn" / variant_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    df_pred.to_csv(out_dir / f"tgcn_h{horizon}_test_predictions.csv", index=False)



def main():
    seq_len = 14
    for h in [1, 7]:
        pre_len = 1 if h == 1 else 7

        train_ds, val_ds, test_ds, edge_index_dict, feature_cols, pkg = build_tgcn_datasets(
            horizon=h,
            seq_len=seq_len,
            pre_len=pre_len,
        )

        # logging số sample per split
        print(
            f"[TGCN-MV H{h}] #train={len(train_ds)}, #val={len(val_ds)}, #test={len(test_ds)}"
        )

        train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
        val_loader   = DataLoader(val_ds, batch_size=8, shuffle=False)
        test_loader  = DataLoader(test_ds, batch_size=1, shuffle=False)

        T, N, Fdim = pkg["X"].shape
        edge_index_dict_dev = {k: v.to(DEVICE) for k, v in edge_index_dict.items()}

        model = MultiViewTGCN(
            num_nodes=N,
            in_channels=Fdim,
            hidden_channels=32,
            gcn_layers=1,
            pre_len=pre_len,
            views=("plant", "product_group", "sub_group", "storage_location"),
            fusion="concat",
        ).to(DEVICE)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

        best_rmse, best_state = float("inf"), None
        for epoch in range(1, 400):
            train_loss = train_epoch_tgcn(model, train_loader, edge_index_dict_dev, optimizer)
            val_mae, val_rmse = eval_epoch_tgcn(model, val_loader, edge_index_dict_dev)
            if val_rmse < best_rmse:
                best_rmse = val_rmse
                best_state = model.state_dict()
            if epoch % 10 == 0:
                print(f"[TGCN-MV H{h}] Epoch {epoch} loss={train_loss:.4f} val_mae={val_mae:.4f} val_rmse={val_rmse:.4f}")

        if best_state is not None:
            model.load_state_dict(best_state)
        test_mae, test_rmse = eval_epoch_tgcn(model, test_loader, edge_index_dict_dev)
        print(f"[TGCN-MV H{h}] Test MAE={test_mae:.4f} RMSE={test_rmse:.4f}")

        predict_and_save_tgcn(
            model,
            test_loader,
            edge_index_dict_dev,
            horizon=h,
            variant_tag="TGCN_multi_view",
            pkg=pkg,
            seq_len=seq_len,
            pre_len=pre_len,
        )

if __name__ == "__main__":
    main()
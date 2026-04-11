import pandas as pd
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from config.config import (PROJECT_ROOT, DATA_DIR, RAW_DIR, NODE_DIR, EDGE_DIR, PROC_DIR)
from torch_geometric.nn import GINConv
import torch.nn.functional as F


import matplotlib.pyplot as plt

def plot_gnn_losses(
    train_losses: list[float],
    val_losses: list[float],
    edge_type: str,
    model_name: str,
    out_dir: Path,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(train_losses, label="Train loss")
    plt.plot(val_losses, label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title(f"{model_name} loss - {edge_type}")
    plt.legend()
    plt.tight_layout()
    out_path = out_dir / f"{model_name}_loss_{edge_type}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {model_name} loss plot for {edge_type} to {out_path}")
# ----------------- GCN encoder -----------------

class CategoricalGCNEncoder(nn.Module):
    def __init__(
        self,
        num_categories_per_field,
        emb_dim=16,
        hidden_dim=64,
        out_dim=32,
        dropout=0.1,
    ):
        super().__init__()
        self.emb_layers = nn.ModuleList([
            nn.Embedding(n_cat, emb_dim) for n_cat in num_categories_per_field
        ])
        in_dim = emb_dim * len(num_categories_per_field)
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(out_dim)

    def forward(self, x_cat, edge_index):
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.emb_layers)]
        x = torch.cat(embs, dim=-1)        # [num_nodes, in_dim]
        x = self.conv1(x, edge_index)
        x = self.act(x)
        x = self.norm1(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        return x                            # [num_nodes, out_dim]

class CategoricalGINEncoder(nn.Module):
    def __init__(
        self,
        num_categories_per_field,
        emb_dim=16,
        hidden_dim=64,
        out_dim=32,
        dropout=0.1,
    ):
        super().__init__()
        self.emb_layers = nn.ModuleList([
            nn.Embedding(n_cat, emb_dim) for n_cat in num_categories_per_field
        ])
        in_dim = emb_dim * len(num_categories_per_field)

        nn1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        nn2 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )
        self.conv1 = GINConv(nn1)
        self.conv2 = GINConv(nn2)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(out_dim)

    def forward(self, x_cat, edge_index):
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.emb_layers)]
        x = torch.cat(embs, dim=-1)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.norm1(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        return x
# ----------------- Helper: load node metadata & features -----------------


def load_node_regression_targets(horizon: int = 1) -> pd.DataFrame:
    """
    Tạo target node-level: ví dụ trung bình y_h{horizon} theo node_id trên train.
    """
    base_path = PROC_DIR / "xgboost_base_filtered.parquet"
    df_base = pd.read_parquet(base_path)
    df_train = df_base[df_base["split"] == "train"].copy()

    label_col = f"y_h{horizon}"
    agg = (
        df_train.groupby("node_id")[label_col]
        .mean()
        .reset_index()
        .rename(columns={label_col: "target_reg"})
    )
    return agg

def load_node_metadata_for_gnn() -> pd.DataFrame:
    """
    Load node metadata and return df with:
      node_id, node_index, group_code, sub_group_code, plant_code, storage_code
    """
    node_index_path = NODE_DIR / "NodesIndex.csv"
    node_group_path = NODE_DIR / "Node Types (Product Group and Subgroup).csv"
    node_plant_storage_path = NODE_DIR / "Nodes Type (Plant & Storage).csv"

    df_index = pd.read_csv(node_index_path)
    df_index = df_index.rename(columns={"Node": "node_id", "NodeIndex": "node_index"})

    df_group = pd.read_csv(node_group_path)
    df_group = df_group.rename(
        columns={"Node": "node_id", "Group": "group", "Sub-Group": "sub_group"}
    )

    df_plant = pd.read_csv(node_plant_storage_path)
    df_plant = df_plant.rename(
        columns={
            "Node": "node_id",
            "Plant": "plant",
            "Storage Location": "storage_location",
        }
    )

    df_meta = df_index.merge(df_group, on="node_id", how="left")
    df_meta = df_meta.merge(df_plant, on="node_id", how="left")

    for col in ["group", "sub_group", "plant", "storage_location"]:
        df_meta[col] = df_meta[col].astype("category")
        df_meta[f"{col}_code"] = df_meta[col].cat.codes
        # cat.codes dùng -1 cho NaN → chuyển thành max+1 để không âm
        max_code = df_meta[f"{col}_code"].max()
        df_meta[f"{col}_code"] = df_meta[f"{col}_code"].replace(-1, max_code + 1)

    return df_meta


def build_pyg_data_for_view(df_meta: pd.DataFrame, edge_type: str) -> Data:
    code_cols = [c for c in ["group_code", "sub_group_code", "plant_code", "storage_location_code"]
                 if c in df_meta.columns]
    x_cat = df_meta[code_cols].values
    x_cat = torch.tensor(x_cat, dtype=torch.long)

    # Tạo mapping từ node_index gốc sang 0..N-1
    unique_idx = df_meta["node_index"].values
    idx2pos = {idx: pos for pos, idx in enumerate(unique_idx)}

    # Load edges (node_index-based)
    if edge_type == "plant":
        edge_file = EDGE_DIR / "Edges (Plant).csv"
    elif edge_type == "product_group":
        edge_file = EDGE_DIR / "Edges (Product Group).csv"
    elif edge_type == "sub_group":
        edge_file = EDGE_DIR / "Edges (Product Sub-Group).csv"
    elif edge_type == "storage_location":
        edge_file = EDGE_DIR / "Edges (Storage Location).csv"
    else:
        raise ValueError(f"Unknown edge_type: {edge_type}")

    edges = pd.read_csv(edge_file)


    if {"node1", "node2"}.issubset(edges.columns):
        u_col, v_col = "node1", "node2"
    else:
        raise ValueError(f"Unknown edge file format for {edge_file}")

    # Map node_index trong edge file sang 0..N-1
    u_raw = edges[u_col].map(idx2pos)
    v_raw = edges[v_col].map(idx2pos)

    # Drop edge nào có node không nằm trong df_meta
    mask = u_raw.notna() & v_raw.notna()
    u = torch.tensor(u_raw[mask].astype(int).values, dtype=torch.long)
    v = torch.tensor(v_raw[mask].astype(int).values, dtype=torch.long)
    edge_index = torch.stack([u, v], dim=0)

    data = Data(x=x_cat, edge_index=edge_index)
    return data

# ----------------- Train GCN encoder for 1 view -----------------

def train_gcn_encoder_for_view(
    df_meta: pd.DataFrame,
    edge_type: str,
    emb_dim: int = 16,
    hidden_dim: int = 64,
    out_dim: int = 32,
    lr: float = 1e-3,
    epochs: int = 200,
    patience: int = 20,
    device: str = "cpu",
) -> np.ndarray:
    """
    Train GCN encoder với node-level regression target (target_reg),
    dùng early stopping theo val_loss. Trả về embeddings (num_nodes, out_dim).
    """
    data = build_pyg_data_for_view(df_meta, edge_type).to(device)

    code_cols = [c for c in ["group_code", "sub_group_code", "plant_code", "storage_location_code"]
                 if c in df_meta.columns]
    num_cats = [int(df_meta[c].max()) + 1 for c in code_cols]

    model = CategoricalGCNEncoder(
        num_categories_per_field=num_cats,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
    ).to(device)

    reg_head = nn.Linear(out_dim, 1).to(device)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(reg_head.parameters()), lr=lr)
    criterion = nn.MSELoss()

    y = torch.tensor(df_meta["target_reg"].values, dtype=torch.float32, device=device)

    # ---- split train/val nodes ----
    idx = np.arange(len(df_meta))
    np.random.shuffle(idx)
    split = int(0.8 * len(idx))
    train_idx = torch.tensor(idx[:split], device=device)
    val_idx   = torch.tensor(idx[split:], device=device)

    best_val = float("inf")
    best_state = None
    wait = 0

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        # ---- train ----
        model.train()
        reg_head.train()
        optimizer.zero_grad()
        z = model(data.x, data.edge_index)       # [num_nodes, out_dim]
        y_pred = reg_head(z).squeeze(-1)         # [num_nodes]
        loss = criterion(y_pred[train_idx], y[train_idx])
        loss.backward()
        optimizer.step()

        # ---- validation ----
        model.eval()
        reg_head.eval()
        with torch.no_grad():
            z_val = model(data.x, data.edge_index)
            y_val_pred = reg_head(z_val).squeeze(-1)
            val_loss = criterion(y_val_pred[val_idx], y[val_idx]).item()

        train_losses.append(loss.item())
        val_losses.append(val_loss)

        if (epoch + 1) % 20 == 0:
            print(f"[GCN-{edge_type}] Epoch {epoch+1}/{epochs}, "
                  f"train_loss={loss.item():.6f}, val_loss={val_loss:.6f}")

        # ---- early stopping ----
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {
                "model": model.state_dict(),
                "head": reg_head.state_dict(),
            }
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"[GCN-{edge_type}] Early stopping at epoch {epoch+1}, best_val={best_val:.6f}")
                break

    # load best weights (nếu có)
    if best_state is not None:
        model.load_state_dict(best_state["model"])
        reg_head.load_state_dict(best_state["head"])

    # vẽ loss
    plot_dir = PROC_DIR / "gnn_embeddings" / "loss_plots"
    plot_gnn_losses(train_losses, val_losses, edge_type, "gcn", plot_dir)

    model.eval()
    with torch.no_grad():
        z = model(data.x, data.edge_index)

    return z.cpu().numpy()

def train_gin_encoder_for_view(
    df_meta: pd.DataFrame,
    edge_type: str,
    emb_dim: int = 16,
    hidden_dim: int = 64,
    out_dim: int = 32,
    lr: float = 1e-3,
    epochs: int = 200,
    patience: int = 20,
    device: str = "cpu",
) -> np.ndarray:
    """
    Train GIN encoder với node-level regression target (target_reg),
    dùng early stopping theo val_loss. Trả về embeddings (num_nodes, out_dim).
    """
    data = build_pyg_data_for_view(df_meta, edge_type).to(device)

    code_cols = [c for c in ["group_code","sub_group_code","plant_code","storage_location_code"]
                 if c in df_meta.columns]
    num_cats = [int(df_meta[c].max()) + 1 for c in code_cols]

    model = CategoricalGINEncoder(
        num_categories_per_field=num_cats,
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
    ).to(device)
    reg_head = nn.Linear(out_dim, 1).to(device)

    optimizer = torch.optim.Adam(list(model.parameters()) + list(reg_head.parameters()), lr=lr)
    criterion = nn.MSELoss()

    y = torch.tensor(df_meta["target_reg"].values, dtype=torch.float32, device=device)

    # ---- split train/val nodes ----
    idx = np.arange(len(df_meta))
    np.random.shuffle(idx)
    split = int(0.8 * len(idx))
    train_idx = torch.tensor(idx[:split], device=device)
    val_idx   = torch.tensor(idx[split:], device=device)

    best_val = float("inf")
    best_state = None
    wait = 0

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        # ---- train ----
        model.train()
        reg_head.train()
        optimizer.zero_grad()
        z = model(data.x, data.edge_index)
        y_pred = reg_head(z).squeeze(-1)
        loss = criterion(y_pred[train_idx], y[train_idx])
        loss.backward()
        optimizer.step()

        # ---- validation ----
        model.eval()
        reg_head.eval()
        with torch.no_grad():
            z_val = model(data.x, data.edge_index)
            y_val_pred = reg_head(z_val).squeeze(-1)
            val_loss = criterion(y_val_pred[val_idx], y[val_idx]).item()

        train_losses.append(loss.item())
        val_losses.append(val_loss)

        if (epoch + 1) % 20 == 0:
            print(f"[GIN-{edge_type}] Epoch {epoch+1}/{epochs}, "
                  f"train_loss={loss.item():.6f}, val_loss={val_loss:.6f}")

        # ---- early stopping ----
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {
                "model": model.state_dict(),
                "head": reg_head.state_dict(),
            }
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"[GIN-{edge_type}] Early stopping at epoch {epoch+1}, best_val={best_val:.6f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state["model"])
        reg_head.load_state_dict(best_state["head"])

    plot_dir = PROC_DIR / "gnn_embeddings" / "loss_plots"
    plot_gnn_losses(train_losses, val_losses, edge_type, "gin", plot_dir)

    model.eval()
    with torch.no_grad():
        z = model(data.x, data.edge_index)

    return z.cpu().numpy()
# ----------------- Build embeddings for all views & merge -----------------

def build_gnn_embeddings_all_views(horizon: int = 1) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df_meta = load_node_metadata_for_gnn()
    df_target = load_node_regression_targets(horizon=horizon)
    df_meta = df_meta.merge(df_target, on="node_id", how="left")
    df_meta["target_reg"] = df_meta["target_reg"].fillna(0.0)
    edge_types = ["plant", "product_group", "sub_group", "storage_location"]
    df_emb_all = df_meta[["node_id", "node_index"]].copy()

    for et in edge_types:
        emb_np = train_gcn_encoder_for_view(
            df_meta=df_meta,
            edge_type=et,
            emb_dim=16,
            hidden_dim=64,
            out_dim=32,
            lr=0.01,
            epochs=2000,
            device=device,
        )
        cols = [f"{et}_gcn_emb_{i}" for i in range(emb_np.shape[1])]
        gcn_block = pd.DataFrame(emb_np, columns=cols, index=df_meta.index)
        df_emb_all = pd.concat([df_emb_all, gcn_block], axis=1)

    for et in edge_types:
        emb_np = train_gin_encoder_for_view(
            df_meta=df_meta,
            edge_type=et,
            emb_dim=16,
            hidden_dim=64,
            out_dim=32,
            lr=0.01,
            epochs=2000,
            device=device,
        )
        cols = [f"{et}_gin_emb_{i}" for i in range(emb_np.shape[1])]
        gin_block = pd.DataFrame(emb_np, columns=cols, index=df_meta.index)
        df_emb_all = pd.concat([df_emb_all, gin_block], axis=1)

    df_emb_all = df_emb_all.copy()
    for c in df_emb_all.columns:
        if "_gcn_emb_" in c or "_gin_emb_" in c:
            df_emb_all[c] = df_emb_all[c].astype("float32")

    out_path = PROC_DIR / "gnn_embeddings" / "gnn_node_embeddings_all_views.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_emb_all.to_parquet(out_path, index=False)
# ----------------- Merge embeddings vào XGBoost datasets -----------------


def main():
    # 1) Train GCN encoders and create embeddings for all 4 views
    build_gnn_embeddings_all_views()


if __name__ == "__main__":
    main()
import torch
from torch import nn
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

from models_gnn import (
    ProjectedGINRegressor,
    HomogeneousFiveTypeGINRegressor,
    HeterogeneousGINRegressor,
)
from config.config import PROC_DIR, DEFAULT_EXPERIMENTS


RUN_SUMMARY = []

HORIZON = 7
LAG_WINDOWS = [7, 14]
PROJECTED_VIEWS = ["same_group", "same_subgroup", "same_plant", "same_storage"]


# =========================
# EarlyStopping helper
# =========================

class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.best_state_dict = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_state_dict = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            self.counter = 0
            return False

        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.best_state_dict = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
            return True
        return False

    def load_best(self, model: nn.Module):
        if self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)


# =========================
# Helper chung
# =========================

def load_gnn_pkg(graph_type: str, temporal_type: str, lag_window: int, horizon: int = 7):
    if graph_type == "projected":
        name = f"gnn_projected_h{horizon}_lag{lag_window}_{temporal_type}.pt"
    elif graph_type == "homo5":
        name = f"gnn_homo5_h{horizon}_lag{lag_window}_{temporal_type}.pt"
    elif graph_type == "hetero5":
        name = f"gnn_hetero5_h{horizon}_lag{lag_window}_{temporal_type}.pt"
    else:
        raise ValueError(f"Unknown graph_type={graph_type}")

    path = Path(PROC_DIR) / "gnn" / name
    print(f"[LOAD] {path}")
    pkg = torch.load(path, map_location="cpu", weights_only=False)
    return pkg


def get_time_splits(days, split):
    idx_train = [t for t in range(len(days)) if split[t] == "train"]
    idx_val = [t for t in range(len(days)) if split[t] == "val"]
    idx_test = [t for t in range(len(days)) if split[t] == "test"]
    print(f"[SPLIT] train={len(idx_train)}, val={len(idx_val)}, test={len(idx_test)}")
    return idx_train, idx_val, idx_test


def mape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.abs(y_true) > eps
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def smape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    denom = (np.abs(y_true) + np.abs(y_pred)) + eps
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom) * 100.0)


def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def get_mode_name(is_softplus: bool, is_log1p: bool) -> str:
    if is_softplus:
        return "softplus"
    if is_log1p:
        return "log1p"
    return "raw"


# =========================
# Plot predictions per product (test TS)
# =========================

def plot_predictions_per_product(
    days_test: np.ndarray,
    y_true_test: np.ndarray,
    y_pred_test: np.ndarray,
    out_dir,
    temporal_type: str,
    lag_window: int,
    graph_tag: str,
    max_plots: int | None = None,
) -> None:
    """
    Vẽ y_true vs y_pred theo ngày cho từng sản phẩm trên test split.
    Lưu 1 file .png / sản phẩm vào out_dir, prefix theo graph_tag.
    days_test: [T_test]
    y_true_test, y_pred_test: [T_test, N_prod]
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    T_test, N_prod = y_true_test.shape
    if max_plots is None or max_plots > N_prod:
        max_plots = N_prod

    for j in range(max_plots):
        plt.figure(figsize=(10, 4))
        plt.plot(days_test, y_true_test[:, j], label="True", marker="o", linewidth=1)
        plt.plot(days_test, y_pred_test[:, j], label="Pred", marker="x", linewidth=1)
        plt.title(
            f"{graph_tag} (temporal={temporal_type}, lag={lag_window}) - product_idx={j}"
        )
        plt.xlabel("Date")
        plt.ylabel("Sales")
        plt.legend()
        plt.tight_layout()

        fname = out_dir / f"{graph_tag}_lag{lag_window}_{temporal_type}_product{j}.png"
        plt.savefig(str(fname), dpi=150)
        plt.close()

    print(f"[PLOT] Saved per-product prediction plots to {out_dir}")


# =========================
# 1) Projected GNN baseline
# =========================

def run_projected_gnn_baseline(
    pkg,
    temporal_type: str,
    lag_window: int,
    edge_view: str,
    device: str = "cuda",
    epochs: int = 30,
    batch_days: int = 8,
    es_patience: int = 5,
    es_min_delta: float = 0.0,
    is_softplus: bool = False,
    is_log1p: bool = False,
    base_dir: Path | None = None,
):
    if is_softplus and is_log1p:
        raise ValueError("Only one of is_softplus / is_log1p can be True.")

    mode_name = get_mode_name(is_softplus, is_log1p)
    if base_dir is None:
        base_dir = Path(PROC_DIR) / "predictions" / "baseline_4" / f"{temporal_type}_{mode_name}"

    print(
        f"\n=== Training Projected GNN Baseline "
        f"[H{HORIZON}][lag{lag_window}][{temporal_type}][view={edge_view}] "
        f"[mode={mode_name}] ==="
    )

    X = pkg["X_product"].float()  # [T, N, F]
    Y = pkg["Y_product"].float()  # [T, N]
    days = np.array(pkg["days"])
    split = pkg["split"]
    edge_index_dict = pkg["edge_index_dict"]

    if edge_view not in edge_index_dict:
        raise ValueError(f"edge_view={edge_view} not in {list(edge_index_dict.keys())}")
    edge_index = edge_index_dict[edge_view]

    T, N, Fdim = X.shape
    idx_train, idx_val, idx_test = get_time_splits(days, split)

    device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    model = ProjectedGINRegressor(
        in_channels=Fdim,
        hidden_channels=128,
        num_layers=3,
        is_softplus=is_softplus,
        is_log1p=is_log1p,
    ).to(device)
    edge_index = edge_index.to(device)
    
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    loss_fn = nn.MSELoss()
    early_stopper = EarlyStopping(patience=es_patience, min_delta=es_min_delta)

    def iterate_days(day_indices, train_mode=True):
        if train_mode:
            model.train()
        else:
            model.eval()
        total_loss = 0.0
        count = 0

        for start in range(0, len(day_indices), batch_days):
            idx_block = day_indices[start:start + batch_days]
            X_block = X[idx_block].to(device)
            Y_block = Y[idx_block].to(device)

            loss_block = 0.0
            for b in range(X_block.size(0)):
                x_b = X_block[b]
                y_b = Y_block[b]

                y_hat_b = model(x_b, edge_index)
                loss_b = loss_fn(y_hat_b, y_b)
                loss_block += loss_b

            loss_block = loss_block / X_block.size(0)
            if train_mode:
                opt.zero_grad()
                loss_block.backward()
                opt.step()

            total_loss += loss_block.item()
            count += 1

        return total_loss / max(count, 1)

    for epoch in range(1, epochs + 1):
        train_loss = iterate_days(idx_train, train_mode=True)
        val_loss = iterate_days(idx_val, train_mode=False)
        print(
            f"[Projected-{edge_view}][{mode_name}] Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f}"
        )
        if early_stopper.step(val_loss, model):
            print(f"[Projected-{edge_view}][{mode_name}] Early stopping at epoch {epoch:03d}")
            break

    early_stopper.load_best(model)
    model.eval()

    with torch.no_grad():
        def predict_on_indices(idxs):
            preds_t = []
            trues = []
            for t in idxs:
                x_t = X[t].to(device)
                y_t = Y[t].to(device)
                y_hat_t = model(x_t, edge_index)
                preds_t.append(y_hat_t.cpu().numpy())
                trues.append(y_t.cpu().numpy())
            return np.concatenate(trues), np.concatenate(preds_t)

        y_train_true, y_train_pred_flat = predict_on_indices(idx_train)
        y_val_true, y_val_pred_flat = predict_on_indices(idx_val)
        y_test_true_flat, y_test_pred_flat = predict_on_indices(idx_test)

    T_test = len(idx_test)
    y_test_true = y_test_true_flat.reshape(T_test, N)
    y_test_pred = y_test_pred_flat.reshape(T_test, N)
    days_test = days[idx_test]

    mae_train = mae(y_train_true, y_train_pred_flat)
    rmse_train = rmse(y_train_true, y_train_pred_flat)
    mape_train = mape(y_train_true, y_train_pred_flat)
    smape_train = smape(y_train_true, y_train_pred_flat)

    mae_val = mae(y_val_true, y_val_pred_flat)
    rmse_val = rmse(y_val_true, y_val_pred_flat)
    mape_val = mape(y_val_true, y_val_pred_flat)
    smape_val = smape(y_val_true, y_val_pred_flat)

    mae_test = mae(y_test_true_flat, y_test_pred_flat)
    rmse_test = rmse(y_test_true_flat, y_test_pred_flat)
    mape_test = mape(y_test_true_flat, y_test_pred_flat)
    smape_test = smape(y_test_true_flat, y_test_pred_flat)

    tag = f"gnn_projected_{edge_view}_h{HORIZON}_lag{lag_window}_{temporal_type}_{mode_name}"

    print(f"\n[Projected-{edge_view}][{tag}] Train:")
    print(f"  MAE  : {mae_train:.4f}")
    print(f"  RMSE : {rmse_train:.4f}")
    print(f"  MAPE : {mape_train:.4f}")
    print(f"  sMAPE: {smape_train:.4f}")

    print(f"\n[Projected-{edge_view}][{tag}] Val:")
    print(f"  MAE  : {mae_val:.4f}")
    print(f"  RMSE : {rmse_val:.4f}")
    print(f"  MAPE : {mape_val:.4f}")
    print(f"  sMAPE: {smape_val:.4f}")

    print(f"\n[Projected-{edge_view}][{tag}] Test:")
    print(f"  MAE  : {mae_test:.4f}")
    print(f"  RMSE : {rmse_test:.4f}")
    print(f"  MAPE : {mape_test:.4f}")
    print(f"  sMAPE: {smape_test:.4f}")

    # CSV test predictions
    out_pred_dir = base_dir / "csv" / "projected"
    out_pred_dir.mkdir(parents=True, exist_ok=True)
    df_test_pred = pd.DataFrame(
        {
            "date": np.repeat(days_test, N),
            "product_idx": np.tile(np.arange(N), T_test),
            "y_true": y_test_true_flat,
            "y_pred": y_test_pred_flat,
        }
    )
    out_pred_file = out_pred_dir / f"{tag}_test_predictions.csv"
    df_test_pred.to_csv(out_pred_file, index=False)
    print(f"[SAVE] Test predictions saved to {out_pred_file}")

    # plots
    out_plot_dir = base_dir / "plots_projected"
    plot_predictions_per_product(
        days_test=days_test,
        y_true_test=y_test_true,
        y_pred_test=y_test_pred,
        out_dir=out_plot_dir,
        temporal_type=temporal_type,
        lag_window=lag_window,
        graph_tag=f"projected_{edge_view}_{mode_name}",
        max_plots=None,
    )

    RUN_SUMMARY.append(
        {
            "temporal_type": temporal_type,
            "lag_window": lag_window,
            "horizon": HORIZON,
            "variant": "gnn_projected",
            "tag": tag,
            "edge_view": edge_view,
            "target_transform": mode_name,
            "MAE_train": mae_train,
            "RMSE_train": rmse_train,
            "MAPE_train": mape_train,
            "sMAPE_train": smape_train,
            "MAE_val": mae_val,
            "RMSE_val": rmse_val,
            "MAPE_val": mape_val,
            "sMAPE_val": smape_val,
            "MAE_test": mae_test,
            "RMSE_test": rmse_test,
            "MAPE_test": mape_test,
            "sMAPE_test": smape_test,
        }
    )


# =========================
# 2) Homogeneous 5-type GNN baseline
# =========================

def run_homo5_gnn_baseline(
    pkg,
    temporal_type: str,
    lag_window: int,
    device: str = "cuda",
    epochs: int = 30,
    batch_days: int = 8,
    es_patience: int = 5,
    es_min_delta: float = 0.0,
    is_softplus: bool = False,
    is_log1p: bool = False,
    base_dir: Path | None = None,
):
    if is_softplus and is_log1p:
        raise ValueError("Only one of is_softplus / is_log1p can be True.")
    mode_name = get_mode_name(is_softplus, is_log1p)
    if base_dir is None:
        base_dir = Path(PROC_DIR) / "predictions" / "baseline_4" / f"{temporal_type}_{mode_name}"

    print(
        f"\n=== Training Homogeneous-5type GNN Baseline "
        f"[H{HORIZON}][lag{lag_window}][{temporal_type}] "
        f"[mode={mode_name}] ==="
    )

    X_prod = pkg["X_product"].float()
    Y_prod = pkg["Y_product"].float()
    days = np.array(pkg["days"])
    split = pkg["split"]
    edge_index_dict = pkg["edge_index_dict"]
    num_nodes_dict = pkg["num_nodes_dict"]

    node_type_order = list(num_nodes_dict.keys())

    offsets = {}
    offset = 0
    for nt in node_type_order:
        offsets[nt] = offset
        offset += num_nodes_dict[nt]
    N_total = offset

    all_edges = []
    for (src_type, rel_name, dst_type), ei in edge_index_dict.items():
        if ei.numel() == 0:
            continue
        src_offset = offsets[src_type]
        dst_offset = offsets[dst_type]
        src_global = ei[0, :] + src_offset
        dst_global = ei[1, :] + dst_offset
        all_edges.append(torch.stack([src_global, dst_global], dim=0))
        all_edges.append(torch.stack([dst_global, src_global], dim=0))

    if len(all_edges) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.cat(all_edges, dim=1)

    print(f"[Homo5] N_total={N_total}, edges={edge_index.size(1)}")

    T, N_prod, Fdim = X_prod.shape
    idx_train, idx_val, idx_test = get_time_splits(days, split)

    device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    model = HomogeneousFiveTypeGINRegressor(
        in_channels=Fdim,
        num_nodes_dict=num_nodes_dict,
        node_type_order=node_type_order,
        hidden_channels=128,
        num_layers=3,
        node_type_emb_dim=8,
        is_softplus=is_softplus,
        is_log1p=is_log1p,
    ).to(device)
    edge_index = edge_index.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    other_types = [nt for nt in node_type_order if nt != "product"]
    early_stopper = EarlyStopping(patience=es_patience, min_delta=es_min_delta)

    def iterate_days(day_indices, train_mode=True):
        if train_mode:
            model.train()
        else:
            model.eval()
        total_loss = 0.0
        count = 0

        for start in range(0, len(day_indices), batch_days):
            idx_block = day_indices[start:start + batch_days]
            X_block = X_prod[idx_block].to(device)
            Y_block = Y_prod[idx_block].to(device)

            loss_block = 0.0
            for b in range(X_block.size(0)):
                x_prod_b = X_block[b]
                y_b = Y_block[b]

                x_dict = {
                    "product": x_prod_b,
                    **{
                        nt: torch.zeros(num_nodes_dict[nt], Fdim, device=device)
                        for nt in other_types
                    },
                }

                y_hat_b = model(x_dict, edge_index)
                loss_b = loss_fn(y_hat_b, y_b)
                loss_block += loss_b

            loss_block = loss_block / X_block.size(0)
            if train_mode:
                opt.zero_grad()
                loss_block.backward()
                opt.step()

            total_loss += loss_block.item()
            count += 1

        return total_loss / max(count, 1)

    for epoch in range(1, epochs + 1):
        train_loss = iterate_days(idx_train, train_mode=True)
        val_loss = iterate_days(idx_val, train_mode=False)
        print(f"[Homo5][{mode_name}] Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if early_stopper.step(val_loss, model):
            print(f"[Homo5][{mode_name}] Early stopping at epoch {epoch:03d}")
            break

    early_stopper.load_best(model)
    model.eval()

    with torch.no_grad():
        def predict_on_indices(idxs):
            preds_t = []
            trues = []
            for t in idxs:
                x_prod_t = X_prod[t].to(device)
                y_t = Y_prod[t].to(device)
                x_dict_t = {
                    "product": x_prod_t,
                    **{
                        nt: torch.zeros(num_nodes_dict[nt], Fdim, device=device)
                        for nt in other_types
                    },
                }
                y_hat_t = model(x_dict_t, edge_index)
                preds_t.append(y_hat_t.cpu().numpy())
                trues.append(y_t.cpu().numpy())
            return np.concatenate(trues), np.concatenate(preds_t)

        y_train_true, y_train_pred_flat = predict_on_indices(idx_train)
        y_val_true, y_val_pred_flat = predict_on_indices(idx_val)
        y_test_true_flat, y_test_pred_flat = predict_on_indices(idx_test)

    T_test = len(idx_test)
    N_prod = Y_prod.shape[1]
    y_test_true = y_test_true_flat.reshape(T_test, N_prod)
    y_test_pred = y_test_pred_flat.reshape(T_test, N_prod)
    days_test = days[idx_test]

    mae_train = mae(y_train_true, y_train_pred_flat)
    rmse_train = rmse(y_train_true, y_train_pred_flat)
    mape_train = mape(y_train_true, y_train_pred_flat)
    smape_train = smape(y_train_true, y_train_pred_flat)

    mae_val = mae(y_val_true, y_val_pred_flat)
    rmse_val = rmse(y_val_true, y_val_pred_flat)
    mape_val = mape(y_val_true, y_val_pred_flat)
    smape_val = smape(y_val_true, y_val_pred_flat)

    mae_test = mae(y_test_true_flat, y_test_pred_flat)
    rmse_test = rmse(y_test_true_flat, y_test_pred_flat)
    mape_test = mape(y_test_true_flat, y_test_pred_flat)
    smape_test = smape(y_test_true_flat, y_test_pred_flat)

    tag = f"gnn_homo5_h{HORIZON}_lag{lag_window}_{temporal_type}_{mode_name}"

    print(f"\n[Homo5][{tag}] Train:")
    print(f"  MAE  : {mae_train:.4f}")
    print(f"  RMSE : {rmse_train:.4f}")
    print(f"  MAPE : {mape_train:.4f}")
    print(f"  sMAPE: {smape_train:.4f}")

    print(f"\n[Homo5][{tag}] Val:")
    print(f"  MAE  : {mae_val:.4f}")
    print(f"  RMSE : {rmse_val:.4f}")
    print(f"  MAPE : {mape_val:.4f}")
    print(f"  sMAPE: {smape_val:.4f}")

    print(f"\n[Homo5][{tag}] Test:")
    print(f"  MAE  : {mae_test:.4f}")
    print(f"  RMSE : {rmse_test:.4f}")
    print(f"  MAPE : {mape_test:.4f}")
    print(f"  sMAPE: {smape_test:.4f}")

    out_pred_dir = base_dir / "csv" / "homo5"
    out_pred_dir.mkdir(parents=True, exist_ok=True)
    df_test_pred = pd.DataFrame(
        {
            "date": np.repeat(days_test, N_prod),
            "product_idx": np.tile(np.arange(N_prod), T_test),
            "y_true": y_test_true_flat,
            "y_pred": y_test_pred_flat,
        }
    )
    out_pred_file = out_pred_dir / f"{tag}_test_predictions.csv"
    df_test_pred.to_csv(out_pred_file, index=False)
    print(f"[SAVE] Test predictions saved to {out_pred_file}")

    out_plot_dir = base_dir / "plots_homo5"
    plot_predictions_per_product(
        days_test=days_test,
        y_true_test=y_test_true,
        y_pred_test=y_test_pred,
        out_dir=out_plot_dir,
        temporal_type=temporal_type,
        lag_window=lag_window,
        graph_tag=f"homo5_{mode_name}",
        max_plots=None,
    )

    RUN_SUMMARY.append(
        {
            "temporal_type": temporal_type,
            "lag_window": lag_window,
            "horizon": HORIZON,
            "variant": "gnn_homo5",
            "tag": tag,
            "edge_view": None,
            "target_transform": mode_name,
            "MAE_train": mae_train,
            "RMSE_train": rmse_train,
            "MAPE_train": mape_train,
            "sMAPE_train": smape_train,
            "MAE_val": mae_val,
            "RMSE_val": rmse_val,
            "MAPE_val": mape_val,
            "sMAPE_val": smape_val,
            "MAE_test": mae_test,
            "RMSE_test": rmse_test,
            "MAPE_test": mape_test,
            "sMAPE_test": smape_test,
        }
    )


# =========================
# 3) Heterogeneous 5-type GNN baseline
# =========================

def run_hetero5_gnn_baseline(
    pkg,
    temporal_type: str,
    lag_window: int,
    device: str = "cuda",
    epochs: int = 30,
    batch_days: int = 8,
    es_patience: int = 5,
    es_min_delta: float = 0.0,
    is_softplus: bool = False,
    is_log1p: bool = False,
    base_dir: Path | None = None,
):
    if is_softplus and is_log1p:
        raise ValueError("Only one of is_softplus / is_log1p can be True.")
    mode_name = get_mode_name(is_softplus, is_log1p)
    if base_dir is None:
        base_dir = Path(PROC_DIR) / "predictions" / "baseline_4" / f"{temporal_type}_{mode_name}"

    print(
        f"\n=== Training Heterogeneous-5type GNN Baseline "
        f"[H{HORIZON}][lag{lag_window}][{temporal_type}] "
        f"[mode={mode_name}] ==="
    )

    X_prod = pkg["X_product"].float()
    Y_prod = pkg["Y_product"].float()
    days = np.array(pkg["days"])
    split = pkg["split"]
    edge_index_dict = pkg["edge_index_dict"]
    num_nodes_dict = pkg["num_nodes_dict"]

    idx_train, idx_val, idx_test = get_time_splits(days, split)
    T, N_prod, Fdim = X_prod.shape

    edge_types = list(edge_index_dict.keys())
    in_channels_dict = {"edge_types": edge_types, "product": Fdim}
    other_types = [nt for nt in num_nodes_dict.keys() if nt != "product"]
    for nt in other_types:
        in_channels_dict[nt] = Fdim

    device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    model = HeterogeneousGINRegressor(
        in_channels_dict=in_channels_dict,
        hidden_channels=128,
        num_layers=2,
        is_softplus=is_softplus,
        is_log1p=is_log1p,
    ).to(device)
    edge_index_dict = {k: v.to(device) for k, v in edge_index_dict.items()}

    base_x_dict = {
        nt: torch.zeros(num_nodes_dict[nt], Fdim) for nt in num_nodes_dict.keys()
    }

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    early_stopper = EarlyStopping(patience=es_patience, min_delta=es_min_delta)

    def iterate_days(day_indices, train_mode=True):
        if train_mode:
            model.train()
        else:
            model.eval()

        total_loss = 0.0
        count = 0
        for start in range(0, len(day_indices), batch_days):
            idx_block = day_indices[start:start + batch_days]
            X_block = X_prod[idx_block].to(device)
            Y_block = Y_prod[idx_block].to(device)

            loss_block = 0.0
            for b in range(X_block.size(0)):
                x_prod_b = X_block[b]
                y_b = Y_block[b]

                x_dict = {nt: base_x_dict[nt].to(device) for nt in base_x_dict.keys()}
                x_dict["product"] = x_prod_b

                y_hat_b = model(x_dict, edge_index_dict)
                loss_b = loss_fn(y_hat_b, y_b)
                loss_block += loss_b

            loss_block = loss_block / X_block.size(0)
            if train_mode:
                opt.zero_grad()
                loss_block.backward()
                opt.step()

            total_loss += loss_block.item()
            count += 1

        return total_loss / max(count, 1)

    for epoch in range(1, epochs + 1):
        train_loss = iterate_days(idx_train, train_mode=True)
        val_loss = iterate_days(idx_val, train_mode=False)
        print(f"[Hetero5][{mode_name}] Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if early_stopper.step(val_loss, model):
            print(f"[Hetero5][{mode_name}] Early stopping at epoch {epoch:03d}")
            break

    early_stopper.load_best(model)
    model.eval()

    with torch.no_grad():
        def predict_on_indices(idxs):
            preds_t = []
            trues = []
            for t in idxs:
                x_prod_t = X_prod[t].to(device)
                y_t = Y_prod[t].to(device)

                x_dict = {nt: base_x_dict[nt].to(device) for nt in base_x_dict.keys()}
                x_dict["product"] = x_prod_t

                y_hat_t = model(x_dict, edge_index_dict)
                preds_t.append(y_hat_t.cpu().numpy())
                trues.append(y_t.cpu().numpy())
            return np.concatenate(trues), np.concatenate(preds_t)

        y_train_true, y_train_pred_flat = predict_on_indices(idx_train)
        y_val_true, y_val_pred_flat = predict_on_indices(idx_val)
        y_test_true_flat, y_test_pred_flat = predict_on_indices(idx_test)

    T_test = len(idx_test)
    y_test_true = y_test_true_flat.reshape(T_test, N_prod)
    y_test_pred = y_test_pred_flat.reshape(T_test, N_prod)
    days_test = days[idx_test]

    mae_train = mae(y_train_true, y_train_pred_flat)
    rmse_train = rmse(y_train_true, y_train_pred_flat)
    mape_train = mape(y_train_true, y_train_pred_flat)
    smape_train = smape(y_train_true, y_train_pred_flat)

    mae_val = mae(y_val_true, y_val_pred_flat)
    rmse_val = rmse(y_val_true, y_val_pred_flat)
    mape_val = mape(y_val_true, y_val_pred_flat)
    smape_val = smape(y_val_true, y_val_pred_flat)

    mae_test = mae(y_test_true_flat, y_test_pred_flat)
    rmse_test = rmse(y_test_true_flat, y_test_pred_flat)
    mape_test = mape(y_test_true_flat, y_test_pred_flat)
    smape_test = smape(y_test_true_flat, y_test_pred_flat)

    tag = f"gnn_hetero5_h{HORIZON}_lag{lag_window}_{temporal_type}_{mode_name}"

    print(f"\n[Hetero5][{tag}] Train:")
    print(f"  MAE  : {mae_train:.4f}")
    print(f"  RMSE : {rmse_train:.4f}")
    print(f"  MAPE : {mape_train:.4f}")
    print(f"  sMAPE: {smape_train:.4f}")

    print(f"\n[Hetero5][{tag}] Val:")
    print(f"  MAE  : {mae_val:.4f}")
    print(f"  RMSE : {rmse_val:.4f}")
    print(f"  MAPE : {mape_val:.4f}")
    print(f"  sMAPE: {smape_val:.4f}")

    print(f"\n[Hetero5][{tag}] Test:")
    print(f"  MAE  : {mae_test:.4f}")
    print(f"  RMSE : {rmse_test:.4f}")
    print(f"  MAPE : {mape_test:.4f}")
    print(f"  sMAPE: {smape_test:.4f}")

    out_pred_dir = base_dir / "csv" / "hetero5"
    out_pred_dir.mkdir(parents=True, exist_ok=True)
    df_test_pred = pd.DataFrame(
        {
            "date": np.repeat(days_test, N_prod),
            "product_idx": np.tile(np.arange(N_prod), T_test),
            "y_true": y_test_true_flat,
            "y_pred": y_test_pred_flat,
        }
    )
    out_pred_file = out_pred_dir / f"{tag}_test_predictions.csv"
    df_test_pred.to_csv(out_pred_file, index=False)
    print(f"[SAVE] Test predictions saved to {out_pred_file}")

    out_plot_dir = base_dir / "plots_hetero5"
    plot_predictions_per_product(
        days_test=days_test,
        y_true_test=y_test_true,
        y_pred_test=y_test_pred,
        out_dir=out_plot_dir,
        temporal_type=temporal_type,
        lag_window=lag_window,
        graph_tag=f"hetero5_{mode_name}",
        max_plots=None,
    )

    RUN_SUMMARY.append(
        {
            "temporal_type": temporal_type,
            "lag_window": lag_window,
            "horizon": HORIZON,
            "variant": "gnn_hetero5",
            "tag": tag,
            "edge_view": None,
            "target_transform": mode_name,
            "MAE_train": mae_train,
            "RMSE_train": rmse_train,
            "MAPE_train": mape_train,
            "sMAPE_train": smape_train,
            "MAE_val": mae_val,
            "RMSE_val": rmse_val,
            "MAPE_val": mape_val,
            "sMAPE_val": smape_val,
            "MAE_test": mae_test,
            "RMSE_test": rmse_test,
            "MAPE_test": mape_test,
            "sMAPE_test": smape_test,
        }
    )


# =========================
# main: chạy lần lượt 3 graph type cho từng ExperimentConfig
# =========================

def main():
    global RUN_SUMMARY
    RUN_SUMMARY = []

    es_patience = 20
    es_min_delta = 0.001

    for exp in DEFAULT_EXPERIMENTS:
        temporal_type = exp.temporal_type
        is_sp = exp.is_softplus
        is_log1p = exp.is_log1p

        if is_sp:
            mode_name = "softplus"
        elif is_log1p:
            mode_name = "log1p"
        else:
            mode_name = "raw"

        base_dir = Path(PROC_DIR) / "predictions" / "baseline_4" / f"{temporal_type}_{mode_name}"

        for lag_window in exp.lag_windows:
            print(f"\n############ GNN Baselines: temporal={temporal_type}, lag={lag_window} ############")

            # 1) Projected
            pkg_proj = load_gnn_pkg("projected", temporal_type, lag_window)
            for view in PROJECTED_VIEWS:
                run_projected_gnn_baseline(
                    pkg=pkg_proj,
                    temporal_type=temporal_type,
                    lag_window=lag_window,
                    edge_view=view,
                    device="cuda",
                    epochs=300,
                    batch_days=8,
                    es_patience=es_patience,
                    es_min_delta=es_min_delta,
                    is_softplus=is_sp,
                    is_log1p=is_log1p,
                    base_dir=base_dir,
                )

            # 2) Homogeneous 5-type
            pkg_homo5 = load_gnn_pkg("homo5", temporal_type, lag_window)
            run_homo5_gnn_baseline(
                pkg=pkg_homo5,
                temporal_type=temporal_type,
                lag_window=lag_window,
                device="cuda",
                epochs=300,
                batch_days=8,
                es_patience=es_patience,
                es_min_delta=es_min_delta,
                is_softplus=is_sp,
                is_log1p=is_log1p,
                base_dir=base_dir,
            )

            # 3) Heterogeneous 5-type
            pkg_hetero5 = load_gnn_pkg("hetero5", temporal_type, lag_window)
            run_hetero5_gnn_baseline(
                pkg=pkg_hetero5,
                temporal_type=temporal_type,
                lag_window=lag_window,
                device="cuda",
                epochs=300,
                batch_days=8,
                es_patience=es_patience,
                es_min_delta=es_min_delta,
                is_softplus=is_sp,
                is_log1p=is_log1p,
                base_dir=base_dir,
            )

        if RUN_SUMMARY:
            df_sum = pd.DataFrame(RUN_SUMMARY)
            print("\n=== GNN baselines summary (this ExperimentConfig) ===")
            df_sum = df_sum.sort_values(
                [
                    "temporal_type",
                    "lag_window",
                    "horizon",
                    "variant",
                    "tag",
                    "edge_view",
                    "target_transform",
                ]
            )

            base_dir.mkdir(parents=True, exist_ok=True)
            out_path = base_dir / f"summary_baseline_4_{temporal_type}_{mode_name}.csv"
            df_sum.to_csv(out_path, index=False)
            print(f"\nSaved GNN baseline summary to {out_path}")

            RUN_SUMMARY = []


if __name__ == "__main__":
    main()
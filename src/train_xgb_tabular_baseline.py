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
from config.config import PROC_DIR, TEMPORAL_TYPE


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
# Target transform helpers (chỉ dùng raw)
# =========================

def transform_target(y: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "raw":
        return y
    else:
        raise ValueError(f"Only 'raw' target is supported, got mode={mode}")


def inverse_transform_target(y_pred: np.ndarray, mode: str) -> np.ndarray:
    if mode == "raw":
        return y_pred
    else:
        raise ValueError(f"Only 'raw' target is supported, got mode={mode}")


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
    # explicit weights_only=False để tránh cảnh báo các version mới
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


# =========================
# Plot time series target theo product
# =========================

def plot_product_time_series(pkg, temporal_type: str, lag_window: int, max_products: int = 20):
    """
    Vẽ time series target theo từng product từ Y_product và days.
    Lưu PNG dưới PROC_DIR/predictions/gnn_baselines/plots/.
    """
    Y_prod = pkg["Y_product"].float().numpy()  # [T, N_prod]
    days = np.array(pkg["days"])
    T, N_prod = Y_prod.shape

    n = min(N_prod, max_products)
    n_cols = 4
    n_rows = int(np.ceil(n / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), sharex=True)
    axes = axes.flatten()

    for i in range(n):
        ax = axes[i]
        ax.plot(days, Y_prod[:, i], lw=1.0)
        ax.set_title(f"Product {i}")
        ax.tick_params(axis="x", rotation=45)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        f"Product demand time series (temporal={temporal_type}, lag={lag_window})",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_dir = Path(PROC_DIR) / "predictions" / "gnn_baselines" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"ts_products_{temporal_type}_lag{lag_window}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"[PLOT] Saved product time series plot to {out_path}")


# =========================
# Plot predictions per product (giống XGB)
# =========================

def plot_gnn_predictions_per_product(
    days: np.ndarray,
    y_true_ts: np.ndarray,      # [T, N_prod]
    y_pred_ts: np.ndarray,      # [T, N_prod]
    node_ids: np.ndarray | None,
    out_dir: Path,
    temporal_type: str,
    lag_window: int,
    graph_variant: str,
    target_mode: str,
    view: str | None = None,
    max_plots: int | None = None,
) -> None:
    """
    Vẽ y_true vs y_pred theo ngày cho từng product (node cột) trên toàn bộ T.
    Giống plot XGB: 1 file .png / product.
    """
    T, N = y_true_ts.shape
    assert y_pred_ts.shape == (T, N)

    if node_ids is None:
        node_ids = np.arange(N)
    else:
        node_ids = np.asarray(node_ids)

    unique_nodes = node_ids
    if max_plots is not None:
        unique_nodes = unique_nodes[:max_plots]

    out_dir.mkdir(parents=True, exist_ok=True)

    for j, node in enumerate(unique_nodes):
        idx = np.where(node_ids == node)[0]
        if len(idx) == 0:
            continue
        col = idx[0]

        y_true_j = y_true_ts[:, col]
        y_pred_j = y_pred_ts[:, col]

        plt.figure(figsize=(10, 4))
        plt.plot(days, y_true_j, label="True", marker="o", linewidth=1)
        plt.plot(days, y_pred_j, label="Pred", marker="x", linewidth=1)
        title = f"GNN {graph_variant}"
        if view is not None:
            title += f" ({view})"
        title += f" - {target_mode} - node={node}"
        plt.title(title)
        plt.xlabel("Day")
        plt.ylabel("Sales order")
        plt.legend()
        plt.tight_layout()

        if view is not None:
            fname = out_dir / f"gnn_{graph_variant}_{view}_{target_mode}_lag{lag_window}_node_{node}.png"
        else:
            fname = out_dir / f"gnn_{graph_variant}_{target_mode}_lag{lag_window}_node_{node}.png"

        plt.savefig(fname, dpi=150)
        plt.close()


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
    target_transform: str = "raw",       # chỉ 'raw'
    clip_pred: bool = True,             # clip 0 khi compute metric
    use_softplus_output: bool = False,  # softplus ở output
):
    tag_mode = f"{target_transform}{'_clip' if clip_pred else '_noclip'}"
    print(
        f"\n=== Training Projected GNN Baseline "
        f"[H{HORIZON}][lag{lag_window}][{temporal_type}][view={edge_view}] "
        f"[mode={tag_mode}][softplus={use_softplus_output}] ==="
    )

    X = pkg["X_product"].float()  # [T, N, F]
    Y = pkg["Y_product"].float()  # [T, N]
    days = pkg["days"]
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
        use_softplus_output=use_softplus_output,
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
                x_b = X_block[b]        # [N, F]
                y_b = Y_block[b]        # [N]
                y_b_t = transform_target(y_b, target_transform)

                y_hat_b = model(x_b, edge_index)
                loss_b = loss_fn(y_hat_b, y_b_t)
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
            f"[Projected-{edge_view}][{tag_mode}] Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f}"
        )
        if early_stopper.step(val_loss, model):
            print(f"[Projected-{edge_view}][{tag_mode}] Early stopping at epoch {epoch:03d}")
            break

    early_stopper.load_best(model)
    model.eval()

    # full time series prediction [T, N]
    with torch.no_grad():
        y_true_ts = []
        y_pred_ts = []
        for t in range(T):
            x_t = X[t].to(device)
            y_t = Y[t].to(device)
            y_hat_t = model(x_t, edge_index)

            y_true_ts.append(y_t.cpu().numpy())
            y_pred_ts.append(y_hat_t.cpu().numpy())

        y_true_ts = np.stack(y_true_ts, axis=0)  # [T, N]
        y_pred_ts = np.stack(y_pred_ts, axis=0)  # [T, N]

    # lấy theo split để tính metric
    def gather_by_indices(idxs):
        idxs = np.asarray(idxs, dtype=int)
        return (
            y_true_ts[idxs].reshape(-1),
            y_pred_ts[idxs].reshape(-1),
        )

    y_train_true, y_train_pred_t = gather_by_indices(idx_train)
    y_val_true, y_val_pred_t = gather_by_indices(idx_val)
    y_test_true, y_test_pred_t = gather_by_indices(idx_test)

    # inverse transform & clip (trên vector)
    y_train_pred = inverse_transform_target(y_train_pred_t, target_transform)
    y_val_pred = inverse_transform_target(y_val_pred_t, target_transform)
    y_test_pred = inverse_transform_target(y_test_pred_t, target_transform)

    if clip_pred:
        y_train_pred = np.maximum(y_train_pred, 0.0)
        y_val_pred = np.maximum(y_val_pred, 0.0)
        y_test_pred = np.maximum(y_test_pred, 0.0)

    # phiên bản series cho plot
    y_pred_ts_plot = inverse_transform_target(y_pred_ts.reshape(-1), target_transform).reshape(T, N)
    if clip_pred:
        y_pred_ts_plot = np.maximum(y_pred_ts_plot, 0.0)
    y_true_ts_plot = y_true_ts

    mae_train = mae(y_train_true, y_train_pred)
    rmse_train = rmse(y_train_true, y_train_pred)
    mape_train = mape(y_train_true, y_train_pred)
    smape_train = smape(y_train_true, y_train_pred)

    mae_val = mae(y_val_true, y_val_pred)
    rmse_val = rmse(y_val_true, y_val_pred)
    mape_val = mape(y_val_true, y_val_pred)
    smape_val = smape(y_val_true, y_val_pred)

    mae_test = mae(y_test_true, y_test_pred)
    rmse_test = rmse(y_test_true, y_test_pred)
    mape_test = mape(y_test_true, y_test_pred)
    smape_test = smape(y_test_true, y_test_pred)

    tag = f"gnn_projected_{edge_view}_lag{lag_window}_{temporal_type}_{tag_mode}{'_sp' if use_softplus_output else ''}"
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

    RUN_SUMMARY.append(
        {
            "temporal_type": temporal_type,
            "lag_window": lag_window,
            "horizon": HORIZON,
            "variant": "gnn_projected",
            "tag": tag,
            "edge_view": edge_view,
            "target_transform": target_transform,
            "clip_pred": clip_pred,
            "use_softplus_output": use_softplus_output,
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

    # Plot per-product như XGB
    days_np = np.array(days)
    node_ids = pkg.get("node_ids", None)
    base_pred_dir = Path(PROC_DIR) / "predictions" / "gnn_baselines"
    plot_dir = base_pred_dir / "plots_per_product" / f"{temporal_type}" / f"lag{lag_window}"
    target_mode_str = tag_mode if not use_softplus_output else f"{tag_mode}_sp"

    plot_gnn_predictions_per_product(
        days=days_np,
        y_true_ts=y_true_ts_plot,
        y_pred_ts=y_pred_ts_plot,
        node_ids=node_ids,
        out_dir=plot_dir,
        temporal_type=temporal_type,
        lag_window=lag_window,
        graph_variant="projected",
        target_mode=target_mode_str,
        view=edge_view,
        max_plots=None,
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
    target_transform: str = "raw",
    clip_pred: bool = True,
    use_softplus_output: bool = False,
):
    tag_mode = f"{target_transform}{'_clip' if clip_pred else '_noclip'}"
    print(
        f"\n=== Training Homogeneous-5type GNN Baseline "
        f"[H{HORIZON}][lag{lag_window}][{temporal_type}] "
        f"[mode={tag_mode}][softplus={use_softplus_output}] ==="
    )

    X_prod = pkg["X_product"].float()
    Y_prod = pkg["Y_product"].float()
    days = pkg["days"]
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
        use_softplus_output=use_softplus_output,
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
                y_b_t = transform_target(y_b, target_transform)

                x_dict = {
                    "product": x_prod_b,
                    **{
                        nt: torch.zeros(num_nodes_dict[nt], Fdim, device=device)
                        for nt in other_types
                    },
                }

                y_hat_b = model(x_dict, edge_index)
                loss_b = loss_fn(y_hat_b, y_b_t)
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
        print(f"[Homo5][{tag_mode}] Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if early_stopper.step(val_loss, model):
            print(f"[Homo5][{tag_mode}] Early stopping at epoch {epoch:03d}")
            break

    early_stopper.load_best(model)
    model.eval()

    # full time series prediction [T, N_prod]
    with torch.no_grad():
        y_true_ts = []
        y_pred_ts = []
        for t in range(T):
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

            y_true_ts.append(y_t.cpu().numpy())
            y_pred_ts.append(y_hat_t.cpu().numpy())

        y_true_ts = np.stack(y_true_ts, axis=0)
        y_pred_ts = np.stack(y_pred_ts, axis=0)

    def gather_by_indices(idxs):
        idxs = np.asarray(idxs, dtype=int)
        return (
            y_true_ts[idxs].reshape(-1),
            y_pred_ts[idxs].reshape(-1),
        )

    y_train_true, y_train_pred_t = gather_by_indices(idx_train)
    y_val_true, y_val_pred_t = gather_by_indices(idx_val)
    y_test_true, y_test_pred_t = gather_by_indices(idx_test)

    y_train_pred = inverse_transform_target(y_train_pred_t, target_transform)
    y_val_pred = inverse_transform_target(y_val_pred_t, target_transform)
    y_test_pred = inverse_transform_target(y_test_pred_t, target_transform)

    if clip_pred:
        y_train_pred = np.maximum(y_train_pred, 0.0)
        y_val_pred = np.maximum(y_val_pred, 0.0)
        y_test_pred = np.maximum(y_test_pred, 0.0)

    y_pred_ts_plot = inverse_transform_target(y_pred_ts.reshape(-1), target_transform).reshape(T, N_prod)
    if clip_pred:
        y_pred_ts_plot = np.maximum(y_pred_ts_plot, 0.0)
    y_true_ts_plot = y_true_ts

    mae_train = mae(y_train_true, y_train_pred)
    rmse_train = rmse(y_train_true, y_train_pred)
    mape_train = mape(y_train_true, y_train_pred)
    smape_train = smape(y_train_true, y_train_pred)

    mae_val = mae(y_val_true, y_val_pred)
    rmse_val = rmse(y_val_true, y_val_pred)
    mape_val = mape(y_val_true, y_val_pred)
    smape_val = smape(y_val_true, y_val_pred)

    mae_test = mae(y_test_true, y_test_pred)
    rmse_test = rmse(y_test_true, y_test_pred)
    mape_test = mape(y_test_true, y_test_pred)
    smape_test = smape(y_test_true, y_test_pred)

    tag = f"gnn_homo5_lag{lag_window}_{temporal_type}_{tag_mode}{'_sp' if use_softplus_output else ''}"

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

    RUN_SUMMARY.append(
        {
            "temporal_type": temporal_type,
            "lag_window": lag_window,
            "horizon": HORIZON,
            "variant": "gnn_homo5",
            "tag": tag,
            "edge_view": None,
            "target_transform": target_transform,
            "clip_pred": clip_pred,
            "use_softplus_output": use_softplus_output,
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

    # Plot per-product
    days_np = np.array(days)
    node_ids = pkg.get("node_ids", None)
    base_pred_dir = Path(PROC_DIR) / "predictions" / "gnn_baselines"
    plot_dir = base_pred_dir / "plots_per_product" / f"{temporal_type}" / f"lag{lag_window}"
    target_mode_str = tag_mode if not use_softplus_output else f"{tag_mode}_sp"

    plot_gnn_predictions_per_product(
        days=days_np,
        y_true_ts=y_true_ts_plot,
        y_pred_ts=y_pred_ts_plot,
        node_ids=node_ids,
        out_dir=plot_dir,
        temporal_type=temporal_type,
        lag_window=lag_window,
        graph_variant="homo5",
        target_mode=target_mode_str,
        view=None,
        max_plots=None,
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
    target_transform: str = "raw",
    clip_pred: bool = True,
    use_softplus_output: bool = False,
):
    tag_mode = f"{target_transform}{'_clip' if clip_pred else '_noclip'}"
    print(
        f"\n=== Training Heterogeneous-5type GNN Baseline "
        f"[H{HORIZON}][lag{lag_window}][{temporal_type}] "
        f"[mode={tag_mode}][softplus={use_softplus_output}] ==="
    )

    X_prod = pkg["X_product"].float()
    Y_prod = pkg["Y_product"].float()
    days = pkg["days"]
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
        use_softplus_output=use_softplus_output,
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
                y_b_t = transform_target(y_b, target_transform)

                x_dict = {nt: base_x_dict[nt].to(device) for nt in base_x_dict.keys()}
                x_dict["product"] = x_prod_b

                y_hat_b = model(x_dict, edge_index_dict)
                loss_b = loss_fn(y_hat_b, y_b_t)
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
        print(f"[Hetero5][{tag_mode}] Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if early_stopper.step(val_loss, model):
            print(f"[Hetero5][{tag_mode}] Early stopping at epoch {epoch:03d}")
            break

    early_stopper.load_best(model)
    model.eval()

    # full time series prediction [T, N_prod]
    with torch.no_grad():
        y_true_ts = []
        y_pred_ts = []
        for t in range(T):
            x_prod_t = X_prod[t].to(device)
            y_t = Y_prod[t].to(device)

            x_dict = {nt: base_x_dict[nt].to(device) for nt in base_x_dict.keys()}
            x_dict["product"] = x_prod_t

            y_hat_t = model(x_dict, edge_index_dict)
            y_true_ts.append(y_t.cpu().numpy())
            y_pred_ts.append(y_hat_t.cpu().numpy())

        y_true_ts = np.stack(y_true_ts, axis=0)
        y_pred_ts = np.stack(y_pred_ts, axis=0)

    def gather_by_indices(idxs):
        idxs = np.asarray(idxs, dtype=int)
        return (
            y_true_ts[idxs].reshape(-1),
            y_pred_ts[idxs].reshape(-1),
        )

    y_train_true, y_train_pred_t = gather_by_indices(idx_train)
    y_val_true, y_val_pred_t = gather_by_indices(idx_val)
    y_test_true, y_test_pred_t = gather_by_indices(idx_test)

    y_train_pred = inverse_transform_target(y_train_pred_t, target_transform)
    y_val_pred = inverse_transform_target(y_val_pred_t, target_transform)
    y_test_pred = inverse_transform_target(y_test_pred_t, target_transform)

    if clip_pred:
        y_train_pred = np.maximum(y_train_pred, 0.0)
        y_val_pred = np.maximum(y_val_pred, 0.0)
        y_test_pred = np.maximum(y_test_pred, 0.0)

    y_pred_ts_plot = inverse_transform_target(y_pred_ts.reshape(-1), target_transform).reshape(T, N_prod)
    if clip_pred:
        y_pred_ts_plot = np.maximum(y_pred_ts_plot, 0.0)
    y_true_ts_plot = y_true_ts

    mae_train = mae(y_train_true, y_train_pred)
    rmse_train = rmse(y_train_true, y_train_pred)
    mape_train = mape(y_train_true, y_train_pred)
    smape_train = smape(y_train_true, y_train_pred)

    mae_val = mae(y_val_true, y_val_pred)
    rmse_val = rmse(y_val_true, y_val_pred)
    mape_val = mape(y_val_true, y_val_pred)
    smape_val = smape(y_val_true, y_val_pred)

    mae_test = mae(y_test_true, y_test_pred)
    rmse_test = rmse(y_test_true, y_test_pred)
    mape_test = mape(y_test_true, y_test_pred)
    smape_test = smape(y_test_true, y_test_pred)

    tag = f"gnn_hetero5_lag{lag_window}_{temporal_type}_{tag_mode}{'_sp' if use_softplus_output else ''}"

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

    RUN_SUMMARY.append(
        {
            "temporal_type": temporal_type,
            "lag_window": lag_window,
            "horizon": HORIZON,
            "variant": "gnn_hetero5",
            "tag": tag,
            "edge_view": None,
            "target_transform": target_transform,
            "clip_pred": clip_pred,
            "use_softplus_output": use_softplus_output,
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

    # Plot per-product
    days_np = np.array(days)
    node_ids = pkg.get("node_ids", None)
    base_pred_dir = Path(PROC_DIR) / "predictions" / "gnn_baselines"
    plot_dir = base_pred_dir / "plots_per_product" / f"{temporal_type}" / f"lag{lag_window}"
    target_mode_str = tag_mode if not use_softplus_output else f"{tag_mode}_sp"

    plot_gnn_predictions_per_product(
        days=days_np,
        y_true_ts=y_true_ts_plot,
        y_pred_ts=y_pred_ts_plot,
        node_ids=node_ids,
        out_dir=plot_dir,
        temporal_type=temporal_type,
        lag_window=lag_window,
        graph_variant="hetero5",
        target_mode=target_mode_str,
        view=None,
        max_plots=None,
    )


# =========================
# main: chạy 4 biến thể target + plot TS
# =========================

def main():
    global RUN_SUMMARY
    RUN_SUMMARY = []

    temporal_types = TEMPORAL_TYPE

    es_patience = 5
    es_min_delta = 0.0

    # 4 biến thể: raw_noclip, raw_clip, softplus_noclip, softplus_clip
    target_setups = [
        ("raw_noclip",   "raw", False, False),
        ("raw_clip",     "raw", True,  False),
        ("sp_noclip",    "raw", False, True),
        ("sp_clip",      "raw", True,  True),
    ]

    for temporal_type in temporal_types:
        for lag_window in LAG_WINDOWS:
            print(f"\n############ GNN Baselines: temporal={temporal_type}, lag={lag_window} ############")

            # 1) Projected – load, plot TS theo product, train 4 biến thể
            pkg_proj = load_gnn_pkg("projected", temporal_type, lag_window)
            plot_product_time_series(pkg_proj, temporal_type, lag_window, max_products=20)

            for view in PROJECTED_VIEWS:
                for tag_suffix, t_mode, do_clip, use_sp in target_setups:
                    run_projected_gnn_baseline(
                        pkg=pkg_proj,
                        temporal_type=temporal_type,
                        lag_window=lag_window,
                        edge_view=view,
                        device="cuda",
                        epochs=30,
                        batch_days=8,
                        es_patience=es_patience,
                        es_min_delta=es_min_delta,
                        target_transform=t_mode,
                        clip_pred=do_clip,
                        use_softplus_output=use_sp,
                    )

            # 2) Homogeneous 5-type
            pkg_homo5 = load_gnn_pkg("homo5", temporal_type, lag_window)
            for tag_suffix, t_mode, do_clip, use_sp in target_setups:
                run_homo5_gnn_baseline(
                    pkg=pkg_homo5,
                    temporal_type=temporal_type,
                    lag_window=lag_window,
                    device="cuda",
                    epochs=30,
                    batch_days=8,
                    es_patience=es_patience,
                    es_min_delta=es_min_delta,
                    target_transform=t_mode,
                    clip_pred=do_clip,
                    use_softplus_output=use_sp,
                )

            # 3) Heterogeneous 5-type
            pkg_hetero5 = load_gnn_pkg("hetero5", temporal_type, lag_window)
            for tag_suffix, t_mode, do_clip, use_sp in target_setups:
                run_hetero5_gnn_baseline(
                    pkg=pkg_hetero5,
                    temporal_type=temporal_type,
                    lag_window=lag_window,
                    device="cuda",
                    epochs=30,
                    batch_days=8,
                    es_patience=es_patience,
                    es_min_delta=es_min_delta,
                    target_transform=t_mode,
                    clip_pred=do_clip,
                    use_softplus_output=use_sp,
                )

    if RUN_SUMMARY:
        df_sum = pd.DataFrame(RUN_SUMMARY)
        print("\n=== GNN baselines summary ===")
        df_sum = df_sum.sort_values(
            [
                "temporal_type",
                "lag_window",
                "horizon",
                "variant",
                "tag",
                "edge_view",
                "target_transform",
                "use_softplus_output",
            ]
        )
        print(
            df_sum[
                [
                    "temporal_type",
                    "lag_window",
                    "horizon",
                    "variant",
                    "tag",
                    "edge_view",
                    "target_transform",
                    "clip_pred",
                    "use_softplus_output",
                    "MAE_train",
                    "RMSE_train",
                    "MAE_val",
                    "RMSE_val",
                    "MAE_test",
                    "RMSE_test",
                ]
            ]
        )

        out_path = Path(PROC_DIR) / "predictions" / "gnn_baselines" / "summary_gnn_baselines_all_variants.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_sum.to_csv(out_path, index=False)
        print(f"\nSaved GNN baseline summary to {out_path}")


if __name__ == "__main__":
    main()
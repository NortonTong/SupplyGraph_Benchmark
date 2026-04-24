import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

from config.config import PROC_DIR, DEFAULT_EXPERIMENTS
from build_graphs import (
    load_node_metadata,
    load_xgb_tabular_for_gnn,
    build_residual_time_tensors_for_gnn,
    build_projected_edge_indices,
    build_homo5type_from_parquet,
    build_hetero5type_from_parquet,
    make_homo5_flat_edge_index,
    HORIZON,
)

from models_gnn import (
    ProjectedGINRegressor,
    HomogeneousFiveTypeGINRegressor,
    HeterogeneousGINRegressor,
)

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_baseline6_predictions_per_product(
    days_test: np.ndarray,
    node_ids: np.ndarray,
    y_test_true_flat: np.ndarray,
    y_test_pred_flat: np.ndarray,
    out_dir: Path,
    temporal_type: str,
    lag_window: int,
    graph_type: str,
    mode_name: str,
    edge_view: str | None = None,
    max_plots: int | None = None,
) -> None:
    """
    Vẽ y_true vs y_pred theo ngày cho từng product node trên test split
    dành riêng cho baseline 6 (residual XGB + GNN).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # flatten -> DataFrame để group theo node_id
    import pandas as pd
    N = len(node_ids)
    T_test = len(days_test)

    df_plot = pd.DataFrame(
        {
            "date": np.repeat(days_test, N),
            "node_id": np.tile(node_ids, T_test),
            "y_true": np.asarray(y_test_true_flat, dtype=float),
            "y_pred": np.asarray(y_test_pred_flat, dtype=float),
        }
    )

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

        title = f"Baseline 6 - {graph_type}"
        if edge_view is not None:
            title += f" ({edge_view})"
        title += f" [{mode_name}] H{HORIZON}, lag={lag_window}, {temporal_type} - node_id={node}"

        plt.title(title)
        plt.xlabel("Date")
        plt.ylabel("Sales")
        plt.legend()
        plt.tight_layout()

        # tên file
        ev = edge_view if edge_view is not None else "noedgeview"
        fname = out_dir / (
            f"baseline6_{graph_type}_{ev}_h{HORIZON}_lag{lag_window}_{temporal_type}_{mode_name}_node_{node}.png"
        )
        plt.savefig(fname, dpi=150)
        plt.close()

    print(f"[PLOT-BL6] Saved per-product prediction plots to {out_dir}")
# ========= GLOBAL RUN SUMMARY =========

RUN_SUMMARY: list[dict] = []

# ========= Metrics & EarlyStopping =========


def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


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


def get_time_splits(days: np.ndarray, split) -> Tuple[List[int], List[int], List[int]]:
    idx_train = [t for t in range(len(days)) if split[t] == "train"]
    idx_val = [t for t in range(len(days)) if split[t] == "val"]
    idx_test = [t for t in range(len(days)) if split[t] == "test"]
    print(f"[SPLIT] train={len(idx_train)}, val={len(idx_val)}, test={len(idx_test)}")
    return idx_train, idx_val, idx_test


# ========= Shared residual training core =========


def train_residual_projected(
    temporal_type: str,
    lag_window: int,
    edge_view: str,
    is_log1p: bool,
    device: str = "cuda",
    epochs: int = 400,
    batch_days: int = 8,
    es_patience: int = 20,
    es_min_delta: float = 0.001,
):
    global RUN_SUMMARY

    mode_name = "log1p" if is_log1p else "raw"
    print(
        f"\n=== Baseline 6: Residual Projected GNN "
        f"[H{HORIZON}][lag{lag_window}][{temporal_type}][view={edge_view}][{mode_name}] ==="
    )

    df_meta = load_node_metadata()
    df_xgb = load_xgb_tabular_for_gnn(temporal_type, lag_window, HORIZON)

    pred_path = (
        PROC_DIR
        / "baseline"
        / "xgboost"
        / f"xgboost_predictions_h{HORIZON}_lag{lag_window}_{temporal_type}.parquet"
    )
    print(f"[RESID-PROJ] Loading XGBoost predictions from {pred_path}")
    df_pred = pd.read_parquet(pred_path)

    pkg_res, nodeindex2pos_prod = build_residual_time_tensors_for_gnn(
        df_xgb_tabular=df_xgb,
        df_xgb_pred=df_pred,
        is_log1p=is_log1p,
    )

    X_res = pkg_res["X_residual"].float()  # [T, N, F_res]
    R_res = pkg_res["R_residual"].float()  # [T, N]
    days = np.array(pkg_res["days"])
    split = pkg_res["split"]
    node_ids = pkg_res["node_ids_product"]
    N = X_res.size(1)

    edge_index_dict = build_projected_edge_indices(nodeindex2pos_prod, df_meta)
    if edge_view not in edge_index_dict:
        raise ValueError(f"edge_view={edge_view} not in {list(edge_index_dict.keys())}")
    edge_index = edge_index_dict[edge_view]

    idx_train, idx_val, idx_test = get_time_splits(days, split)

    device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    model = ProjectedGINRegressor(
        in_channels=X_res.size(-1),
        hidden_channels=128,
        num_layers=2,
        is_softplus=False,
        is_log1p=False,
    ).to(device)
    edge_index = edge_index.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
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
            X_block = X_res[idx_block].to(device)
            R_block = R_res[idx_block].to(device)

            if X_block.size(0) == 0:
                continue

            loss_block = 0.0
            for b in range(X_block.size(0)):
                x_b = X_block[b]        # [N, F_res]
                r_true_b = R_block[b]   # [N]

                r_pred_b = model(x_b, edge_index)

                mask = torch.isfinite(r_true_b)
                if mask.sum() == 0:
                    continue

                loss_b = loss_fn(r_pred_b[mask], r_true_b[mask])
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
            f"[Residual-Proj-{edge_view}][{mode_name}] Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f}"
        )
        if early_stopper.step(val_loss, model):
            print(f"[Residual-Proj-{edge_view}][{mode_name}] Early stopping at epoch {epoch:03d}")
            break

    early_stopper.load_best(model)
    model.eval()

    with torch.no_grad():
        def predict_on_indices(idxs):
            y_true_list = []
            y_pred_list = []
            for t in idxs:
                x_t = X_res[t].to(device)
                r_true_t = R_res[t].to(device)
                y_xgb_t = X_res[t, :, -1].to(device)

                r_pred_t = model(x_t, edge_index)

                if is_log1p:
                    z_xgb_t = torch.log1p(y_xgb_t.clamp_min(0.0))
                    z_hat_t = z_xgb_t + r_pred_t
                    y_hat_t = torch.expm1(z_hat_t).clamp_min(0.0)
                    y_true_t = torch.expm1(z_xgb_t + r_true_t).clamp_min(0.0)
                else:
                    y_hat_t = (y_xgb_t + r_pred_t).clamp_min(0.0)
                    y_true_t = (y_xgb_t + r_true_t).clamp_min(0.0)

                y_true_list.append(y_true_t.cpu().numpy())
                y_pred_list.append(y_hat_t.cpu().numpy())
            return np.concatenate(y_true_list), np.concatenate(y_pred_list)

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

    tag = f"baseline6_residual_proj_{edge_view}_h{HORIZON}_lag{lag_window}_{temporal_type}_{mode_name}"

    print(f"\n[Residual-Proj-{edge_view}][{tag}] Train:")
    print(f"  MAE  : {mae_train:.4f}")
    print(f"  RMSE : {rmse_train:.4f}")
    print(f"  MAPE : {mape_train:.4f}")
    print(f"  sMAPE: {smape_train:.4f}")

    print(f"\n[Residual-Proj-{edge_view}][{tag}] Val:")
    print(f"  MAE  : {mae_val:.4f}")
    print(f"  RMSE : {rmse_val:.4f}")
    print(f"  MAPE : {mape_val:.4f}")
    print(f"  sMAPE: {smape_val:.4f}")

    print(f"\n[Residual-Proj-{edge_view}][{tag}] Test:")
    print(f"  MAE  : {mae_test:.4f}")
    print(f"  RMSE : {rmse_test:.4f}")
    print(f"  MAPE : {mape_test:.4f}")
    print(f"  sMAPE: {smape_test:.4f}")

    out_dir = (
        Path(PROC_DIR) / "predictions" / "baseline_6" / f"{temporal_type}_{mode_name}" / "projected"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    df_test_pred = pd.DataFrame(
        {
            "date": np.repeat(days_test, N),
            "node_id": np.tile(node_ids, T_test),
            "y_true": y_test_true_flat,
            "y_pred": y_test_pred_flat,
        }
    )
    out_pred_file = out_dir / f"{tag}_test_predictions.csv"
    df_test_pred.to_csv(out_pred_file, index=False)
    print(f"[SAVE] Baseline 6 projected test predictions -> {out_pred_file}")
    # plots per product cho baseline 6 - projected
    out_plot_dir = out_dir / "plots_per_product"
    plot_baseline6_predictions_per_product(
        days_test=days_test,
        node_ids=node_ids,
        y_test_true_flat=y_test_true_flat,
        y_test_pred_flat=y_test_pred_flat,
        out_dir=out_plot_dir,
        temporal_type=temporal_type,
        lag_window=lag_window,
        graph_type=f"projected_{edge_view}",
        mode_name=mode_name,
        edge_view=edge_view,
        max_plots=None,  # hoặc giới hạn số node muốn vẽ
    )
    # summary
    RUN_SUMMARY.append(
        {
            "temporal_type": temporal_type,
            "lag_window": lag_window,
            "horizon": HORIZON,
            "graph_type": "projected",
            "edge_view": edge_view,
            "variant": "baseline_6_residual",
            "mode": mode_name,
            "tag": tag,
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


def train_residual_homo5(
    temporal_type: str,
    lag_window: int,
    is_log1p: bool,
    device: str = "cuda",
    epochs: int = 400,
    batch_days: int = 8,
    es_patience: int = 20,
    es_min_delta: float = 0.001,
):
    global RUN_SUMMARY

    mode_name = "log1p" if is_log1p else "raw"
    print(
        f"\n=== Baseline 6: Residual Homo5 GNN "
        f"[H{HORIZON}][lag{lag_window}][{temporal_type}][{mode_name}] ==="
    )

    df_meta = load_node_metadata()
    df_xgb = load_xgb_tabular_for_gnn(temporal_type, lag_window, HORIZON)

    pred_path = (
        PROC_DIR
        / "baseline"
        / "xgboost"
        / f"xgboost_predictions_h{HORIZON}_lag{lag_window}_{temporal_type}.parquet"
    )
    print(f"[RESID-HOMO5] Loading XGBoost predictions from {pred_path}")
    df_pred = pd.read_parquet(pred_path)

    pkg_res, nodeindex2pos_prod = build_residual_time_tensors_for_gnn(
        df_xgb_tabular=df_xgb,
        df_xgb_pred=df_pred,
        is_log1p=is_log1p,
    )

    X_res = pkg_res["X_residual"].float()  # [T, N_prod, F_res]
    R_res = pkg_res["R_residual"].float()  # [T, N_prod]
    days = np.array(pkg_res["days"])
    split = pkg_res["split"]
    node_ids = pkg_res["node_ids_product"]
    N_prod = X_res.size(1)
    F_res = X_res.size(-1)

    idx_train, idx_val, idx_test = get_time_splits(days, split)

    # Build homo5 graph
    edge_index_homo5, num_nodes_homo5, nodes_homo_tbl = build_homo5type_from_parquet()
    node_type_order = sorted(nodes_homo_tbl["node_type"].unique().tolist())
    edge_index_flat = make_homo5_flat_edge_index(
        edge_index_dict=edge_index_homo5,
        num_nodes_dict=num_nodes_homo5,
        node_type_order=node_type_order,
    )

    # Map product positions (0..N_prod-1) to global homo index
    df_prod_nodes = nodes_homo_tbl[nodes_homo_tbl["node_type"] == "product"].copy()
    df_prod_nodes = df_prod_nodes.sort_values("node_index")  # node_index khớp với df_xgb
    nodeindex2local = {int(r["node_index"]): int(i) for i, (_, r) in enumerate(df_prod_nodes.iterrows())}
    offsets = {}
    offset = 0
    for nt in node_type_order:
        offsets[nt] = offset
        offset += num_nodes_homo5[nt]
    product_offset = offsets["product"]

    # Check: thứ tự node_index_product phải align với df_prod_nodes
    node_index_product = pkg_res["node_index_product"].numpy().astype(int)
    prod_global_idx = np.zeros_like(node_index_product, dtype=int)
    for i, idx in enumerate(node_index_product):
        local = nodeindex2local[int(idx)]
        prod_global_idx[i] = product_offset + local
    device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"

    # sau khi build edge_index_flat
    edge_index_flat = edge_index_flat.to(device)

    model = HomogeneousFiveTypeGINRegressor(
        in_channels=F_res,
        num_nodes_dict=num_nodes_homo5,
        node_type_order=node_type_order,
        hidden_channels=128,
        num_layers=2,
        node_type_emb_dim=8,
        is_softplus=False,
        is_log1p=False,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
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
            X_block = X_res[idx_block].to(device)
            R_block = R_res[idx_block].to(device)

            if X_block.size(0) == 0:
                continue

            loss_block = 0.0
            for b in range(X_block.size(0)):
                x_prod_b = X_block[b]   # [N_prod, F_res]
                r_true_b = R_block[b]   # [N_prod]

                x_all = torch.zeros(offset, F_res, device=device)
                x_all[torch.from_numpy(prod_global_idx).to(device)] = x_prod_b

                # build x_dict theo node_type_order
                x_dict = {}
                start = 0
                for nt in node_type_order:
                    n_nt = num_nodes_homo5[nt]
                    x_dict[nt] = x_all[start:start + n_nt]
                    start += n_nt

                r_pred_b = model(x_dict, edge_index_flat)  # [N_prod] logits for 'product'
                mask = torch.isfinite(r_true_b)
                if mask.sum() == 0:
                    continue

                loss_b = loss_fn(r_pred_b[mask], r_true_b[mask])
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
            f"[Residual-Homo5][{mode_name}] Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f}"
        )
        if early_stopper.step(val_loss, model):
            print(f"[Residual-Homo5][{mode_name}] Early stopping at epoch {epoch:03d}")
            break

    early_stopper.load_best(model)
    model.eval()

    with torch.no_grad():
        def predict_on_indices(idxs):
            y_true_list = []
            y_pred_list = []
            prod_global_idx_t = torch.from_numpy(prod_global_idx).to(device)

            for t in idxs:
                x_prod_t = X_res[t].to(device)
                r_true_t = R_res[t].to(device)
                y_xgb_t = X_res[t, :, -1].to(device)

                x_all = torch.zeros(offset, F_res, device=device)
                x_all[prod_global_idx_t] = x_prod_t

                x_dict = {}
                start = 0
                for nt in node_type_order:
                    n_nt = num_nodes_homo5[nt]
                    x_dict[nt] = x_all[start:start + n_nt]
                    start += n_nt

                r_pred_t = model(x_dict, edge_index_flat)  # [N_prod]

                if is_log1p:
                    z_xgb_t = torch.log1p(y_xgb_t.clamp_min(0.0))
                    z_hat_t = z_xgb_t + r_pred_t
                    y_hat_t = torch.expm1(z_hat_t).clamp_min(0.0)
                    y_true_t = torch.expm1(z_xgb_t + r_true_t).clamp_min(0.0)
                else:
                    y_hat_t = (y_xgb_t + r_pred_t).clamp_min(0.0)
                    y_true_t = (y_xgb_t + r_true_t).clamp_min(0.0)

                y_true_list.append(y_true_t.cpu().numpy())
                y_pred_list.append(y_hat_t.cpu().numpy())
            return np.concatenate(y_true_list), np.concatenate(y_pred_list)

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

    tag = f"baseline6_residual_homo5_h{HORIZON}_lag{lag_window}_{temporal_type}_{mode_name}"

    print(f"\n[Residual-Homo5][{tag}] Train:")
    print(f"  MAE  : {mae_train:.4f}")
    print(f"  RMSE : {rmse_train:.4f}")
    print(f"  MAPE : {mape_train:.4f}")
    print(f"  sMAPE: {smape_train:.4f}")

    print(f"\n[Residual-Homo5][{tag}] Val:")
    print(f"  MAE  : {mae_val:.4f}")
    print(f"  RMSE : {rmse_val:.4f}")
    print(f"  MAPE : {mape_val:.4f}")
    print(f"  sMAPE: {smape_val:.4f}")

    print(f"\n[Residual-Homo5][{tag}] Test:")
    print(f"  MAE  : {mae_test:.4f}")
    print(f"  RMSE : {rmse_test:.4f}")
    print(f"  MAPE : {mape_test:.4f}")
    print(f"  sMAPE: {smape_test:.4f}")

    out_dir = (
        Path(PROC_DIR) / "predictions" / "baseline_6" / f"{temporal_type}_{mode_name}" / "homo5"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    df_test_pred = pd.DataFrame(
        {
            "date": np.repeat(days_test, N_prod),
            "node_id": np.tile(node_ids, T_test),
            "y_true": y_test_true_flat,
            "y_pred": y_test_pred_flat,
        }
    )
    out_pred_file = out_dir / f"{tag}_test_predictions.csv"
    df_test_pred.to_csv(out_pred_file, index=False)
    print(f"[SAVE] Baseline 6 homo5 test predictions -> {out_pred_file}")
    # plots per product cho baseline 6 - homo5
    out_plot_dir = out_dir / "plots_per_product"
    plot_baseline6_predictions_per_product(
        days_test=days_test,
        node_ids=node_ids,
        y_test_true_flat=y_test_true_flat,
        y_test_pred_flat=y_test_pred_flat,
        out_dir=out_plot_dir,
        temporal_type=temporal_type,
        lag_window=lag_window,
        graph_type="homo5",
        mode_name=mode_name,
        edge_view=None,
        max_plots=None,
    )
    RUN_SUMMARY.append(
        {
            "temporal_type": temporal_type,
            "lag_window": lag_window,
            "horizon": HORIZON,
            "graph_type": "homo5",
            "edge_view": None,
            "variant": "baseline_6_residual",
            "mode": mode_name,
            "tag": tag,
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


def train_residual_hetero5(
    temporal_type: str,
    lag_window: int,
    is_log1p: bool,
    device: str = "cuda",
    epochs: int = 400,
    batch_days: int = 8,
    es_patience: int = 20,
    es_min_delta: float = 0.001,
):
    global RUN_SUMMARY

    mode_name = "log1p" if is_log1p else "raw"
    print(
        f"\n=== Baseline 6: Residual Hetero5 GNN "
        f"[H{HORIZON}][lag{lag_window}][{temporal_type}][{mode_name}] ==="
    )

    df_meta = load_node_metadata()
    df_xgb = load_xgb_tabular_for_gnn(temporal_type, lag_window, HORIZON)

    pred_path = (
        PROC_DIR
        / "baseline"
        / "xgboost"
        / f"xgboost_predictions_h{HORIZON}_lag{lag_window}_{temporal_type}.parquet"
    )
    print(f"[RESID-HET5] Loading XGBoost predictions from {pred_path}")
    df_pred = pd.read_parquet(pred_path)

    pkg_res, nodeindex2pos_prod = build_residual_time_tensors_for_gnn(
        df_xgb_tabular=df_xgb,
        df_xgb_pred=df_pred,
        is_log1p=is_log1p,
    )

    X_res = pkg_res["X_residual"].float()  # [T, N_prod, F_res]
    R_res = pkg_res["R_residual"].float()
    days = np.array(pkg_res["days"])
    split = pkg_res["split"]
    node_ids = pkg_res["node_ids_product"]
    N_prod = X_res.size(1)
    F_res = X_res.size(-1)

    idx_train, idx_val, idx_test = get_time_splits(days, split)

    # Build hetero5 graph
    edge_index_het5, num_nodes_het5, nodes_het_tbl = build_hetero5type_from_parquet()

    # Build base_x_dict: zeros features cho non-product nodes
    device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    base_x_dict = {}
    for nt, num_nodes in num_nodes_het5.items():
        if nt == "product":
            continue
        base_x_dict[nt] = torch.zeros(num_nodes, F_res, dtype=torch.float32)

    # Map product node_index -> local index in hetero table
    df_prod_nodes = nodes_het_tbl[nodes_het_tbl["node_type"] == "product"].copy()
    df_prod_nodes["node_index"] = df_prod_nodes["node_index"].astype(int)
    df_prod_nodes = df_prod_nodes.sort_values("node_index")
    nodeindex2local = {int(r["node_index"]): int(i) for i, (_, r) in enumerate(df_prod_nodes.iterrows())}

    # X_res index order uses node_index_product sorted
    node_index_product = pkg_res["node_index_product"].numpy().astype(int)
    prod_local_idx = np.zeros_like(node_index_product, dtype=int)
    for i, idx in enumerate(node_index_product):
        prod_local_idx[i] = nodeindex2local[int(idx)]

    in_channels_dict = {
        "edge_types": list(edge_index_het5.keys()),
    }
    for nt in num_nodes_het5.keys():
        in_channels_dict[nt] = F_res  # mọi node-type dùng cùng F_res

    model = HeterogeneousGINRegressor(
        in_channels_dict=in_channels_dict,
        hidden_channels=128,
        num_layers=2,
        is_softplus=False,
        is_log1p=False,
    ).to(device)

    edge_index_dict = {k: v.to(device) for k, v in edge_index_het5.items()}
    base_x_dict = {nt: x.to(device) for nt, x in base_x_dict.items()}
    prod_local_idx_t = torch.from_numpy(prod_local_idx).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
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
            X_block = X_res[idx_block].to(device)
            R_block = R_res[idx_block].to(device)

            if X_block.size(0) == 0:
                continue

            loss_block = 0.0
            for b in range(X_block.size(0)):
                x_prod_b = X_block[b]   # [N_prod, F_res]
                r_true_b = R_block[b]   # [N_prod]

                x_dict = {nt: base_x_dict[nt] for nt in base_x_dict.keys()}
                x_prod_full = torch.zeros(num_nodes_het5["product"], F_res, device=device)
                x_prod_full[prod_local_idx_t] = x_prod_b
                x_dict["product"] = x_prod_full

                r_pred_b = model(x_dict, edge_index_dict)  # [N_prod], đã theo đúng thứ tự prod_local_idx

                mask = torch.isfinite(r_true_b)
                if mask.sum() == 0:
                    continue

                loss_b = loss_fn(r_pred_b[mask], r_true_b[mask])
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
            f"[Residual-Hetero5][{mode_name}] Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f}"
        )
        if early_stopper.step(val_loss, model):
            print(f"[Residual-Hetero5][{mode_name}] Early stopping at epoch {epoch:03d}")
            break

    early_stopper.load_best(model)
    model.eval()

    with torch.no_grad():
        def predict_on_indices(idxs):
            y_true_list = []
            y_pred_list = []

            for t in idxs:
                x_prod_t = X_res[t].to(device)
                r_true_t = R_res[t].to(device)
                y_xgb_t = X_res[t, :, -1].to(device)

                x_dict = {nt: base_x_dict[nt] for nt in base_x_dict.keys()}
                x_prod_full = torch.zeros(num_nodes_het5["product"], F_res, device=device)
                x_prod_full[prod_local_idx_t] = x_prod_t
                x_dict["product"] = x_prod_full

                r_pred_t = model(x_dict, edge_index_dict)  # [N_prod]

                if is_log1p:
                    z_xgb_t = torch.log1p(y_xgb_t.clamp_min(0.0))
                    z_hat_t = z_xgb_t + r_pred_t
                    y_hat_t = torch.expm1(z_hat_t).clamp_min(0.0)
                    y_true_t = torch.expm1(z_xgb_t + r_true_t).clamp_min(0.0)
                else:
                    y_hat_t = (y_xgb_t + r_pred_t).clamp_min(0.0)
                    y_true_t = (y_xgb_t + r_true_t).clamp_min(0.0)

                y_true_list.append(y_true_t.cpu().numpy())
                y_pred_list.append(y_hat_t.cpu().numpy())
            return np.concatenate(y_true_list), np.concatenate(y_pred_list)

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

    tag = f"baseline6_residual_hetero5_h{HORIZON}_lag{lag_window}_{temporal_type}_{mode_name}"

    print(f"\n[Residual-Hetero5][{tag}] Train:")
    print(f"  MAE  : {mae_train:.4f}")
    print(f"  RMSE : {rmse_train:.4f}")
    print(f"  MAPE : {mape_train:.4f}")
    print(f"  sMAPE: {smape_train:.4f}")

    print(f"\n[Residual-Hetero5][{tag}] Val:")
    print(f"  MAE  : {mae_val:.4f}")
    print(f"  RMSE : {rmse_val:.4f}")
    print(f"  MAPE : {mape_val:.4f}")
    print(f"  sMAPE: {smape_val:.4f}")

    print(f"\n[Residual-Hetero5][{tag}] Test:")
    print(f"  MAE  : {mae_test:.4f}")
    print(f"  RMSE : {rmse_test:.4f}")
    print(f"  MAPE : {mape_test:.4f}")
    print(f"  sMAPE: {smape_test:.4f}")

    out_dir = (
        Path(PROC_DIR) / "predictions" / "baseline_6" / f"{temporal_type}_{mode_name}" / "hetero5"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    df_test_pred = pd.DataFrame(
        {
            "date": np.repeat(days_test, N_prod),
            "node_id": np.tile(node_ids, T_test),
            "y_true": y_test_true_flat,
            "y_pred": y_test_pred_flat,
        }
    )
    out_pred_file = out_dir / f"{tag}_test_predictions.csv"
    df_test_pred.to_csv(out_pred_file, index=False)
    print(f"[SAVE] Baseline 6 hetero5 test predictions -> {out_pred_file}")
    # plots per product cho baseline 6 - hetero5
    out_plot_dir = out_dir / "plots_per_product"
    plot_baseline6_predictions_per_product(
        days_test=days_test,
        node_ids=node_ids,
        y_test_true_flat=y_test_true_flat,
        y_test_pred_flat=y_test_pred_flat,
        out_dir=out_plot_dir,
        temporal_type=temporal_type,
        lag_window=lag_window,
        graph_type="hetero5",
        mode_name=mode_name,
        edge_view=None,
        max_plots=None,
    )
    RUN_SUMMARY.append(
        {
            "temporal_type": temporal_type,
            "lag_window": lag_window,
            "horizon": HORIZON,
            "graph_type": "hetero5",
            "edge_view": None,
            "variant": "baseline_6_residual",
            "mode": mode_name,
            "tag": tag,
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


def main():
    global RUN_SUMMARY
    RUN_SUMMARY = []

    device = "cuda"
    epochs = 400
    batch_days = 8
    only_graph = "all"   # "all", "projected", "homo5", "hetero5"

    PROJECTED_VIEWS = ["same_group", "same_subgroup", "same_plant", "same_storage"]

    for exp in DEFAULT_EXPERIMENTS:
        temporal_type = exp.temporal_type

        # quyết định có chạy raw/log1p theo config
        modes: list[bool] = []
        if exp.is_log1p:
            modes.append(True)
        # luôn thêm raw
        modes.append(False)

        for lag_window in exp.lag_windows:
            for is_log1p in modes:
                if only_graph in ["all", "projected"]:
                    for view in PROJECTED_VIEWS:
                        train_residual_projected(
                            temporal_type=temporal_type,
                            lag_window=lag_window,
                            edge_view=view,
                            is_log1p=is_log1p,
                            device=device,
                            epochs=epochs,
                            batch_days=batch_days,
                        )

                if only_graph in ["all", "homo5"]:
                    train_residual_homo5(
                        temporal_type=temporal_type,
                        lag_window=lag_window,
                        is_log1p=is_log1p,
                        device=device,
                        epochs=epochs,
                        batch_days=batch_days,
                    )

                if only_graph in ["all", "hetero5"]:
                    train_residual_hetero5(
                        temporal_type=temporal_type,
                        lag_window=lag_window,
                        is_log1p=is_log1p,
                        device=device,
                        epochs=epochs,
                        batch_days=batch_days,
                    )

    # ====== Summary CSV ======
    if RUN_SUMMARY:
        df_sum = pd.DataFrame(RUN_SUMMARY)
        print("\n=== Baseline 6 (Residual XGB + GNN) summary ===")
        df_sum = df_sum.sort_values(
            [
                "temporal_type",
                "lag_window",
                "horizon",
                "graph_type",
                "edge_view",
                "mode",
                "tag",
            ]
        )

        out_dir = Path(PROC_DIR) / "predictions" / "baseline_6"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "summary_baseline_6_residual_xgb_gnn.csv"
        df_sum.to_csv(out_path, index=False)
        print(f"\nSaved baseline 6 summary to {out_path}")


if __name__ == "__main__":
    main()
# build_advanced_graph_features.py

import numpy as np
import pandas as pd

from config.config import PROC_DIR, DEFAULT_EXPERIMENTS
from data_preprocessing_baselines import (
    load_node_metadata,
)
from build_graphs import (
    build_homo5type_from_parquet,
    build_hetero5type_from_parquet,
)

# =========================
# Helper: mapping & Y[t,i]
# =========================

def build_product_index_mapping(df_base: pd.DataFrame):
    """
    Dùng node_index (int) để mapping product -> vị trí trong ma trận Y.
    """
    df_nodes = (
        df_base[["node_index"]]
        .drop_duplicates()
        .sort_values("node_index")
    )
    node_indices = df_nodes["node_index"].to_numpy()
    idx2pos = {int(idx): i for i, idx in enumerate(node_indices)}
    return node_indices, idx2pos


def build_Y_from_base(df_base: pd.DataFrame, node_indices) -> tuple[np.ndarray, np.ndarray]:
    """
    Y[t, i] = signal dùng cho neighbor aggregation tại day t, product i (theo node_index).

    Ưu tiên:
    - Nếu có cột 'sales_order' thì dùng.
    - Nếu không có, dùng 'target' (file baseline XGB đã rename label thành target).
    """
    df = df_base.sort_values(["day", "node_index"]).copy()
    days = np.sort(df["day"].unique())
    T = len(days)
    N = len(node_indices)
    Y = np.full((T, N), np.nan, dtype=float)

    # chọn cột làm tín hiệu
    if "sales_order" in df.columns:
        value_col = "sales_order"
    elif "y" in df.columns:
        value_col = "y"
    elif "y_h7" in df.columns:
        value_col = "y_h7"
    elif "target" in df.columns:
        value_col = "target"
    else:
        raise KeyError(
            "Không tìm thấy cột giá trị để build Y (sales_order / y / y_h7 / target) "
            f"trong df_base: {df.columns.tolist()}"
        )

    day2idx = {int(d): k for k, d in enumerate(days)}
    idx2pos = {int(n): i for i, n in enumerate(node_indices)}

    for _, r in df.iterrows():
        t = day2idx[int(r["day"])]
        pos = idx2pos[int(r["node_index"])]
        val = r[value_col]
        if pd.isna(val):
            continue
        Y[t, pos] = float(val)

    return Y, days

# =========================
# Neighbor indices: projected
# =========================

def build_neighbor_indices_projected(df_meta: pd.DataFrame, idx2pos: dict):
    """
    Dùng metadata để tạo neighbor list cho 4 projected views
    trên trục node_index (int).
    """
    neighbors = {
        k: [[] for _ in range(len(idx2pos))]
        for k in ["same_group", "same_subgroup", "same_plant", "same_storage"]
    }

    df_prod = df_meta[
        ["node_index", "group", "sub_group", "plant", "storage_location"]
    ].copy()
    df_prod = df_prod[df_prod["node_index"].isin(idx2pos.keys())]

    for col, key in [
        ("group", "same_group"),
        ("sub_group", "same_subgroup"),
        ("plant", "same_plant"),
        ("storage_location", "same_storage"),
    ]:
        for _, sub in df_prod.groupby(col):
            idxs = [idx2pos[int(i)] for i in sub["node_index"]]
            for i in idxs:
                neighbors[key][i].extend(j for j in idxs if j != i)

    for key in neighbors:
        neighbors[key] = [sorted(set(lst)) for lst in neighbors[key]]
    return neighbors


# =========================
# Neighbor indices: homo5
# =========================

def build_neighbor_indices_homo5(edge_index_homo5, nodes_homo_tbl: pd.DataFrame, idx2pos: dict):
    """
    Dùng graph homo5 để xác định neighbors theo group/sub_group/plant/storage.
    """
    nodes = nodes_homo_tbl.copy()
    nodes["node_id"] = nodes["node_id"].astype(str)

    type_groups = {}
    for nt in nodes["node_type"].unique():
        df_nt = nodes[nodes["node_type"] == nt].copy().reset_index(drop=True)
        type_groups[nt] = df_nt

    df_prod = type_groups.get("product", pd.DataFrame()).copy()
    prod_nodeindex2local = {}
    for i, row in df_prod.iterrows():
        node_index = int(row["node_index"])
        prod_nodeindex2local[node_index] = int(i)

    def _neighbors_via_edge_key(rel_key, dst_type):
        ei = edge_index_homo5.get(("product", rel_key, dst_type), None)
        if ei is None or ei.numel() == 0:
            return [[] for _ in range(len(idx2pos))]

        ei = ei.long().cpu().numpy()
        src_local = ei[0]
        dst_local = ei[1]

        type2prods = {}
        for p_loc, t_loc in zip(src_local, dst_local):
            type2prods.setdefault(int(t_loc), []).append(int(p_loc))

        prod_local_neighbors = {}
        for t_loc, prods in type2prods.items():
            prods = sorted(set(prods))
            for i in prods:
                prod_local_neighbors.setdefault(i, set()).update(
                    j for j in prods if j != i
                )

        neighbors = [[] for _ in range(len(idx2pos))]
        for node_index, p_loc in prod_nodeindex2local.items():
            if node_index not in idx2pos:
                continue
            pos_i = idx2pos[node_index]
            neigh_loc = prod_local_neighbors.get(p_loc, set())
            neigh_pos = []
            for j_loc in neigh_loc:
                row_j = df_prod.iloc[j_loc]
                idx_j = int(row_j["node_index"])
                if idx_j in idx2pos and idx_j != node_index:
                    neigh_pos.append(idx2pos[idx_j])
            neighbors[pos_i] = sorted(set(neigh_pos))
        return neighbors

    neighbors_homo = {}
    neighbors_homo["homo_group"] = _neighbors_via_edge_key("product_group_edge", "product_group")
    neighbors_homo["homo_subgroup"] = _neighbors_via_edge_key("product_sub_group_edge", "product_sub_group")
    neighbors_homo["homo_plant"] = _neighbors_via_edge_key("product_plant_edge", "plant")
    neighbors_homo["homo_storage"] = _neighbors_via_edge_key("product_storage_edge", "storage_location")
    return neighbors_homo


# =========================
# Neighbor indices: hetero5
# =========================

def build_neighbor_indices_hetero5(edge_index_het5, nodes_het_tbl: pd.DataFrame, idx2pos: dict):
    """
    Dùng graph hetero5 để xác định neighbors theo group/sub_group/plant/storage.
    """
    nodes = nodes_het_tbl.copy()
    nodes["node_id"] = nodes["node_id"].astype(str)

    type_groups = {}
    for nt in nodes["node_type"].unique():
        df_nt = nodes[nodes["node_type"] == nt].copy().reset_index(drop=True)
        type_groups[nt] = df_nt

    df_prod = type_groups.get("product", pd.DataFrame()).copy()
    prod_nodeindex2local = {}
    for i, row in df_prod.iterrows():
        node_index = int(row["node_index"])
        prod_nodeindex2local[node_index] = int(i)

    def _neighbors_via_edge_type(rel_fwd, dst_type):
        ei = edge_index_het5.get(("product", rel_fwd, dst_type), None)
        if ei is None or ei.numel() == 0:
            return [[] for _ in range(len(idx2pos))]

        ei = ei.long().cpu().numpy()
        src_local = ei[0]
        dst_local = ei[1]

        type2prods = {}
        for p_loc, t_loc in zip(src_local, dst_local):
            type2prods.setdefault(int(t_loc), []).append(int(p_loc))

        prod_local_neighbors = {}
        for t_loc, prods in type2prods.items():
            prods = sorted(set(prods))
            for i in prods:
                prod_local_neighbors.setdefault(i, set()).update(
                    j for j in prods if j != i
                )

        neighbors = [[] for _ in range(len(idx2pos))]
        for node_index, p_loc in prod_nodeindex2local.items():
            if node_index not in idx2pos:
                continue
            pos_i = idx2pos[node_index]
            neigh_loc = prod_local_neighbors.get(p_loc, set())
            neigh_pos = []
            for j_loc in neigh_loc:
                row_j = df_prod.iloc[j_loc]
                idx_j = int(row_j["node_index"])
                if idx_j in idx2pos and idx_j != node_index:
                    neigh_pos.append(idx2pos[idx_j])
            neighbors[pos_i] = sorted(set(neigh_pos))
        return neighbors

    neighbors_het = {}
    neighbors_het["het_group"] = _neighbors_via_edge_type("product_group", "product_group")
    neighbors_het["het_subgroup"] = _neighbors_via_edge_type("product_subgroup", "product_sub_group")
    neighbors_het["het_plant"] = _neighbors_via_edge_type("product_plant", "plant")
    neighbors_het["het_storage"] = _neighbors_via_edge_type("product_storage", "storage_location")
    return neighbors_het


# =========================
# Neighbor-based features
# =========================

def neighbor_mean_lag(Y, neighbor_idx, lag: int):
    T, N = Y.shape
    feat = np.full((T, N), np.nan, dtype=float)
    for i in range(N):
        neigh = neighbor_idx[i]
        if not neigh:
            continue
        for t in range(lag, T):
            vals = Y[t - lag, neigh]
            if np.isfinite(vals).any():
                feat[t, i] = np.nanmean(vals)
    return feat


def neighbor_sum_lag(Y, neighbor_idx, lag: int):
    T, N = Y.shape
    feat = np.full((T, N), np.nan, dtype=float)
    for i in range(N):
        neigh = neighbor_idx[i]
        if not neigh:
            continue
        for t in range(lag, T):
            vals = Y[t - lag, neigh]
            if np.isfinite(vals).any():
                feat[t, i] = np.nansum(vals)
    return feat


def neighbor_max_lag(Y, neighbor_idx, lag: int):
    T, N = Y.shape
    feat = np.full((T, N), np.nan, dtype=float)
    for i in range(N):
        neigh = neighbor_idx[i]
        if not neigh:
            continue
        for t in range(lag, T):
            vals = Y[t - lag, neigh]
            if np.isfinite(vals).any():
                feat[t, i] = np.nanmax(vals)
    return feat


def neighbor_min_lag(Y, neighbor_idx, lag: int):
    T, N = Y.shape
    feat = np.full((T, N), np.nan, dtype=float)
    for i in range(N):
        neigh = neighbor_idx[i]
        if not neigh:
            continue
        for t in range(lag, T):
            vals = Y[t - lag, neigh]
            if np.isfinite(vals).any():
                feat[t, i] = np.nanmin(vals)
    return feat


def neighbor_zero_ratio_window(Y, neighbor_idx, window: int):
    T, N = Y.shape
    feat = np.full((T, N), np.nan, dtype=float)
    if window <= 0:
        return feat
    for i in range(N):
        neigh = neighbor_idx[i]
        if not neigh:
            continue
        for t in range(window - 1, T):
            vals = Y[t - window + 1 : t + 1, :][:, neigh].reshape(-1)
            if vals.size == 0:
                continue
            feat[t, i] = float(np.mean(vals == 0))
    return feat


# =========================
# 3 builders: thêm advanced feature vào df_base đã OHE
# =========================

def build_xgb_with_proj_features(df_base: pd.DataFrame,
                                 temporal_type: str,
                                 horizon: int,
                                 lag_window: int) -> pd.DataFrame:
    node_indices, idx2pos = build_product_index_mapping(df_base)
    Y, days = build_Y_from_base(df_base, node_indices)

    df_meta = load_node_metadata()
    neighbors_proj = build_neighbor_indices_projected(df_meta, idx2pos)

    feats = {}
    lags = [1, lag_window]
    win_zero = lag_window

    for view in ["same_group", "same_subgroup", "same_plant", "same_storage"]:
        neigh = neighbors_proj[view]
        for L in lags:
            feats[f"adv_proj_{view}_mean_lag{L}"] = neighbor_mean_lag(Y, neigh, L)
            feats[f"adv_proj_{view}_sum_lag{L}"]  = neighbor_sum_lag(Y, neigh, L)
            feats[f"adv_proj_{view}_max_lag{L}"]  = neighbor_max_lag(Y, neigh, L)
            feats[f"adv_proj_{view}_min_lag{L}"]  = neighbor_min_lag(Y, neigh, L)
        feats[f"adv_proj_{view}_zero_ratio_win{win_zero}"] = neighbor_zero_ratio_window(Y, neigh, win_zero)

    records = []
    T, N = Y.shape
    for t_idx, day in enumerate(days):
        for i, node_idx in enumerate(node_indices):
            rec = {
                "day": int(day),
                "node_index": int(node_idx),
            }
            for name, arr in feats.items():
                rec[name] = arr[t_idx, i]
            records.append(rec)
    df_feat = pd.DataFrame(records)

    df_merged = df_base.merge(df_feat, on=["day", "node_index"], how="left")
    return df_merged


def build_xgb_with_homo_features(df_base: pd.DataFrame,
                                 temporal_type: str,
                                 horizon: int,
                                 lag_window: int) -> pd.DataFrame:
    node_indices, idx2pos = build_product_index_mapping(df_base)
    Y, days = build_Y_from_base(df_base, node_indices)

    edge_index_homo5, num_nodes_homo5, nodes_homo_tbl = build_homo5type_from_parquet()
    neighbors_homo = build_neighbor_indices_homo5(edge_index_homo5, nodes_homo_tbl, idx2pos)

    feats = {}
    lags = [1, lag_window]
    win_zero = lag_window

    for view, neigh in neighbors_homo.items():
        for L in lags:
            feats[f"adv_{view}_mean_lag{L}"] = neighbor_mean_lag(Y, neigh, L)
            feats[f"adv_{view}_sum_lag{L}"]  = neighbor_sum_lag(Y, neigh, L)
            feats[f"adv_{view}_max_lag{L}"]  = neighbor_max_lag(Y, neigh, L)
            feats[f"adv_{view}_min_lag{L}"]  = neighbor_min_lag(Y, neigh, L)
        feats[f"adv_{view}_zero_ratio_win{win_zero}"] = neighbor_zero_ratio_window(Y, neigh, win_zero)

    records = []
    T, N = Y.shape
    for t_idx, day in enumerate(days):
        for i, node_idx in enumerate(node_indices):
            rec = {
                "day": int(day),
                "node_index": int(node_idx),
            }
            for name, arr in feats.items():
                rec[name] = arr[t_idx, i]
            records.append(rec)
    df_feat = pd.DataFrame(records)

    df_merged = df_base.merge(df_feat, on=["day", "node_index"], how="left")
    return df_merged


def build_xgb_with_hetero_features(df_base: pd.DataFrame,
                                   temporal_type: str,
                                   horizon: int,
                                   lag_window: int) -> pd.DataFrame:
    node_indices, idx2pos = build_product_index_mapping(df_base)
    Y, days = build_Y_from_base(df_base, node_indices)

    edge_index_het5, num_nodes_het5, nodes_het_tbl = build_hetero5type_from_parquet()
    neighbors_het = build_neighbor_indices_hetero5(edge_index_het5, nodes_het_tbl, idx2pos)

    feats = {}
    lags = [1, lag_window]
    win_zero = lag_window

    for view, neigh in neighbors_het.items():
        for L in lags:
            feats[f"adv_{view}_mean_lag{L}"] = neighbor_mean_lag(Y, neigh, L)
            feats[f"adv_{view}_sum_lag{L}"]  = neighbor_sum_lag(Y, neigh, L)
            feats[f"adv_{view}_max_lag{L}"]  = neighbor_max_lag(Y, neigh, L)
            feats[f"adv_{view}_min_lag{L}"]  = neighbor_min_lag(Y, neigh, L)
        feats[f"adv_{view}_zero_ratio_win{win_zero}"] = neighbor_zero_ratio_window(Y, neigh, win_zero)

    records = []
    T, N = Y.shape
    for t_idx, day in enumerate(days):
        for i, node_idx in enumerate(node_indices):
            rec = {
                "day": int(day),
                "node_index": int(node_idx),
            }
            for name, arr in feats.items():
                rec[name] = arr[t_idx, i]
            records.append(rec)
    df_feat = pd.DataFrame(records)

    df_merged = df_base.merge(df_feat, on=["day", "node_index"], how="left")
    return df_merged


# =========================
# main: dùng DEFAULT_EXPERIMENTS + baseline graph
# =========================

def main():
    base_graph_dir = PROC_DIR / "baseline" / "xgb_graph"
    out_dir = PROC_DIR / "baseline" / "xgboost"
    out_dir.mkdir(parents=True, exist_ok=True)

    for exp in DEFAULT_EXPERIMENTS:
        t_type = exp.temporal_type
        for H in exp.horizons:
            for L in exp.lag_windows:
                print(f"\n=== Advanced graph features: temporal_type={t_type}, H={H}, L={L} ===")

                # 1) projected
                path_proj = base_graph_dir / f"xgboost_tabular_graph_projected_h{H}_lag{L}_{t_type}.parquet"
                if path_proj.exists():
                    df_proj = pd.read_parquet(path_proj)
                    print(f"[ADV-GRAPH] Load projected baseline from {path_proj}, shape={df_proj.shape}")
                    df_proj_adv = build_xgb_with_proj_features(
                        df_base=df_proj,
                        temporal_type=t_type,
                        horizon=H,
                        lag_window=L,
                    )
                    out_proj = out_dir / f"xgboost_tabular_graphfeat_projected_h{H}_lag{L}_{t_type}.parquet"
                    df_proj_adv.to_parquet(out_proj, index=False)
                    print(f"Saved projected + advanced features to {out_proj}")
                else:
                    print(f"[ADV-GRAPH] projected baseline not found: {path_proj}")

                # 2) homo5
                path_homo = base_graph_dir / f"xgboost_tabular_graph_homo5_h{H}_lag{L}_{t_type}.parquet"
                if path_homo.exists():
                    df_homo = pd.read_parquet(path_homo)
                    print(f"[ADV-GRAPH] Load homo5 baseline from {path_homo}, shape={df_homo.shape}")
                    df_homo_adv = build_xgb_with_homo_features(
                        df_base=df_homo,
                        temporal_type=t_type,
                        horizon=H,
                        lag_window=L,
                    )
                    out_homo = out_dir / f"xgboost_tabular_graphfeat_homo5_h{H}_lag{L}_{t_type}.parquet"
                    df_homo_adv.to_parquet(out_homo, index=False)
                    print(f"Saved homo5 + advanced features to {out_homo}")
                else:
                    print(f"[ADV-GRAPH] homo5 baseline not found: {path_homo}")

                # 3) hetero5
                path_het = base_graph_dir / f"xgboost_tabular_graph_hetero5_h{H}_lag{L}_{t_type}.parquet"
                if path_het.exists():
                    df_het = pd.read_parquet(path_het)
                    print(f"[ADV-GRAPH] Load hetero5 baseline from {path_het}, shape={df_het.shape}")
                    df_het_adv = build_xgb_with_hetero_features(
                        df_base=df_het,
                        temporal_type=t_type,
                        horizon=H,
                        lag_window=L,
                    )
                    out_het = out_dir / f"xgboost_tabular_graphfeat_hetero5_h{H}_lag{L}_{t_type}.parquet"
                    df_het_adv.to_parquet(out_het, index=False)
                    print(f"Saved hetero5 + advanced features to {out_het}")
                else:
                    print(f"[ADV-GRAPH] hetero5 baseline not found: {path_het}")


if __name__ == "__main__":
    main()
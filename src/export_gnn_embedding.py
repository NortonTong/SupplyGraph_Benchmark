import pandas as pd
import torch
from pathlib import Path

from config.config import PROC_DIR, DEFAULT_EXPERIMENTS
from models_gnn_encoder import (
    ProjectedGINEncoder,
    HomogeneousFiveTypeGINEncoder,
    HeterogeneousGINEncoder,
)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 0)

GNN_DIR = PROC_DIR / "gnn"
GNN_DIR.mkdir(parents=True, exist_ok=True)

EMB_DIR = PROC_DIR / "gnn_embeddings"
EMB_DIR.mkdir(parents=True, exist_ok=True)

XGB_GNN_EMBED_DIR = PROC_DIR / "baseline" / "xgb_gnn_embed"
XGB_GNN_EMBED_DIR.mkdir(parents=True, exist_ok=True)

HORIZON = 7


def get_experiment_params():
    temporal_types = sorted({exp.temporal_type for exp in DEFAULT_EXPERIMENTS})
    horizons = sorted({h for exp in DEFAULT_EXPERIMENTS for h in exp.horizons})
    lag_windows = sorted({L for exp in DEFAULT_EXPERIMENTS for L in exp.lag_windows})
    if len(horizons) != 1:
        raise ValueError(
            f"[export_gnn_embeddings] Expected a single horizon, got {horizons}"
        )
    return temporal_types, horizons[0], lag_windows


TEMPORAL_TYPES, H, LAG_WINDOWS = get_experiment_params()


# ============================================================
# 1. EXPORT EMBEDDINGS
# ============================================================

PROJECTED_VIEWS = ["same_group", "same_subgroup", "same_plant", "same_storage"]


def export_projected_embeddings_for_config(
    temporal_type: str,
    lag_window: int,
    device: str = "cuda",
) -> None:
    """
    Đọc gnn_projected_h{H}_lag{L}_{temporal_type}.pt
    → với mỗi view trong PROJECTED_VIEWS, chạy encoder, lưu embedding dạng long:
      (day, node_index_pos, split, view, emb_0..d-1)
    """
    pkg_path = GNN_DIR / f"gnn_projected_h{H}_lag{lag_window}_{temporal_type}.pt"
    if not pkg_path.exists():
        print(f"[EXPORT-PROJ] {pkg_path} not found, skip.")
        return

    print(f"[EXPORT-PROJ] Loading package from {pkg_path}")
    pkg = torch.load(pkg_path, map_location=device, weights_only = False)

    X_product = pkg["X_product"]          # [T, N_prod, F_in]
    days = pkg["days"]                    # tensor [T]
    split = pkg["split"]                  # list[str]
    edge_index_dict = pkg["edge_index_dict"]

    T, N_prod, F_in = X_product.shape

    encoder = ProjectedGINEncoder(
        in_channels=F_in,
        hidden_channels=128,
        num_layers=3,
    ).to(device)
    encoder.eval()

    rows = []
    with torch.no_grad():
        for view_name in PROJECTED_VIEWS:
            if view_name not in edge_index_dict:
                print(
                    f"[EXPORT-PROJ] view {view_name} not in edge_index_dict, "
                    f"keys={list(edge_index_dict.keys())}, skip this view."
                )
                continue

            edge_index = edge_index_dict[view_name].to(device)

            for t in range(T):
                x_t = X_product[t].to(device)
                h_t = encoder(x_t, edge_index)   # [N_prod, d]
                h_np = h_t.cpu().numpy()
                day_t = int(days[t])
                split_t = split[t]
                for node_pos in range(N_prod):
                    rows.append(
                        {
                            "node_index_pos": node_pos,
                            "day": day_t,
                            "split": split_t,
                            "view": view_name,
                            **{
                                f"emb_{k}": float(h_np[node_pos, k])
                                for k in range(h_np.shape[1])
                            },
                        }
                    )

    if not rows:
        print(
            f"[EXPORT-PROJ] No embeddings exported for temporal_type={temporal_type}, "
            f"lag={lag_window}"
        )
        return

    df_emb = pd.DataFrame(rows)
    out_path = (
        EMB_DIR
        / f"gnn_projected_emb_4views_h{H}_lag{lag_window}_{temporal_type}.parquet"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_emb.to_parquet(out_path, index=False)
    print(f"[EXPORT-PROJ] Saved 4-view projected embeddings to {out_path}")

def export_homo5_embeddings_for_config(
    temporal_type: str,
    lag_window: int,
    device: str = "cuda",
) -> None:
    """
    Đọc gnn_homo5_h{H}_lag{L}_{temporal_type}.pt
    → chạy HomogeneousFiveTypeGINEncoder với edge_index flatten (pkg['edge_index']).
    """
    pkg_path = GNN_DIR / f"gnn_homo5_h{H}_lag{lag_window}_{temporal_type}.pt"
    if not pkg_path.exists():
        print(f"[EXPORT-HOMO5] {pkg_path} not found, skip.")
        return

    print(f"[EXPORT-HOMO5] Loading package from {pkg_path}")
    pkg = torch.load(pkg_path, map_location=device, weights_only = False)

    if "edge_index" not in pkg:
        print(
            "[EXPORT-HOMO5] pkg['edge_index'] not found. "
            "Hãy sửa build_gnn_datasets_for_config để lưu thêm edge_index flatten "
            "vào gnn_homo5_*.pt (gộp tất cả edge types), rồi chạy lại build_graphs.py."
        )
        return

    X_product = pkg["X_product"]          # [T, N_prod, F_in_prod]
    days = pkg["days"]                    # tensor [T]
    split = pkg["split"]                  # list[str]
    edge_index = pkg["edge_index"]        # [2, E_total]
    num_nodes_dict = pkg["num_nodes_dict"]
    nodes_tbl = pkg["nodes_homo_table"]   # DataFrame nodes_homogeneous_5type

    node_type_order = nodes_tbl["node_type"].unique().tolist()
    T, N_prod, F_in = X_product.shape

    encoder = HomogeneousFiveTypeGINEncoder(
        in_channels=F_in,
        num_nodes_dict=num_nodes_dict,
        node_type_order=node_type_order,
        hidden_channels=128,
        num_layers=3,
        node_type_emb_dim=8,
    ).to(device)
    encoder.eval()

    rows = []
    edge_index = edge_index.to(device)
    with torch.no_grad():
        for t in range(T):
            # build x_dict: product có feature, node type khác = 0
            x_dict = {}
            for nt in node_type_order:
                n_type = num_nodes_dict[nt]
                if nt == "product":
                    x_type = torch.zeros((n_type, F_in), dtype=torch.float32)
                    x_type[:N_prod, :] = X_product[t]
                    x_dict[nt] = x_type
                else:
                    x_dict[nt] = torch.zeros((n_type, F_in), dtype=torch.float32)

            for nt in x_dict:
                x_dict[nt] = x_dict[nt].to(device)

            h_prod = encoder(x_dict, edge_index)  # [N_prod, hidden]
            h_np = h_prod.cpu().numpy()
            day_t = int(days[t])
            split_t = split[t]
            for node_pos in range(N_prod):
                rows.append(
                    {
                        "node_index_pos": node_pos,
                        "day": day_t,
                        "split": split_t,
                        **{
                            f"emb_{k}": float(h_np[node_pos, k])
                            for k in range(h_np.shape[1])
                        },
                    }
                )

    if not rows:
        print(
            f"[EXPORT-HOMO5] No embeddings exported for temporal_type={temporal_type}, "
            f"lag={lag_window}"
        )
        return

    df_emb = pd.DataFrame(rows)
    out_path = EMB_DIR / f"gnn_homo5_emb_h{H}_lag{lag_window}_{temporal_type}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_emb.to_parquet(out_path, index=False)
    print(f"[EXPORT-HOMO5] Saved homo5 embeddings to {out_path}")
def export_hetero5_embeddings_for_config(
    temporal_type: str,
    lag_window: int,
    device: str = "cuda",
) -> None:
    """
    Đọc gnn_hetero5_h{H}_lag{L}_{temporal_type}.pt
    → chạy HeterogeneousGINEncoder → embedding cho product nodes.
    """
    pkg_path = GNN_DIR / f"gnn_hetero5_h{H}_lag{lag_window}_{temporal_type}.pt"
    if not pkg_path.exists():
        print(f"[EXPORT-HET5] {pkg_path} not found, skip.")
        return

    print(f"[EXPORT-HET5] Loading package from {pkg_path}")
    pkg = torch.load(pkg_path, map_location=device, weights_only = False)

    X_product = pkg["X_product"]          # [T, N_prod, F_in_prod]
    days = pkg["days"]                    # tensor [T]
    split = pkg["split"]                  # list[str]
    edge_index_dict = pkg["edge_index_dict"]
    num_nodes_dict = pkg["num_nodes_dict"]
    nodes_tbl = pkg["nodes_hetero_table"]

    node_types = nodes_tbl["node_type"].unique().tolist()
    edge_types = list(edge_index_dict.keys())
    F_in = X_product.shape[2]

    in_channels_dict = {"edge_types": edge_types}
    for nt in node_types:
        in_channels_dict[nt] = F_in

    encoder = HeterogeneousGINEncoder(
        in_channels_dict=in_channels_dict,
        hidden_channels=128,
        num_layers=2,
    ).to(device)
    encoder.eval()

    T, N_prod, _ = X_product.shape

    rows = []
    with torch.no_grad():
        for t in range(T):
            x_dict = {}
            for nt in node_types:
                n_type = num_nodes_dict[nt]
                if nt == "product":
                    x_type = torch.zeros((n_type, F_in), dtype=torch.float32)
                    x_type[:N_prod, :] = X_product[t]
                    x_dict[nt] = x_type
                else:
                    x_dict[nt] = torch.zeros((n_type, F_in), dtype=torch.float32)

            for nt in x_dict:
                x_dict[nt] = x_dict[nt].to(device)

            h_prod = encoder(x_dict, edge_index_dict)   # [N_prod, hidden]
            h_np = h_prod.cpu().numpy()
            day_t = int(days[t])
            split_t = split[t]
            for node_pos in range(N_prod):
                rows.append(
                    {
                        "node_index_pos": node_pos,
                        "day": day_t,
                        "split": split_t,
                        **{
                            f"emb_{k}": float(h_np[node_pos, k])
                            for k in range(h_np.shape[1])
                        },
                    }
                )

    df_emb = pd.DataFrame(rows)
    out_path = EMB_DIR / f"gnn_hetero5_emb_h{H}_lag{lag_window}_{temporal_type}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_emb.to_parquet(out_path, index=False)
    print(f"[EXPORT-HET5] Saved hetero5 embeddings to {out_path}")


# ============================================================
# 2. BUILD XGB TABULAR + EMBEDDING
# ============================================================

def build_xgb_tabular_with_gnn_embed_projected(
    temporal_type: str,
    lag_window: int,
):
    """
    base_full_{temporal_type}.parquet + 4-view projected embeddings
    → concat embedding theo feature: emb_same_group_*, emb_same_subgroup_*, ...
    """
    base_full_path = (
        PROC_DIR
        / "baseline"
        / "xgboost"
        / f"xgboost_tabular_h{H}_lag{lag_window}_{temporal_type}.parquet"
    )
    emb_path = (
        EMB_DIR
        / f"gnn_projected_emb_4views_h{H}_lag{lag_window}_{temporal_type}.parquet"
    )

    if not base_full_path.exists():
        print(f"[XGB-GNN-EMB-PROJ] base_full not found: {base_full_path}, skip.")
        return
    if not emb_path.exists():
        print(f"[XGB-GNN-EMB-PROJ] embedding (4 views) not found: {emb_path}, skip.")
        return

    df_base = pd.read_parquet(base_full_path)
    df_emb_long = pd.read_parquet(emb_path)

    # node_index_pos chính là vị trí trong tensor; mapping pos -> node_index
    # giả sử node_index_product là sorted và align với pos. Nếu anh đã lưu
    # node_index_product trong pkg thì ở bước export có thể join để có node_index luôn.
    df_emb_long = df_emb_long.rename(columns={"node_index_pos": "node_index"})
    df_emb_long["node_index"] = df_emb_long["node_index"].astype(int)

    # Pivot theo view: mỗi view trở thành 1 nhóm cột emb_{view}_{k}
    emb_cols = [c for c in df_emb_long.columns if c.startswith("emb_")]

    # tách phần index
    key_cols = ["node_index", "day", "split", "view"]
    df_pivot_src = df_emb_long[key_cols + emb_cols].copy()

    # MultiIndex pivot: index=(node_index, day, split), columns=(view, emb_k)
    df_pivot = (
        df_pivot_src
        .set_index(["node_index", "day", "split", "view"])
        .unstack("view")
    )

    # flatten MultiIndex columns → emb_{view}_{k}
    df_pivot.columns = [
        f"{col_emb}_{view}"
        for col_emb, view in df_pivot.columns.to_flat_index()
    ]
    df_pivot = df_pivot.reset_index()

    # Merge với base_full
    df = df_base.merge(
        df_pivot,
        on=["node_index", "day", "split"],
        how="inner",
    )

    feature_cols = [
        c
        for c in df.columns
        if (
            "lag" in c
            or "roll" in c
            or c in ["day_of_week", "is_weekend", "month", "day_of_month"]
            or c in ["group", "sub_group", "plant", "storage_location"]
            or c.startswith("emb_")
        )
    ]
    base_cols = ["node_id", "node_index", "date", "day", "split"]
    df_out = df[base_cols + feature_cols + ["target"]]

    out_path = (
        XGB_GNN_EMBED_DIR
        / f"xgboost_tabular_gnnembed_projected4view_h{H}_lag{lag_window}_{temporal_type}.parquet"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(out_path, index=False)
    print(
        f"[XGB-GNN-EMB-PROJ] Saved XGB+GNN-embed (4 views concat) tabular to {out_path}"
    )

def build_xgb_tabular_with_gnn_embed_homo5(
    temporal_type: str,
    lag_window: int,
):
    base_full_path = (
        PROC_DIR
        / "baseline"
        / "xgboost"
        / f"xgboost_tabular_h{H}_lag{lag_window}_{temporal_type}.parquet"
    )
    emb_path = EMB_DIR / f"gnn_homo5_emb_h{H}_lag{lag_window}_{temporal_type}.parquet"

    if not base_full_path.exists():
        print(f"[XGB-GNN-EMB-HOMO5] base_full not found: {base_full_path}, skip.")
        return
    if not emb_path.exists():
        print(f"[XGB-GNN-EMB-HOMO5] embedding not found: {emb_path}, skip.")
        return

    df_base = pd.read_parquet(base_full_path)
    df_emb = pd.read_parquet(emb_path)

    df_emb = df_emb.rename(columns={"node_index_pos": "node_index"})
    df_emb["node_index"] = df_emb["node_index"].astype(int)

    df = df_base.merge(
        df_emb,
        on=["node_index", "day", "split"],
        how="inner",
    )

    feature_cols = [
        c
        for c in df.columns
        if (
            "lag" in c
            or "roll" in c
            or c in ["day_of_week", "is_weekend", "month", "day_of_month"]
            or c in ["group", "sub_group", "plant", "storage_location"]
            or c.startswith("emb_")
        )
    ]
    base_cols = ["node_id", "node_index", "date", "day", "split"]
    df_out = df[base_cols + feature_cols + ["target"]]

    out_path = (
        XGB_GNN_EMBED_DIR
        / f"xgboost_tabular_gnnembed_homo5_h{H}_lag{lag_window}_{temporal_type}.parquet"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(out_path, index=False)
    print(
        f"[XGB-GNN-EMB-HOMO5] Saved XGB+GNN-embed tabular to {out_path}"
    )


def build_xgb_tabular_with_gnn_embed_hetero5(
    temporal_type: str,
    lag_window: int,
):
    base_full_path = (
        PROC_DIR
        / "baseline"
        / "xgboost"
        / f"xgboost_tabular_h{H}_lag{lag_window}_{temporal_type}.parquet"
    )
    emb_path = EMB_DIR / f"gnn_hetero5_emb_h{H}_lag{lag_window}_{temporal_type}.parquet"

    if not base_full_path.exists():
        print(f"[XGB-GNN-EMB-HET5] base_full not found: {base_full_path}, skip.")
        return
    if not emb_path.exists():
        print(f"[XGB-GNN-EMB-HET5] embedding not found: {emb_path}, skip.")
        return

    df_base = pd.read_parquet(base_full_path)
    df_emb = pd.read_parquet(emb_path)

    df_emb = df_emb.rename(columns={"node_index_pos": "node_index"})
    df_emb["node_index"] = df_emb["node_index"].astype(int)

    df = df_base.merge(
        df_emb,
        on=["node_index", "day", "split"],
        how="inner",
    )

    feature_cols = [
        c
        for c in df.columns
        if (
            "lag" in c
            or "roll" in c
            or c in ["day_of_week", "is_weekend", "month", "day_of_month"]
            or c in ["group", "sub_group", "plant", "storage_location"]
            or c.startswith("emb_")
        )
    ]
    base_cols = ["node_id", "node_index", "date", "day", "split"]
    df_out = df[base_cols + feature_cols + ["target"]]

    out_path = (
        XGB_GNN_EMBED_DIR
        / f"xgboost_tabular_gnnembed_hetero5_h{H}_lag{lag_window}_{temporal_type}.parquet"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(out_path, index=False)
    print(
        f"[XGB-GNN-EMB-HET5] Saved XGB+GNN-embed tabular to {out_path}"
    )


# ============================================================
# 3. MAIN: FULL CHO TẤT CẢ BIẾN THỂ
# ============================================================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for temporal_type in TEMPORAL_TYPES:
        for lag_window in LAG_WINDOWS:
            print(
                f"\n=== EXPORT GNN EMBEDDINGS + BUILD XGB DATASET "
                f"H{H}, lag={lag_window}, temporal_type={temporal_type} ==="
            )

            # 1) Export embeddings cho 3 graph modes
            export_projected_embeddings_for_config(
                temporal_type=temporal_type,
                lag_window=lag_window,
                device=device,
            )
            export_homo5_embeddings_for_config(
                temporal_type=temporal_type,
                lag_window=lag_window,
                device=device,
            )
            export_hetero5_embeddings_for_config(
                temporal_type=temporal_type,
                lag_window=lag_window,
                device=device,
            )

            build_xgb_tabular_with_gnn_embed_projected(
                temporal_type=temporal_type,
                lag_window=lag_window,
            )
            build_xgb_tabular_with_gnn_embed_homo5(
                temporal_type=temporal_type,
                lag_window=lag_window,
            )
            build_xgb_tabular_with_gnn_embed_hetero5(
                temporal_type=temporal_type,
                lag_window=lag_window,
            )

    print(
        "\n[export_gnn_embeddings] Done exporting all embeddings and building all XGB datasets."
    )


if __name__ == "__main__":
    main()
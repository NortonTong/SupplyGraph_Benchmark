import pandas as pd
import torch
from pathlib import Path
from config.config import PROC_DIR, DEFAULT_EXPERIMENTS
from models_gnn_encoder import (
    ProjectedGINEncoder,
    HomogeneousFiveTypeGINEncoder,
    HeterogeneousGINEncoder,
)
from typing import Literal
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 0)

GNN_DIR = PROC_DIR / "gnn"
GNN_DIR.mkdir(parents=True, exist_ok=True)
GNN_TRAINED_DIR = GNN_DIR / "trained"
GNN_TRAINED_DIR.mkdir(parents=True, exist_ok=True)
EMB_DIR = PROC_DIR / "gnn_embeddings"
EMB_DIR.mkdir(parents=True, exist_ok=True)
XGB_GNN_EMBED_DIR = PROC_DIR / "baseline" / "xgb_gnn_embed"
XGB_GNN_EMBED_DIR.mkdir(parents=True, exist_ok=True)

def get_experiment_params():
    temporal_types = sorted({exp.temporal_type for exp in DEFAULT_EXPERIMENTS})
    horizons = sorted({h for exp in DEFAULT_EXPERIMENTS for h in exp.horizons})
    lag_windows = sorted({L for exp in DEFAULT_EXPERIMENTS for L in exp.lag_windows})
    return temporal_types, horizons, lag_windows

TEMPORAL_TYPES, HORIZONS, LAG_WINDOWS = get_experiment_params()
from pathlib import Path
import pandas as pd
import numpy as np
from config.config import PROC_DIR
EMB_DIR = PROC_DIR / "gnn_embeddings"
XGB_BASE_DIR = PROC_DIR / "baseline" / "xgboost"
XGB_GNN_EMBED_DIR = PROC_DIR / "baseline" / "xgb_gnn_embed"
XGB_GNN_EMBED_DIR.mkdir(parents=True, exist_ok=True)
PROJECTED_VIEWS = ["same_group", "same_subgroup", "same_plant", "same_storage"]

def build_xgb_tabular_gnnembed_projected4view(
    horizon: int,
    lag_window: int,
    temporal_type: str,
    seed: int,
    mode_name: str = "raw",
) -> None:
    base_path = (
        XGB_BASE_DIR
        / f"xgboost_tabular_h{horizon}_lag{lag_window}_{temporal_type}.parquet"
    )
    if not base_path.exists():
        print(f"[XGB-GNNEMB-PROJ] Base tabular not found: {base_path}, skip.")
        return

    df_base = pd.read_parquet(base_path)
    df_base["node_id"] = df_base["node_id"].astype(str)
    df_base["day"] = df_base["day"].astype(int)
    df_base["node_index"] = df_base["node_index"].astype(int)

    df_base = (
        df_base.sort_values(["day", "node_index", "date"])
        .drop_duplicates(subset=["day", "node_index"], keep="last")
    )
    print(
        f"[XGB-GNNEMB-PROJ] base rows={len(df_base)}, "
        f"unique(day,node_index)={df_base[['day','node_index']].drop_duplicates().shape[0]}"
    )

    emb_path = (
        EMB_DIR
        / f"gnn_projected_emb_4views_h{horizon}_lag{lag_window}_{temporal_type}_seed{seed}_{mode_name}.parquet"
    )
    if not emb_path.exists():
        print(f"[XGB-GNNEMB-PROJ] Embedding file not found: {emb_path}, skip.")
        return

    df_emb = pd.read_parquet(emb_path)
    df_emb["day"] = df_emb["day"].astype(int)
    df_emb["node_index_pos"] = df_emb["node_index_pos"].astype(int)

    node_indices = np.sort(df_base["node_index"].unique())
    pos2nodeindex = {pos: int(idx) for pos, idx in enumerate(node_indices)}

    df_emb["node_index"] = df_emb["node_index_pos"].map(pos2nodeindex)

    dfs = []
    for view in PROJECTED_VIEWS:
        d_v = df_emb[df_emb["view"] == view].copy()
        if d_v.empty:
            print(f"[XGB-GNNEMB-PROJ] Warning: no rows for view={view}")
            continue

        emb_cols = [c for c in d_v.columns if c.startswith("emb_")]
        d_v = d_v[["day", "node_index"] + emb_cols]
        d_v = d_v.rename(columns={c: f"{c}_{view}" for c in emb_cols})

        dfs.append(d_v)

    if not dfs:
        print(f"[XGB-GNNEMB-PROJ] No embeddings for any view, skip.")
        return

    from functools import reduce

    df_emb_wide = reduce(
        lambda left, right: left.merge(
            right, on=["day", "node_index"], how="inner"
        ),
        dfs,
    )
    print(
        f"[XGB-GNNEMB-PROJ] emb_wide rows={len(df_emb_wide)}, "
        f"unique(day,node_index)={df_emb_wide[['day','node_index']].drop_duplicates().shape[0]}"
    )

    df_merged = df_base.merge(
        df_emb_wide,
        on=["day", "node_index"],
        how="inner",        
        validate="1:1",
    )
    print(
        f"[XGB-GNNEMB-PROJ] merged rows={len(df_merged)}, "
        f"unique(day,node_index)={df_merged[['day','node_index']].drop_duplicates().shape[0]}"
    )

    out_path = (
        XGB_GNN_EMBED_DIR
        / f"xgboost_tabular_gnnembed_projected4view_"
          f"h{horizon}_lag{lag_window}_{temporal_type}_seed{seed}.parquet"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_merged.to_parquet(out_path, index=False)
    print(f"[XGB-GNNEMB-PROJ] Saved tabular+projected4view to {out_path}")

def build_xgb_tabular_gnnembed_homo_or_hetero(
    horizon: int,
    lag_window: int,
    temporal_type: str,
    seed: int,
    mode_name: str = "raw",
    graph_type: Literal["homo5", "hetero5"] = "homo5",
) -> None:
    base_path = (
        XGB_BASE_DIR
        / f"xgboost_tabular_h{horizon}_lag{lag_window}_{temporal_type}.parquet"
    )
    if not base_path.exists():
        print(f"[XGB-GNNEMB-{graph_type}] Base tabular not found: {base_path}, skip.")
        return

    df_base = pd.read_parquet(base_path)
    df_base["node_id"] = df_base["node_id"].astype(str)
    df_base["day"] = df_base["day"].astype(int)
    df_base["node_index"] = df_base["node_index"].astype(int)

    df_base = (
        df_base.sort_values(["day", "node_index", "date"])
        .drop_duplicates(subset=["day", "node_index"], keep="last")
    )

    if graph_type == "homo5":
        emb_fname = f"gnn_homo5_emb_h{horizon}_lag{lag_window}_{temporal_type}_seed{seed}_{mode_name}.parquet"
        out_fname = f"xgboost_tabular_gnnembed_homo5_h{horizon}_lag{lag_window}_{temporal_type}_seed{seed}.parquet"
    else:
        emb_fname = f"gnn_hetero5_emb_h{horizon}_lag{lag_window}_{temporal_type}_seed{seed}_{mode_name}.parquet"
        out_fname = f"xgboost_tabular_gnnembed_hetero5_h{horizon}_lag{lag_window}_{temporal_type}_seed{seed}.parquet"

    emb_path = EMB_DIR / emb_fname
    if not emb_path.exists():
        print(f"[XGB-GNNEMB-{graph_type}] Embedding file not found: {emb_path}, skip.")
        return

    df_emb = pd.read_parquet(emb_path)
    df_emb["day"] = df_emb["day"].astype(int)
    df_emb["node_index_pos"] = df_emb["node_index_pos"].astype(int)

    node_indices = np.sort(df_base["node_index"].unique())
    pos2nodeindex = {pos: int(idx) for pos, idx in enumerate(node_indices)}
    df_emb["node_index"] = df_emb["node_index_pos"].map(pos2nodeindex)

    emb_cols = [c for c in df_emb.columns if c.startswith("emb_")]
    df_emb_slim = df_emb[["day", "node_index"] + emb_cols]

    df_merged = df_base.merge(
        df_emb_slim,
        on=["day", "node_index"],
        how="inner",
        validate="1:1",
    )

    out_path = XGB_GNN_EMBED_DIR / out_fname
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_merged.to_parquet(out_path, index=False)
    print(f"[XGB-GNNEMB-{graph_type}] Saved tabular+{graph_type} to {out_path}")

PROJECTED_VIEWS = ["same_group", "same_subgroup", "same_plant", "same_storage"]


def export_projected_embeddings_for_config(
    temporal_type: str,
    lag_window: int,
    horizon: int,
    device: str = "cuda",
    seed: int = 0,
    mode_name: str = "raw", 
) -> None:
    pkg_path = GNN_DIR / f"gnn_projected_h{horizon}_lag{lag_window}_{temporal_type}.pt"
    if not pkg_path.exists():
        print(f"[EXPORT-PROJ] {pkg_path} not found, skip.")
        return

    print(f"[EXPORT-PROJ] Loading graph package from {pkg_path}")
    pkg = torch.load(pkg_path, map_location=device, weights_only=False)
    X_product = pkg["X_product"]        
    days = pkg["days"]                    
    split = pkg["split"]                 
    edge_index_dict = pkg["edge_index_dict"]

    T, N_prod, F_in = X_product.shape

    rows = []
    with torch.no_grad():
        for view_name in PROJECTED_VIEWS:

            enc_path = (
                GNN_TRAINED_DIR
                / f"projected_encoder_{view_name}_h{horizon}_lag{lag_window}_"
                  f"{temporal_type}_{mode_name}_seed{seed}.pt"
            )
            if not enc_path.exists():
                print(f"[EXPORT-PROJ] encoder for view={view_name} not found: {enc_path}, skip this view.")
                continue

            print(f"[EXPORT-PROJ] Loading trained encoder for view={view_name} from {enc_path}")
            enc_pkg = torch.load(enc_path, map_location=device, weights_only=False)

            encoder = ProjectedGINEncoder(
                in_channels=F_in,
                hidden_channels=96,
                num_layers=3,
            ).to(device)
            encoder.load_state_dict(enc_pkg["encoder_state_dict"])
            encoder.eval()

            if view_name not in edge_index_dict:
                print(
                    f"[EXPORT-PROJ] view {view_name} not in edge_index_dict, "
                    f"keys={list(edge_index_dict.keys())}, skip this view."
                )
                continue

            edge_index = edge_index_dict[view_name].to(device)

            for t in range(T):
                x_t = X_product[t].to(device)
                h_t = encoder(x_t, edge_index)   
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
            f"lag={lag_window}, H={horizon}"
        )
        return

    df_emb = pd.DataFrame(rows)
    out_path = (
        EMB_DIR
        / f"gnn_projected_emb_4views_h{horizon}_lag{lag_window}_{temporal_type}_seed{seed}_{mode_name}.parquet"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_emb.to_parquet(out_path, index=False)
    print(f"[EXPORT-PROJ] Saved 4-view projected embeddings to {out_path}")

def export_homo5_embeddings_for_config(
    temporal_type: str,
    lag_window: int,
    horizon: int,
    device: str = "cuda",
    seed: int = 0,
    mode_name: str = "raw",
) -> None:
    pkg_path = GNN_DIR / f"gnn_homo5_h{horizon}_lag{lag_window}_{temporal_type}.pt"
    if not pkg_path.exists():
        print(f"[EXPORT-HOMO5] {pkg_path} not found, skip.")
        return

    print(f"[EXPORT-HOMO5] Loading graph package from {pkg_path}")
    pkg = torch.load(pkg_path, map_location=device, weights_only=False)
    if "edge_index" not in pkg:
        print(
            "[EXPORT-HOMO5] pkg['edge_index'] not found. "
            "Hãy sửa build_gnn_datasets_for_config để lưu thêm edge_index flatten "
            "vào gnn_homo5_*.pt (gộp tất cả edge types), rồi chạy lại build_graphs.py."
        )
        return

    X_product = pkg["X_product"]        
    days = pkg["days"]                    
    split = pkg["split"]                  
    edge_index = pkg["edge_index"]        
    num_nodes_dict = pkg["num_nodes_dict"]
    nodes_tbl = pkg["nodes_homo_table"]  

    node_type_order = nodes_tbl["node_type"].unique().tolist()
    T, N_prod, F_in = X_product.shape

    enc_path = (
        GNN_TRAINED_DIR
        / f"homo5_encoder_h{horizon}_lag{lag_window}_{temporal_type}_{mode_name}_seed{seed}.pt"
    )
    if not enc_path.exists():
        print(f"[EXPORT-HOMO5] encoder checkpoint not found: {enc_path}, skip.")
        return

    print(f"[EXPORT-HOMO5] Loading trained encoder from {enc_path}")
    enc_pkg = torch.load(enc_path, map_location=device, weights_only=False)

    encoder = HomogeneousFiveTypeGINEncoder(
        in_channels=F_in,
        num_nodes_dict=num_nodes_dict,
        node_type_order=node_type_order,
        hidden_channels=96,
        num_layers=3,
        node_type_emb_dim=8,
    ).to(device)
    encoder.load_state_dict(enc_pkg["encoder_state_dict"])
    encoder.eval()

    rows = []
    edge_index = edge_index.to(device)
    with torch.no_grad():
        for t in range(T):
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

            h_prod = encoder(x_dict, edge_index)  
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
            f"lag={lag_window}, H={horizon}"
        )
        return

    df_emb = pd.DataFrame(rows)
    out_path = (
        EMB_DIR
        / f"gnn_homo5_emb_h{horizon}_lag{lag_window}_{temporal_type}_seed{seed}_{mode_name}.parquet"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_emb.to_parquet(out_path, index=False)
    print(f"[EXPORT-HOMO5] Saved homo5 embeddings to {out_path}")


def export_hetero5_embeddings_for_config(
    temporal_type: str,
    lag_window: int,
    horizon: int,
    device: str = "cuda",
    seed: int = 0,
    mode_name: str = "raw",
) -> None:
    pkg_path = GNN_DIR / f"gnn_hetero5_h{horizon}_lag{lag_window}_{temporal_type}.pt"
    if not pkg_path.exists():
        print(f"[EXPORT-HET5] {pkg_path} not found, skip.")
        return

    print(f"[EXPORT-HET5] Loading graph package from {pkg_path}")
    pkg = torch.load(pkg_path, map_location=device, weights_only=False)

    X_product = pkg["X_product"]         
    days = pkg["days"]                    
    split = pkg["split"]                  
    edge_index_dict = pkg["edge_index_dict"]
    num_nodes_dict = pkg["num_nodes_dict"]
    nodes_tbl = pkg["nodes_hetero_table"]

    node_types = nodes_tbl["node_type"].unique().tolist()
    edge_types = list(edge_index_dict.keys())
    F_in = X_product.shape[2]

    in_channels_dict = {"edge_types": edge_types}
    for nt in node_types:
        in_channels_dict[nt] = F_in

    enc_path = (
        GNN_TRAINED_DIR
        / f"hetero5_encoder_h{horizon}_lag{lag_window}_{temporal_type}_{mode_name}_seed{seed}.pt"
    )
    if not enc_path.exists():
        print(f"[EXPORT-HET5] encoder checkpoint not found: {enc_path}, skip.")
        return

    print(f"[EXPORT-HET5] Loading trained encoder from {enc_path}")
    enc_pkg = torch.load(enc_path, map_location=device, weights_only=False)

    encoder = HeterogeneousGINEncoder(
        in_channels_dict=in_channels_dict,
        hidden_channels=96,
        num_layers=2,
    ).to(device)
    encoder.load_state_dict(enc_pkg["encoder_state_dict"])
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

            edge_index_dict_device = {
                k: v.to(device) for k, v in edge_index_dict.items()
            }

            h_prod = encoder(x_dict, edge_index_dict_device)   
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
    out_path = (
        EMB_DIR
        / f"gnn_hetero5_emb_h{horizon}_lag{lag_window}_{temporal_type}_seed{seed}_{mode_name}.parquet"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_emb.to_parquet(out_path, index=False)
    print(f"[EXPORT-HET5] Saved hetero5 embeddings to {out_path}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = [0, 1, 2]
    mode_name = "raw"

    for exp in DEFAULT_EXPERIMENTS:
        t_type = exp.temporal_type
        for H in exp.horizons:
            for L in exp.lag_windows:
                for seed in seeds:
                    print(
                        f"\n=== EXPORT EMBEDDINGS + BUILD XGB+GNNEMB "
                        f"H{H}, lag={L}, temporal={t_type}, seed={seed} ==="
                    )

                    export_projected_embeddings_for_config(
                        temporal_type=t_type,
                        lag_window=L,
                        horizon=H,
                        device=device,
                        seed=seed,
                        mode_name=mode_name,
                    )
                    export_homo5_embeddings_for_config(
                        temporal_type=t_type,
                        lag_window=L,
                        horizon=H,
                        device=device,
                        seed=seed,
                        mode_name=mode_name,
                    )
                    export_hetero5_embeddings_for_config(
                        temporal_type=t_type,
                        lag_window=L,
                        horizon=H,
                        device=device,
                        seed=seed,
                        mode_name=mode_name,
                    )

                    build_xgb_tabular_gnnembed_projected4view(
                        horizon=H,
                        lag_window=L,
                        temporal_type=t_type,
                        seed=seed,
                        mode_name=mode_name,
                    )
                    build_xgb_tabular_gnnembed_homo_or_hetero(
                        horizon=H,
                        lag_window=L,
                        temporal_type=t_type,
                        seed=seed,
                        mode_name=mode_name,
                        graph_type="homo5",
                    )
                    build_xgb_tabular_gnnembed_homo_or_hetero(
                        horizon=H,
                        lag_window=L,
                        temporal_type=t_type,
                        seed=seed,
                        mode_name=mode_name,
                        graph_type="hetero5",
                    )

    print("\n[export_and_build_xgb_gnnembed] Done.")


if __name__ == "__main__":
    main()
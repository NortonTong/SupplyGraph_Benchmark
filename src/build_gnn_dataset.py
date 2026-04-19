import numpy as np
import pandas as pd
import torch
from pathlib import Path

from config.config import PROC_DIR, NODE_DIR

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 0)

HORIZON = 7
LAG_WINDOWS = [7, 14]
TEMPORAL_TYPES = ["unit", "weight"]

GRAPH_DIR = PROC_DIR / "graphs"
HOMO_DIR  = GRAPH_DIR / "homogeneous_graphs"
HETERO_DIR = GRAPH_DIR / "heterogeneous_graphs"
PROJ_DIR  = GRAPH_DIR / "projected_product_graphs"

GNN_DIR = PROC_DIR / "gnn"
GNN_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 0. Node metadata (product node_id <-> node_index)
# ------------------------------------------------------------

def load_node_metadata() -> pd.DataFrame:
    node_index_path = NODE_DIR / "NodesIndex.csv"
    node_group_path = NODE_DIR / "Node Types (Product Group and Subgroup).csv"
    node_plant_storage_path = NODE_DIR / "Nodes Type (Plant & Storage).csv"

    df_index = pd.read_csv(node_index_path)
    df_index = df_index.rename(columns={"Node": "node_id", "NodeIndex": "node_index"})
    df_index = df_index.drop_duplicates(keep="first")

    df_group = pd.read_csv(node_group_path)
    df_group = df_group.rename(
        columns={"Node": "node_id", "Group": "group", "Sub-Group": "sub_group"}
    )
    df_group = df_group.drop_duplicates(keep="first")

    df_plant = pd.read_csv(node_plant_storage_path)
    df_plant = df_plant.rename(
        columns={
            "Node": "node_id",
            "Plant": "plant",
            "Storage Location": "storage_location",
        }
    )
    df_plant = df_plant.drop_duplicates(keep="first")

    df_meta = df_index.merge(df_group, on="node_id", how="left")
    df_meta = df_meta.merge(df_plant, on="node_id", how="left")
    df_meta = df_meta.sort_values("node_index").reset_index(drop=True)
    return df_meta


# ------------------------------------------------------------
# 1. Load XGBoost OHE => tensors X_prod[T,N,F], Y_prod[T,N]
# ------------------------------------------------------------

def load_xgb_ohe(
    temporal_type: str,
    lag_window: int,
    horizon: int = HORIZON,
) -> pd.DataFrame:
    """
    Đọc baseline/xgboost_tabular_h{H}_lag{lag_window}_{temporal_type}.parquet đã OHE sẵn.
    """
    base_dir = PROC_DIR / "baseline"
    path = base_dir / f"xgboost_tabular_h{horizon}_lag{lag_window}_{temporal_type}.parquet"
    print(f"[GNN-DATA] Loading XGBoost OHE from {path}")
    df = pd.read_parquet(path)

    required_cols = ["node_id", "node_index", "date", "day", "split", "target"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"[GNN-DATA] {path} missing columns: {missing}")

    df["date"] = pd.to_datetime(df["date"])
    df["day"] = df["day"].astype(int)
    df["node_index"] = df["node_index"].astype(int)

    # unique per (day, node_id)
    df = (
        df.sort_values(["date", "node_id"])
          .drop_duplicates(subset=["date", "node_id"], keep="last")
    )
    return df


def build_time_tensors_from_xgb(df: pd.DataFrame):
    """
    df: baseline XGB OHE (1 temporal_type, 1 lag_window)
    -> pkg_common (X_prod, Y_prod, days, split, node_ids_prod, node_index_prod)
       nodeindex2pos_prod
    """
    df = df.copy()

    # product node_index
    node_indices = np.sort(df["node_index"].dropna().unique())
    nodeindex2pos = {int(idx): i for i, idx in enumerate(node_indices)}
    N = len(node_indices)

    # days
    days = np.sort(df["day"].dropna().unique())
    day2idx = {int(d): i for i, d in enumerate(days)}
    T = len(days)

    # feature columns = tất cả trừ base + target
    drop_cols = ["node_id", "node_index", "date", "day", "split", "target"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    Fdim = len(feature_cols)

    print(f"[GNN-DATA] #days={T}, #prod_nodes={N}, #features={Fdim}")

    X = np.zeros((T, N, Fdim), dtype=np.float32)
    Y = np.full((T, N), np.nan, dtype=np.float32)

    df_sorted = df.sort_values(["day", "node_index"])

    for _, row in df_sorted.iterrows():
        t = day2idx[int(row["day"])]
        n = nodeindex2pos[int(row["node_index"])]
        X[t, n, :] = row[feature_cols].values.astype(np.float32)
        Y[t, n] = row["target"]

    # split theo day
    day_split = (
        df.groupby("day")["split"]
        .first()
        .sort_index()
        .values
    )

    df_nodes = (
        df[["node_id", "node_index"]]
        .drop_duplicates(subset=["node_index"])
        .sort_values("node_index")
    )
    node_ids_sorted = df_nodes["node_id"].values

    pkg_common = {
        "X_product": torch.from_numpy(X),                          # [T, N, F]
        "Y_product": torch.from_numpy(Y),                          # [T, N]
        "days": torch.tensor(days, dtype=torch.long),              # [T]
        "split": day_split,                                        # [T]
        "node_ids_product": node_ids_sorted,                       # [N]
        "node_index_product": torch.tensor(node_indices, dtype=torch.long),
        "feature_cols": feature_cols,
    }
    return pkg_common, nodeindex2pos


# ------------------------------------------------------------
# 2. Projected product graphs (4 view)
# ------------------------------------------------------------

def load_projected_edge_parquet(out_name: str) -> pd.DataFrame:
    """
    Đọc {out_name}_edges.parquet (src,dst node_id sản phẩm)
    """
    path = PROJ_DIR / f"{out_name}_edges.parquet"
    df_e = pd.read_parquet(path)
    return df_e


def build_projected_edge_indices(nodeindex2pos_prod: dict, df_meta: pd.DataFrame):
    """
    Tạo 4 edge_index product–product cho 4 projected graphs
      - same_group
      - same_subgroup
      - same_plant
      - same_storage
    dùng các file *_edges.parquet đã build.
    """
    # map node_id -> node_index (product)
    nodeid2index = dict(
        zip(df_meta["node_id"].astype(str), df_meta["node_index"].astype(int))
    )

    def convert_edges(df_edges: pd.DataFrame) -> torch.Tensor:
        src_ids = df_edges["src"].astype(str).values
        dst_ids = df_edges["dst"].astype(str).values
        src_pos = []
        dst_pos = []
        for s, d in zip(src_ids, dst_ids):
            if s not in nodeid2index or d not in nodeid2index:
                continue
            idx_s = nodeid2index[s]
            idx_d = nodeid2index[d]
            if idx_s not in nodeindex2pos_prod or idx_d not in nodeindex2pos_prod:
                continue
            src_pos.append(nodeindex2pos_prod[idx_s])
            dst_pos.append(nodeindex2pos_prod[idx_d])

        if len(src_pos) == 0:
            return torch.empty((2, 0), dtype=torch.long)
        arr = np.vstack([src_pos, dst_pos])
        return torch.tensor(arr, dtype=torch.long)

    edge_index_dict = {}

    # same_group
    df_e = load_projected_edge_parquet("product_graph_same_group")
    edge_index_dict["same_group"] = convert_edges(df_e)
    print(f"[GNN-PROJ] same_group edges={edge_index_dict['same_group'].size(1)}")

    # same_subgroup
    df_e = load_projected_edge_parquet("product_graph_same_subgroup")
    edge_index_dict["same_subgroup"] = convert_edges(df_e)
    print(f"[GNN-PROJ] same_subgroup edges={edge_index_dict['same_subgroup'].size(1)}")

    # same_plant
    df_e = load_projected_edge_parquet("product_graph_same_plant")
    edge_index_dict["same_plant"] = convert_edges(df_e)
    print(f"[GNN-PROJ] same_plant edges={edge_index_dict['same_plant'].size(1)}")

    # same_storage
    df_e = load_projected_edge_parquet("product_graph_same_storage")
    edge_index_dict["same_storage"] = convert_edges(df_e)
    print(f"[GNN-PROJ] same_storage edges={edge_index_dict['same_storage'].size(1)}")

    return edge_index_dict


# ------------------------------------------------------------
# 3. Homogeneous 5-node-type graph (product–type, untyped)
# ------------------------------------------------------------

def build_homo5type_from_parquet():
    """
    Đọc nodes_homogeneous_5type.parquet + edges_homogeneous_5type.parquet
    -> mapping per node_type và edge_index_dict (product–type, untyped).
    """
    nodes_tbl = pd.read_parquet(HOMO_DIR / "nodes_homogeneous_5type.parquet")
    edges_tbl = pd.read_parquet(HOMO_DIR / "edges_homogeneous_5type.parquet")

    nodes_tbl["node_id"] = nodes_tbl["node_id"].astype(str)

    # tách node theo node_type
    type_groups = {}
    for nt in nodes_tbl["node_type"].unique():
        df_nt = nodes_tbl[nodes_tbl["node_type"] == nt].copy()
        df_nt = df_nt.reset_index(drop=True)
        type_groups[nt] = df_nt

    # mapping node_id -> type, local index
    nodeid2type = {}
    nodeid2local = {}
    num_nodes_dict = {}
    for nt, df_nt in type_groups.items():
        num_nodes_dict[nt] = len(df_nt)
        for i, row in df_nt.reset_index().iterrows():
            nid = str(row["node_id"])
            nodeid2type[nid] = nt
            nodeid2local[nid] = int(i)

    edge_index_dict = {}

    def add_edge_type(src_type, rel_name, dst_type, df_e: pd.DataFrame):
        src_local = []
        dst_local = []
        for _, row in df_e.iterrows():
            s = str(row["src"])
            d = str(row["dst"])
            if s not in nodeid2type or d not in nodeid2type:
                continue
            if nodeid2type[s] != src_type or nodeid2type[d] != dst_type:
                continue
            src_local.append(nodeid2local[s])
            dst_local.append(nodeid2local[d])
        if len(src_local) == 0:
            ei = torch.empty((2, 0), dtype=torch.long)
        else:
            arr = np.vstack([src_local, dst_local])
            ei = torch.tensor(arr, dtype=torch.long)
        edge_index_dict[(src_type, rel_name, dst_type)] = ei
        print(
            f"[GNN-HOMO5] edge ({src_type}, {rel_name}, {dst_type}) "
            f"edges={ei.size(1) if ei.numel() > 0 else 0}"
        )

    # join edges với node_type của dst để biết dst thuộc loại nào
    df_g = edges_tbl.merge(
        nodes_tbl[["node_id", "node_type"]],
        left_on="dst",
        right_on="node_id",
        how="left",
    )
    # df_g: src, dst, node_id, node_type

    # product_group
    df_pg = df_g[df_g["node_type"] == "product_group"][["src", "dst"]]
    add_edge_type("product", "product_group_edge", "product_group", df_pg)

    # product_sub_group
    df_psg = df_g[df_g["node_type"] == "product_sub_group"][["src", "dst"]]
    add_edge_type("product", "product_sub_group_edge", "product_sub_group", df_psg)

    # plant
    df_pl = df_g[df_g["node_type"] == "plant"][["src", "dst"]]
    add_edge_type("product", "product_plant_edge", "plant", df_pl)

    # storage_location
    df_st = df_g[df_g["node_type"] == "storage_location"][["src", "dst"]]
    add_edge_type("product", "product_storage_edge", "storage_location", df_st)

    return edge_index_dict, num_nodes_dict, nodes_tbl


# ------------------------------------------------------------
# 4. Heterogeneous 5-node-type graph (product–type typed)
# ------------------------------------------------------------

def build_hetero5type_from_parquet():
    """
    Đọc nodes_heterogeneous_5type.parquet + edges_heterogeneous_5type.parquet
    -> edge_index_dict hetero cho 4 loại edge_type, kèm chiều ngược vào product.
    """
    nodes_tbl = pd.read_parquet(HETERO_DIR / "nodes_heterogeneous_5type.parquet")
    edges_tbl = pd.read_parquet(HETERO_DIR / "edges_heterogeneous_5type.parquet")

    nodes_tbl["node_id"] = nodes_tbl["node_id"].astype(str)
    type_groups = {}
    for nt in nodes_tbl["node_type"].unique():
        df_nt = nodes_tbl[nodes_tbl["node_type"] == nt].copy()
        df_nt = df_nt.reset_index(drop=True)
        type_groups[nt] = df_nt

    nodeid2type = {}
    nodeid2local = {}
    num_nodes_dict = {}
    for nt, df_nt in type_groups.items():
        num_nodes_dict[nt] = len(df_nt)
        for i, row in df_nt.reset_index().iterrows():
            nid = str(row["node_id"])
            nodeid2type[nid] = nt
            nodeid2local[nid] = int(i)

    edge_index_dict = {}

    def add_edge_pair(edge_type_name: str, rel_fwd: str, dst_type: str, rel_rev: str):
        # edge_type_name trong edges_tbl: product_group / product_subgroup / product_plant / product_storage
        df_et = edges_tbl[edges_tbl["edge_type"] == edge_type_name]
        src_local_fwd = []
        dst_local_fwd = []
        for _, row in df_et.iterrows():
            s = str(row["src"])
            d = str(row["dst"])
            if s not in nodeid2type or d not in nodeid2type:
                continue
            # tất cả các edge đều từ product sang type
            if nodeid2type[s] != "product":
                continue
            if nodeid2type[d] != dst_type:
                continue
            src_local_fwd.append(nodeid2local[s])
            dst_local_fwd.append(nodeid2local[d])

        if len(src_local_fwd) == 0:
            ei_fwd = torch.empty((2, 0), dtype=torch.long)
            ei_rev = torch.empty((2, 0), dtype=torch.long)
        else:
            arr_fwd = np.vstack([src_local_fwd, dst_local_fwd])
            ei_fwd = torch.tensor(arr_fwd, dtype=torch.long)
            # reverse: dst->src
            arr_rev = np.vstack([dst_local_fwd, src_local_fwd])
            ei_rev = torch.tensor(arr_rev, dtype=torch.long)

        edge_index_dict[("product", rel_fwd, dst_type)] = ei_fwd
        edge_index_dict[(dst_type, rel_rev, "product")] = ei_rev

        print(
            f"[GNN-HET5] edge_type={edge_type_name}, "
            f"('product', {rel_fwd}, {dst_type}) edges={ei_fwd.size(1)}, "
            f"({dst_type}, {rel_rev}, 'product') edges={ei_rev.size(1)}"
        )

    add_edge_pair("product_group",    "product_group",    "product_group",    "rev_product_group")
    add_edge_pair("product_subgroup", "product_subgroup", "product_sub_group","rev_product_subgroup")
    add_edge_pair("product_plant",    "product_plant",    "plant",            "rev_product_plant")
    add_edge_pair("product_storage",  "product_storage",  "storage_location", "rev_product_storage")

    return edge_index_dict, num_nodes_dict, nodes_tbl


# ------------------------------------------------------------
# 5. Build all 3 GNN datasets for each config
# ------------------------------------------------------------

def build_gnn_datasets_for_config(temporal_type: str, lag_window: int):
    print(
        f"\n=== Build GNN datasets (projected/homo5/hetero5) "
        f"from XGB OHE: temporal_type={temporal_type}, lag_window={lag_window} ==="
    )

    df_meta = load_node_metadata()

    # 1) Product features from XGB OHE
    df_xgb = load_xgb_ohe(temporal_type, lag_window, HORIZON)
    pkg_common, nodeindex2pos_prod = build_time_tensors_from_xgb(df_xgb)

    # 2) Projected product graphs
    edge_index_proj = build_projected_edge_indices(nodeindex2pos_prod, df_meta)
    pkg_proj = {
        **pkg_common,
        "edge_index_dict": edge_index_proj,
        "graph_def": "projected_product_4view",
    }
    out_proj = (
        GNN_DIR
        / f"gnn_projected_h{HORIZON}_lag{lag_window}_{temporal_type}.pt"
    )
    torch.save(pkg_proj, out_proj)
    print(f"[GNN-SAVE] Saved projected dataset to {out_proj}")

    # 3) Homogeneous 5-node-type (product–type untyped)
    edge_index_homo5, num_nodes_homo5, nodes_homo_tbl = build_homo5type_from_parquet()
    pkg_homo5 = {
        **pkg_common,
        "edge_index_dict": edge_index_homo5,
        "num_nodes_dict": num_nodes_homo5,
        "nodes_homo_table": nodes_homo_tbl,
        "graph_def": "homogeneous_5node_types",
    }
    out_homo5 = (
        GNN_DIR
        / f"gnn_homo5_h{HORIZON}_lag{lag_window}_{temporal_type}.pt"
    )
    torch.save(pkg_homo5, out_homo5)
    print(f"[GNN-SAVE] Saved homogeneous-5type dataset to {out_homo5}")

    # 4) Heterogeneous 5-node-type (product–type typed, có chiều ngược)
    edge_index_het5, num_nodes_het5, nodes_het_tbl = build_hetero5type_from_parquet()
    pkg_het5 = {
        **pkg_common,
        "edge_index_dict": edge_index_het5,
        "num_nodes_dict": num_nodes_het5,
        "nodes_hetero_table": nodes_het_tbl,
        "graph_def": "heterogeneous_5node_types",
    }
    out_het5 = (
        GNN_DIR
        / f"gnn_hetero5_h{HORIZON}_lag{lag_window}_{temporal_type}.pt"
    )
    torch.save(pkg_het5, out_het5)
    print(f"[GNN-SAVE] Saved heterogeneous-5type dataset to {out_het5}")


def main():
    for temporal_type in TEMPORAL_TYPES:
        for lag_window in LAG_WINDOWS:
            build_gnn_datasets_for_config(temporal_type, lag_window)


if __name__ == "__main__":
    main()
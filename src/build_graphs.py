import pickle
from typing import Dict, Tuple, List

import networkx as nx
import numpy as np
import pandas as pd
import torch

from config.config import (
    NODE_DIR,
    PROC_DIR,
    DEFAULT_EXPERIMENTS,
)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 0)

GRAPH_DIR = PROC_DIR / "graphs"
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

HOMO_DIR = GRAPH_DIR / "homogeneous_graphs"
HETERO_DIR = GRAPH_DIR / "heterogeneous_graphs"
PROJ_DIR = GRAPH_DIR / "projected_product_graphs"

HOMO_DIR.mkdir(parents=True, exist_ok=True)
HETERO_DIR.mkdir(parents=True, exist_ok=True)
PROJ_DIR.mkdir(parents=True, exist_ok=True)

GNN_DIR = PROC_DIR / "gnn"
GNN_DIR.mkdir(parents=True, exist_ok=True)

XGB_BASE_DIR = PROC_DIR / "baseline" / "xgboost"
XGB_GRAPH_DIR = PROC_DIR / "baseline" / "xgb_graph"
XGB_GRAPH_DIR.mkdir(parents=True, exist_ok=True)

NODE_TYPE_PRODUCT = "product"
NODE_TYPE_PRODUCT_GROUP = "product_group"
NODE_TYPE_PRODUCT_SUBGR = "product_sub_group"
NODE_TYPE_PLANT = "plant"
NODE_TYPE_STORAGE = "storage_location"

EDGE_TYPE_PRODUCT_GROUP = "product_group"
EDGE_TYPE_PRODUCT_SUBGROUP = "product_subgroup"
EDGE_TYPE_PRODUCT_PLANT = "product_plant"
EDGE_TYPE_PRODUCT_STORAGE = "product_storage"

HORIZON = 7

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
        columns={ "Node": "node_id", "Plant": "plant", "Storage Location": "storage_location",}
    )
    df_plant = df_plant.drop_duplicates(keep="first")

    df_meta = df_index.merge(df_group, on="node_id", how="left")
    df_meta = df_meta.merge(df_plant, on="node_id", how="left")
    return df_meta

def sanity_check_alignment(df_meta: pd.DataFrame):
    temporal_types = sorted({exp.temporal_type for exp in DEFAULT_EXPERIMENTS})

    for temporal_type in temporal_types:
        raw_path = PROC_DIR / "base" / f"base_raw_h{HORIZON}_{temporal_type}.parquet"
        if not raw_path.exists():
            print(f"[SanityCheck] {raw_path} not found, skip.")
            continue

        df_ts = pd.read_parquet(raw_path)
        df_ts_idx = (
            df_ts[["node_id", "node_index"]]
            .drop_duplicates()
            .sort_values("node_index")
            .reset_index(drop=True)
        )
        df_meta_idx = (
            df_meta[["node_id", "node_index"]]
            .drop_duplicates()
            .sort_values("node_index")
            .reset_index(drop=True)
        )

        if len(df_ts_idx) != len(df_meta_idx):
            print(
                f"[SanityCheck][{temporal_type}] WARNING: "
                f"#nodes in timeseries ({len(df_ts_idx)}) != #nodes in meta ({len(df_meta_idx)})"
            )

        merged = df_ts_idx.merge(df_meta_idx, on=["node_id", "node_index"], how="inner")
        if len(merged) != len(df_ts_idx):
            print(
                f"[SanityCheck][{temporal_type}] WARNING: "
                f"{len(df_ts_idx) - len(merged)} nodes mismatch between timeseries and meta."
            )
        else:
            print(
                f"[SanityCheck][{temporal_type}] OK: "
                f"product node_id/node_index aligned ({len(merged)} nodes)."
            )

def build_homogeneous_5type_graph(df_meta: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()

    for _, row in df_meta.iterrows():
        prod_id = row["node_id"]
        G.add_node(
            prod_id,
            node_type=NODE_TYPE_PRODUCT,
            node_index=int(row["node_index"]),
            group=row.get("group"),
            sub_group=row.get("sub_group"),
            plant=row.get("plant"),
            storage_location=row.get("storage_location"),
        )

    unique_groups = df_meta["group"].dropna().astype(str).unique().tolist()
    for g in unique_groups:
        nid = f"GROUP::{g}"
        G.add_node(nid, node_type=NODE_TYPE_PRODUCT_GROUP, raw_value=g)

    unique_subgroups = df_meta["sub_group"].dropna().astype(str).unique().tolist()
    for sg in unique_subgroups:
        nid = f"SUBGROUP::{sg}"
        G.add_node(nid, node_type=NODE_TYPE_PRODUCT_SUBGR, raw_value=sg)

    unique_plants = df_meta["plant"].dropna().astype(str).unique().tolist()
    for p in unique_plants:
        nid = f"PLANT::{p}"
        G.add_node(nid, node_type=NODE_TYPE_PLANT, raw_value=p)

    unique_storages = df_meta["storage_location"].dropna().astype(str).unique().tolist()
    for s in unique_storages:
        nid = f"STORAGE::{s}"
        G.add_node(nid, node_type=NODE_TYPE_STORAGE, raw_value=s)

    df_g = df_meta.dropna(subset=["group"])
    for _, row in df_g.iterrows():
        prod_id = row["node_id"]
        g_val = str(row["group"])
        g_node = f"GROUP::{g_val}"
        if G.has_node(g_node):
            G.add_edge(prod_id, g_node)

    df_sg = df_meta.dropna(subset=["sub_group"])
    for _, row in df_sg.iterrows():
        prod_id = row["node_id"]
        sg_val = str(row["sub_group"])
        sg_node = f"SUBGROUP::{sg_val}"
        if G.has_node(sg_node):
            G.add_edge(prod_id, sg_node)

    df_p = df_meta.dropna(subset=["plant"])
    for _, row in df_p.iterrows():
        prod_id = row["node_id"]
        p_val = str(row["plant"])
        p_node = f"PLANT::{p_val}"
        if G.has_node(p_node):
            G.add_edge(prod_id, p_node)

    df_st = df_meta.dropna(subset=["storage_location"])
    for _, row in df_st.iterrows():
        prod_id = row["node_id"]
        st_val = str(row["storage_location"])
        st_node = f"STORAGE::{st_val}"
        if G.has_node(st_node):
            G.add_edge(prod_id, st_node)

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    print(f"[Homogeneous-5type] |V|={n_nodes}, |E|={n_edges}")

    out_gpickle = HOMO_DIR / "homogeneous_5node_types.gpickle"
    with open(out_gpickle, "wb") as f:
        pickle.dump(G, f)
    print(f"[Homogeneous-5type] Saved Graph (pickle) to {out_gpickle}")

    node_records = []
    for nid, data in G.nodes(data=True):
        node_records.append(
            {
                "node_id": nid,
                "node_type": data.get("node_type", "unknown"),
                "node_index": data.get("node_index", -1),
                "raw_value": data.get("raw_value", None),
                "group": data.get("group", None),
                "sub_group": data.get("sub_group", None),
                "plant": data.get("plant", None),
                "storage_location": data.get("storage_location", None),
            }
        )
    df_nodes = pd.DataFrame(node_records)
    df_nodes.to_parquet(HOMO_DIR / "nodes_homogeneous_5type.parquet", index=False)
    print("[Homogeneous-5type] Saved node table")

    edge_records = [{"src": u, "dst": v} for u, v in G.edges()]
    df_edges = pd.DataFrame(edge_records)
    df_edges.to_parquet(HOMO_DIR / "edges_homogeneous_5type.parquet", index=False)
    print("[Homogeneous-5type] Saved edge list")

    return G

def build_heterogeneous_5type_graph(df_meta: pd.DataFrame) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()

    for _, row in df_meta.iterrows():
        prod_id = row["node_id"]
        G.add_node(
            prod_id,
            node_type=NODE_TYPE_PRODUCT,
            node_index=int(row["node_index"]),
            group=row.get("group"),
            sub_group=row.get("sub_group"),
            plant=row.get("plant"),
            storage_location=row.get("storage_location"),
        )

    for g in df_meta["group"].dropna().astype(str).unique().tolist():
        nid = f"GROUP::{g}"
        G.add_node(nid, node_type=NODE_TYPE_PRODUCT_GROUP, raw_value=g)

    for sg in df_meta["sub_group"].dropna().astype(str).unique().tolist():
        nid = f"SUBGROUP::{sg}"
        G.add_node(nid, node_type=NODE_TYPE_PRODUCT_SUBGR, raw_value=sg)

    for p in df_meta["plant"].dropna().astype(str).unique().tolist():
        nid = f"PLANT::{p}"
        G.add_node(nid, node_type=NODE_TYPE_PLANT, raw_value=p)

    for s in df_meta["storage_location"].dropna().astype(str).unique().tolist():
        nid = f"STORAGE::{s}"
        G.add_node(nid, node_type=NODE_TYPE_STORAGE, raw_value=s)

    df_g = df_meta.dropna(subset=["group"])[["node_id", "group"]].drop_duplicates()
    for _, row in df_g.iterrows():
        prod_id = row["node_id"]
        g_val = str(row["group"])
        g_node = f"GROUP::{g_val}"
        if G.has_node(g_node):
            G.add_edge(prod_id, g_node, edge_type=EDGE_TYPE_PRODUCT_GROUP)

    df_sg = df_meta.dropna(subset=["sub_group"])[["node_id", "sub_group"]].drop_duplicates()
    for _, row in df_sg.iterrows():
        prod_id = row["node_id"]
        sg_val = str(row["sub_group"])
        sg_node = f"SUBGROUP::{sg_val}"
        if G.has_node(sg_node):
            G.add_edge(prod_id, sg_node, edge_type=EDGE_TYPE_PRODUCT_SUBGROUP)

    df_p = df_meta.dropna(subset=["plant"])[["node_id", "plant"]].drop_duplicates()
    for _, row in df_p.iterrows():
        prod_id = row["node_id"]
        p_val = str(row["plant"])
        p_node = f"PLANT::{p_val}"
        if G.has_node(p_node):
            G.add_edge(prod_id, p_node, edge_type=EDGE_TYPE_PRODUCT_PLANT)

    df_st = (
        df_meta.dropna(subset=["storage_location"])[["node_id", "storage_location"]]
        .drop_duplicates()
    )
    for _, row in df_st.iterrows():
        prod_id = row["node_id"]
        st_val = str(row["storage_location"])
        st_node = f"STORAGE::{st_val}"
        if G.has_node(st_node):
            G.add_edge(prod_id, st_node, edge_type=EDGE_TYPE_PRODUCT_STORAGE)

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    print(f"[Heterogeneous-5type] |V|={n_nodes}, |E|={n_edges}")

    out_gpickle = HETERO_DIR / "heterogeneous_5node_types.gpickle"
    with open(out_gpickle, "wb") as f:
        pickle.dump(G, f)
    print("[Heterogeneous-5type] Saved MultiDiGraph pickle")

    node_records = []
    for nid, data in G.nodes(data=True):
        node_records.append(
            {
                "node_id": nid,
                "node_type": data.get("node_type", "unknown"),
                "node_index": data.get("node_index", -1),
                "raw_value": data.get("raw_value", None),
                "group": data.get("group", None),
                "sub_group": data.get("sub_group", None),
                "plant": data.get("plant", None),
                "storage_location": data.get("storage_location", None),
            }
        )
    df_nodes = pd.DataFrame(node_records)
    df_nodes.to_parquet(HETERO_DIR / "nodes_heterogeneous_5type.parquet", index=False)
    print("[Heterogeneous-5type] Saved node table")

    edge_records = []
    for u, v, data in G.edges(data=True):
        edge_records.append(
            {"src": u, "dst": v, "edge_type": data.get("edge_type", "unknown")}
        )
    df_edges = pd.DataFrame(edge_records)
    df_edges.to_parquet(HETERO_DIR / "edges_heterogeneous_5type.parquet", index=False)
    print("[Heterogeneous-5type] Saved edge list")

    return G

def build_projected_graph(df_meta: pd.DataFrame, by_col: str, out_name: str) -> nx.Graph:
    G = nx.Graph()

    for _, row in df_meta.iterrows():
        G.add_node(
            row["node_id"],
            node_index=int(row["node_index"]),
            group=row.get("group"),
            sub_group=row.get("sub_group"),
            plant=row.get("plant"),
            storage_location=row.get("storage_location"),
        )

    df_valid = df_meta.dropna(subset=[by_col])
    for _, grp in df_valid.groupby(by_col):
        nodes = grp["node_id"].tolist()
        n = len(nodes)
        if n <= 1:
            continue
        for i in range(n):
            u = nodes[i]
            for j in range(i + 1, n):
                v = nodes[j]
                if not G.has_edge(u, v):
                    G.add_edge(u, v)

    print(
        f"[Projected] {out_name} via {by_col}: |V|={G.number_of_nodes()}, |E|={G.number_of_edges()}"
    )

    gpath = PROJ_DIR / f"{out_name}.gpickle"
    with open(gpath, "wb") as f:
        pickle.dump(G, f)
    print(f"[Projected] Saved graph pickle to {gpath}")

    edges = [{"src": u, "dst": v} for u, v in G.edges()]
    pd.DataFrame(edges).to_parquet(PROJ_DIR / f"{out_name}_edges.parquet", index=False)

    nodes = []
    for nid, data in G.nodes(data=True):
        nodes.append(
            {
                "node_id": nid,
                "node_index": data.get("node_index", -1),
                "group": data.get("group"),
                "sub_group": data.get("sub_group"),
                "plant": data.get("plant"),
                "storage_location": data.get("storage_location"),
            }
        )
    pd.DataFrame(nodes).to_parquet(PROJ_DIR / f"{out_name}_nodes.parquet", index=False)

    return G


def build_all_projected_graphs(df_meta: pd.DataFrame):
    build_projected_graph(df_meta, "group", "product_graph_same_group")
    build_projected_graph(df_meta, "sub_group", "product_graph_same_subgroup")
    build_projected_graph(df_meta, "plant", "product_graph_same_plant")
    build_projected_graph(df_meta, "storage_location", "product_graph_same_storage")
    print(f"[Projected] Built all 4 projected product graphs in {PROJ_DIR}")

def is_graph_side_ohe(col: str) -> bool:
    col_lower = col.lower()
    if col_lower.startswith("group_"):
        return True
    if col_lower.startswith("sub_group_"):
        return True
    if col_lower.startswith("plant_"):
        return True
    if col_lower.startswith("storage_") or col_lower.startswith("storage_location_"):
        return True
    return False


def load_xgb_tabular_for_gnn(
    temporal_type: str,
    lag_window: int,
    horizon: int = HORIZON,
) -> pd.DataFrame:
    path = XGB_BASE_DIR / f"xgboost_tabular_h{horizon}_lag{lag_window}_{temporal_type}.parquet"
    print(f"[GNN-DATA] Loading XGBoost tabular from {path}")
    df = pd.read_parquet(path)

    required_cols = ["node_id", "node_index", "date", "day", "split", "target"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"[GNN-DATA] {path} missing columns: {missing}")

    df["date"] = pd.to_datetime(df["date"])
    df["day"] = df["day"].astype(int)
    df["node_index"] = df["node_index"].astype(int)

    df = (
        df.sort_values(["date", "node_id"])
        .drop_duplicates(subset=["date", "node_id"], keep="last")
    )
    return df


def build_time_tensors_from_xgb_for_gnn(df: pd.DataFrame):
    df = df.copy()

    node_indices = np.sort(df["node_index"].dropna().unique())
    nodeindex2pos = {int(idx): i for i, idx in enumerate(node_indices)}
    N = len(node_indices)

    days = np.sort(df["day"].dropna().unique())
    day2idx = {int(d): i for i, d in enumerate(days)}
    T = len(days)

    drop_base = ["node_id", "node_index", "date", "day", "split", "target"]
    feature_cols = []
    for c in df.columns:
        if c in drop_base:
            continue
        if is_graph_side_ohe(c):
            continue
        feature_cols.append(c)
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
        "X_product": torch.from_numpy(X),
        "Y_product": torch.from_numpy(Y),
        "days": torch.tensor(days, dtype=torch.long),
        "split": day_split,
        "node_ids_product": node_ids_sorted,
        "node_index_product": torch.tensor(node_indices, dtype=torch.long),
        "feature_cols": feature_cols,
    }
    return pkg_common, nodeindex2pos


def load_projected_edge_parquet(out_name: str) -> pd.DataFrame:
    return pd.read_parquet(PROJ_DIR / f"{out_name}_edges.parquet")


def build_projected_edge_indices(nodeindex2pos_prod: Dict[int, int], df_meta: pd.DataFrame):
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

    df_e = load_projected_edge_parquet("product_graph_same_group")
    edge_index_dict["same_group"] = convert_edges(df_e)
    print(f"[GNN-PROJ] same_group edges={edge_index_dict['same_group'].size(1)}")

    df_e = load_projected_edge_parquet("product_graph_same_subgroup")
    edge_index_dict["same_subgroup"] = convert_edges(df_e)
    print(f"[GNN-PROJ] same_subgroup edges={edge_index_dict['same_subgroup'].size(1)}")

    df_e = load_projected_edge_parquet("product_graph_same_plant")
    edge_index_dict["same_plant"] = convert_edges(df_e)
    print(f"[GNN-PROJ] same_plant edges={edge_index_dict['same_plant'].size(1)}")

    df_e = load_projected_edge_parquet("product_graph_same_storage")
    edge_index_dict["same_storage"] = convert_edges(df_e)
    print(f"[GNN-PROJ] same_storage edges={edge_index_dict['same_storage'].size(1)}")

    return edge_index_dict


def build_homo5type_from_parquet():
    nodes_tbl = pd.read_parquet(HOMO_DIR / "nodes_homogeneous_5type.parquet")
    edges_tbl = pd.read_parquet(HOMO_DIR / "edges_homogeneous_5type.parquet")

    nodes_tbl["node_id"] = nodes_tbl["node_id"].astype(str)

    type_groups = {}
    for nt in nodes_tbl["node_type"].unique():
        df_nt = nodes_tbl[nodes_tbl["node_type"] == nt].copy().reset_index(drop=True)
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

    df_g = edges_tbl.merge(
        nodes_tbl[["node_id", "node_type"]],
        left_on="dst",
        right_on="node_id",
        how="left",
    )

    df_pg = df_g[df_g["node_type"] == "product_group"][["src", "dst"]]
    add_edge_type("product", "product_group_edge", "product_group", df_pg)

    df_psg = df_g[df_g["node_type"] == "product_sub_group"][["src", "dst"]]
    add_edge_type("product", "product_sub_group_edge", "product_sub_group", df_psg)

    df_pl = df_g[df_g["node_type"] == "plant"][["src", "dst"]]
    add_edge_type("product", "product_plant_edge", "plant", df_pl)

    df_st = df_g[df_g["node_type"] == "storage_location"][["src", "dst"]]
    add_edge_type("product", "product_storage_edge", "storage_location", df_st)

    return edge_index_dict, num_nodes_dict, nodes_tbl


def build_hetero5type_from_parquet():
    nodes_tbl = pd.read_parquet(HETERO_DIR / "nodes_heterogeneous_5type.parquet")
    edges_tbl = pd.read_parquet(HETERO_DIR / "edges_heterogeneous_5type.parquet")

    nodes_tbl["node_id"] = nodes_tbl["node_id"].astype(str)
    type_groups = {}
    for nt in nodes_tbl["node_type"].unique():
        df_nt = nodes_tbl[nodes_tbl["node_type"] == nt].copy().reset_index(drop=True)
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
        df_et = edges_tbl[edges_tbl["edge_type"] == edge_type_name]
        src_local_fwd = []
        dst_local_fwd = []
        for _, row in df_et.iterrows():
            s = str(row["src"])
            d = str(row["dst"])
            if s not in nodeid2type or d not in nodeid2type:
                continue
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

def build_gnn_datasets_for_config(temporal_type: str, lag_window: int, df_meta: pd.DataFrame):
    print(
        f"\n=== Build GNN datasets (projected/homo5/hetero5) "
        f"from tabular: temporal_type={temporal_type}, lag_window={lag_window} ==="
    )

    df_xgb = load_xgb_tabular_for_gnn(temporal_type, lag_window, HORIZON)
    pkg_common, nodeindex2pos_prod = build_time_tensors_from_xgb_for_gnn(df_xgb)

    edge_index_proj = build_projected_edge_indices(nodeindex2pos_prod, df_meta)
    pkg_proj = {
        **pkg_common,
        "edge_index_dict": edge_index_proj,
        "graph_def": "projected_product_4view",
    }
    out_proj = GNN_DIR / f"gnn_projected_h{HORIZON}_lag{lag_window}_{temporal_type}.pt"
    torch.save(pkg_proj, out_proj)
    print(f"[GNN-SAVE] projected dataset -> {out_proj}")

    edge_index_homo5, num_nodes_homo5, nodes_homo_tbl = build_homo5type_from_parquet()
    pkg_homo5 = {
        **pkg_common,
        "edge_index_dict": edge_index_homo5,
        "num_nodes_dict": num_nodes_homo5,
        "nodes_homo_table": nodes_homo_tbl,
        "graph_def": "homogeneous_5node_types",
    }
    out_homo5 = GNN_DIR / f"gnn_homo5_h{HORIZON}_lag{lag_window}_{temporal_type}.pt"
    torch.save(pkg_homo5, out_homo5)
    print(f"[GNN-SAVE] homo5 dataset -> {out_homo5}")

    edge_index_het5, num_nodes_het5, nodes_het_tbl = build_hetero5type_from_parquet()
    pkg_het5 = {
        **pkg_common,
        "edge_index_dict": edge_index_het5,
        "num_nodes_dict": num_nodes_het5,
        "nodes_hetero_table": nodes_het_tbl,
        "graph_def": "heterogeneous_5node_types",
    }
    out_het5 = GNN_DIR / f"gnn_hetero5_h{HORIZON}_lag{lag_window}_{temporal_type}.pt"
    torch.save(pkg_het5, out_het5)
    print(f"[GNN-SAVE] hetero5 dataset -> {out_het5}")

def compute_projected_graph_features(df_meta: pd.DataFrame) -> pd.DataFrame:
    def degree_and_basic_stats(G: nx.Graph, suffix: str) -> pd.DataFrame:
        deg = dict(G.degree())
        df_deg = pd.DataFrame({"node_id": list(deg.keys()), f"deg_proj_{suffix}": list(deg.values())})

        clust = nx.clustering(G)
        df_clust = pd.DataFrame({"node_id": list(clust.keys()), f"clust_proj_{suffix}": list(clust.values())})

        between = nx.betweenness_centrality(G, k=None, normalized=True)
        df_between = pd.DataFrame({"node_id": list(between.keys()), f"btw_proj_{suffix}": list(between.values())})

        close = nx.closeness_centrality(G)
        df_close = pd.DataFrame({"node_id": list(close.keys()), f"close_proj_{suffix}": list(close.values())})

        df = df_deg.merge(df_clust, on="node_id", how="outer")
        df = df.merge(df_between, on="node_id", how="outer")
        df = df.merge(df_close, on="node_id", how="outer")
        return df

    graphs_info = [
        ("product_graph_same_group", "group"),
        ("product_graph_same_subgroup", "subgroup"),
        ("product_graph_same_plant", "plant"),
        ("product_graph_same_storage", "storage"),
    ]

    df_feat_all = None
    for fname, suffix in graphs_info:
        with open(PROJ_DIR / f"{fname}.gpickle", "rb") as f:
            G = pickle.load(f)
        df_feat = degree_and_basic_stats(G, suffix)
        df_feat["node_id"] = df_feat["node_id"].astype(str)
        if df_feat_all is None:
            df_feat_all = df_feat
        else:
            df_feat_all = df_feat_all.merge(df_feat, on="node_id", how="outer")

    df_feat_all = df_feat_all.fillna(0.0)

    df_meta_prod = df_meta[["node_id", "node_index"]].copy()
    df_meta_prod["node_id"] = df_meta_prod["node_id"].astype(str)

    df_out = df_meta_prod.merge(df_feat_all, on="node_id", how="left").fillna(0.0)
    return df_out


def compute_homo_graph_features() -> pd.DataFrame:
    with open(HOMO_DIR / "homogeneous_5node_types.gpickle", "rb") as f:
        G = pickle.load(f)

    deg_total = dict(G.degree())
    df_deg_total = pd.DataFrame({"node_id": list(deg_total.keys()), "deg_homo_total": list(deg_total.values())})

    between = nx.betweenness_centrality(G, normalized=True)
    close = nx.closeness_centrality(G)
    eig = nx.eigenvector_centrality_numpy(G)

    df_c = pd.DataFrame({
        "node_id": list(between.keys()),
        "btw_homo": list(between.values()),
        "close_homo": [close[n] for n in between.keys()],
        "eig_homo": [eig[n] for n in between.keys()],
    })

    df_feat = df_deg_total.merge(df_c, on="node_id", how="outer")

    df_nodes = pd.read_parquet(HOMO_DIR / "nodes_homogeneous_5type.parquet")
    df_nodes["node_id"] = df_nodes["node_id"].astype(str)
    df_prod = df_nodes[df_nodes["node_type"] == "product"].copy()

    neighbor_type_counts = {nid: {"pg": 0, "psg": 0, "plant": 0, "storage": 0} for nid in df_prod["node_id"]}
    for u, v in G.edges():
        u = str(u)
        v = str(v)
        if u in neighbor_type_counts:
            ntype = G.nodes[v].get("node_type", "")
            if ntype == "product_group":
                neighbor_type_counts[u]["pg"] += 1
            elif ntype == "product_sub_group":
                neighbor_type_counts[u]["psg"] += 1
            elif ntype == "plant":
                neighbor_type_counts[u]["plant"] += 1
            elif ntype == "storage_location":
                neighbor_type_counts[u]["storage"] += 1
        if v in neighbor_type_counts:
            ntype = G.nodes[u].get("node_type", "")
            if ntype == "product_group":
                neighbor_type_counts[v]["pg"] += 1
            elif ntype == "product_sub_group":
                neighbor_type_counts[v]["psg"] += 1
            elif ntype == "plant":
                neighbor_type_counts[v]["plant"] += 1
            elif ntype == "storage_location":
                neighbor_type_counts[v]["storage"] += 1

    records = []
    for nid, cnt in neighbor_type_counts.items():
        records.append(
            {
                "node_id": nid,
                "deg_homo_pg": cnt["pg"],
                "deg_homo_psg": cnt["psg"],
                "deg_homo_plant": cnt["plant"],
                "deg_homo_storage": cnt["storage"],
            }
        )
    df_deg_type = pd.DataFrame(records)

    df_feat = df_feat.merge(df_deg_type, on="node_id", how="left").fillna(0.0)

    df_feat = df_feat.merge(df_prod[["node_id", "node_index"]], on="node_id", how="inner")
    return df_feat


def compute_hetero_graph_features() -> pd.DataFrame:

    with open(HETERO_DIR / "heterogeneous_5node_types.gpickle", "rb") as f:
        G_multi: nx.MultiDiGraph = pickle.load(f)

    G_dir = nx.DiGraph()
    for u, v, data in G_multi.edges(data=True):
        if not G_dir.has_edge(u, v):
            G_dir.add_edge(u, v)

    df_nodes = pd.read_parquet(HETERO_DIR / "nodes_heterogeneous_5type.parquet")
    df_nodes["node_id"] = df_nodes["node_id"].astype(str)
    df_prod = df_nodes[df_nodes["node_type"] == "product"].copy()

    deg_in = dict(G_dir.in_degree())
    deg_out = dict(G_dir.out_degree())
    df_deg = pd.DataFrame({
        "node_id": list(deg_in.keys()),
        "deg_het_in_total": list(deg_in.values()),
        "deg_het_out_total": [deg_out.get(n, 0) for n in deg_in.keys()],
    })

    edge_type_names = [
        "product_group",
        "product_subgroup",
        "product_plant",
        "product_storage",
    ]
    deg_type_records = {nid: {et: 0 for et in edge_type_names} for nid in df_prod["node_id"].astype(str)}
    for u, v, data in G_multi.edges(data=True):
        u = str(u)
        et = data.get("edge_type", "")
        if et in edge_type_names and u in deg_type_records:
            deg_type_records[u][et] += 1

    deg_type_list = []
    for nid, d in deg_type_records.items():
        rec = {"node_id": nid}
        for et in edge_type_names:
            rec[f"deg_het_out_{et}"] = d[et]
        deg_type_list.append(rec)
    df_deg_type = pd.DataFrame(deg_type_list)

    pagerank = nx.pagerank(G_dir)
    df_pr = pd.DataFrame({"node_id": list(pagerank.keys()), "pr_het": list(pagerank.values())})

    G_und = G_dir.to_undirected()
    btw_und = nx.betweenness_centrality(G_und, normalized=True)
    df_btw = pd.DataFrame({"node_id": list(btw_und.keys()), "btw_het": list(btw_und.values())})

    df_feat = df_deg.merge(df_deg_type, on="node_id", how="outer")
    df_feat = df_feat.merge(df_pr, on="node_id", how="outer")
    df_feat = df_feat.merge(df_btw, on="node_id", how="outer")

    df_feat = df_feat.merge(df_prod[["node_id", "node_index"]], on="node_id", how="inner")
    df_feat = df_feat.fillna(0.0)
    return df_feat


def build_xgb_graph_baselines_for_config(temporal_type: str, horizon: int, lag_window: int, df_meta: pd.DataFrame):
    base_path = XGB_BASE_DIR / f"xgboost_tabular_h{horizon}_lag{lag_window}_{temporal_type}.parquet"
    if not base_path.exists():
        print(f"[XGB-GRAPH] Base XGBoost file not found: {base_path}, skip.")
        return

    df_xgb = pd.read_parquet(base_path)
    df_xgb["node_id"] = df_xgb["node_id"].astype(str)
    print("df_xgb shape:", df_xgb.shape)
    print("df_xgb unique (node_id,date):", df_xgb[["node_id","date"]].drop_duplicates().shape[0])
    print("df_xgb dup on (node_id,date):", df_xgb.duplicated(subset=["node_id","date"]).sum())
    print("df_xgb dup on (node_id,node_index):", df_xgb.duplicated(subset=["node_id","node_index"]).sum())

    df_proj_feat = compute_projected_graph_features(df_meta)
    df_proj_feat["node_id"] = df_proj_feat["node_id"].astype(str)

    # đảm bảo 1 dòng / (node_id,node_index)
    df_proj_feat = df_proj_feat.drop_duplicates(subset=["node_id", "node_index"])

    # base time series cũng đảm bảo unique trên (node_id,date)
    df_xgb = df_xgb.drop_duplicates(subset=["node_id", "node_index", "date"])

    df_proj = df_xgb.merge(
        df_proj_feat,
        on=["node_id", "node_index"],
        how="left",
        validate="m:1",  # mỗi (node_id,node_index,date) map tới 1 dòng graph feature
    ).fillna(0.0)
    out_proj = XGB_GRAPH_DIR / f"xgboost_tabular_graph_projected_h{horizon}_lag{lag_window}_{temporal_type}.parquet"
    df_proj.to_parquet(out_proj, index=False)
    print(f"[XGB-GRAPH] saved projected graph baseline -> {out_proj}")

    df_homo_feat = compute_homo_graph_features()
    df_homo_feat["node_id"] = df_homo_feat["node_id"].astype(str)
    df_homo_feat = df_homo_feat.drop_duplicates(subset=["node_id", "node_index"])
    df_homo = df_xgb.merge(
        df_homo_feat,
        on=["node_id", "node_index"],
        how="left",
        validate="m:1",
    ).fillna(0.0)
    out_homo = XGB_GRAPH_DIR / f"xgboost_tabular_graph_homo5_h{horizon}_lag{lag_window}_{temporal_type}.parquet"
    df_homo.to_parquet(out_homo, index=False)
    print(f"[XGB-GRAPH] saved homo5 graph baseline -> {out_homo}")

    df_het_feat = compute_hetero_graph_features()
    df_het_feat["node_id"] = df_het_feat["node_id"].astype(str)
    df_het_feat = df_het_feat.drop_duplicates(subset=["node_id", "node_index"])
    df_het = df_xgb.merge(
        df_het_feat,
        on=["node_id", "node_index"],
        how="left",
        validate="m:1",
    ).fillna(0.0)
    out_het = XGB_GRAPH_DIR / f"xgboost_tabular_graph_hetero5_h{horizon}_lag{lag_window}_{temporal_type}.parquet"
    df_het.to_parquet(out_het, index=False)
    print(f"[XGB-GRAPH] saved hetero5 graph baseline -> {out_het}")

def main():
    df_meta = load_node_metadata()
    sanity_check_alignment(df_meta)

    build_homogeneous_5type_graph(df_meta)
    build_heterogeneous_5type_graph(df_meta)
    build_all_projected_graphs(df_meta)

    for exp in DEFAULT_EXPERIMENTS:
        t_type = exp.temporal_type
        for H in exp.horizons:
            if H != HORIZON:
                print(
                    f"[WARN] Experiment horizon={H} != HORIZON={HORIZON} in build_graphs.py. "
                    f"Using HORIZON={HORIZON} for GNN datasets."
                )
            for L in exp.lag_windows:
                build_gnn_datasets_for_config(t_type, L, df_meta)
                build_xgb_graph_baselines_for_config(t_type, H, L, df_meta)

    print("\n[build_graphs] Done building graphs + GNN datasets + XGB graph baselines.")


if __name__ == "__main__":
    main()
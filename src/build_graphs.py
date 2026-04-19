import pandas as pd
import networkx as nx
import pickle
from pathlib import Path

from config.config import NODE_DIR, PROC_DIR

# ============================================================
# 0. Paths & constants
# ============================================================

GRAPH_DIR = PROC_DIR / "graphs"
GRAPH_DIR.mkdir(parents=True, exist_ok=True)

HOMO_DIR     = GRAPH_DIR / "homogeneous_graphs"
HETERO_DIR   = GRAPH_DIR / "heterogeneous_graphs"
PROJ_DIR     = GRAPH_DIR / "projected_product_graphs"

HOMO_DIR.mkdir(parents=True, exist_ok=True)
HETERO_DIR.mkdir(parents=True, exist_ok=True)
PROJ_DIR.mkdir(parents=True, exist_ok=True)

# Node type labels
NODE_TYPE_PRODUCT        = "product"
NODE_TYPE_PRODUCT_GROUP  = "product_group"
NODE_TYPE_PRODUCT_SUBGR  = "product_sub_group"
NODE_TYPE_PLANT          = "plant"
NODE_TYPE_STORAGE        = "storage_location"

# Edge type labels (cho heterogeneous graph)
EDGE_TYPE_PRODUCT_GROUP    = "product_group"
EDGE_TYPE_PRODUCT_SUBGROUP = "product_subgroup"
EDGE_TYPE_PRODUCT_PLANT    = "product_plant"
EDGE_TYPE_PRODUCT_STORAGE  = "product_storage"

# ============================================================
# 1. Load node metadata (align với pipeline timeseries)
# ============================================================

def load_node_metadata() -> pd.DataFrame:
    """
    Load node_id, node_index, group, sub_group, plant, storage_location.
    Khớp logic build_base_raw / build_gru_sequence / xgboost_tabular.
    """
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

    out_meta = PROC_DIR / "nodes_meta.parquet"
    df_meta.to_parquet(out_meta, index=False)
    print(f"[build_graphs] Saved node metadata to {out_meta}")

    return df_meta

# ============================================================
# 2. Homogeneous graph: 5 node types, 1 edge type
# ============================================================

def build_homogeneous_5type_graph(df_meta: pd.DataFrame) -> nx.Graph:
    """
    Homogeneous graph (1 loại edge, nhiều loại node).
    Node types:
      - product
      - product_group
      - product_sub_group
      - plant
      - storage_location

    Edges (untyped, 1 loại):
      - product -- product_group
      - product -- product_sub_group
      - product -- plant
      - product -- storage_location
    """
    G = nx.Graph()

    # -------- Product nodes --------
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

    # -------- Other node types --------

    # product_group nodes
    unique_groups = (
        df_meta["group"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    for g in unique_groups:
        nid = f"GROUP::{g}"
        G.add_node(
            nid,
            node_type=NODE_TYPE_PRODUCT_GROUP,
            raw_value=g,
        )

    # product_sub_group nodes
    unique_subgroups = (
        df_meta["sub_group"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    for sg in unique_subgroups:
        nid = f"SUBGROUP::{sg}"
        G.add_node(
            nid,
            node_type=NODE_TYPE_PRODUCT_SUBGR,
            raw_value=sg,
        )

    # plant nodes
    unique_plants = (
        df_meta["plant"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    for p in unique_plants:
        nid = f"PLANT::{p}"
        G.add_node(
            nid,
            node_type=NODE_TYPE_PLANT,
            raw_value=p,
        )

    # storage_location nodes
    unique_storages = (
        df_meta["storage_location"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    for s in unique_storages:
        nid = f"STORAGE::{s}"
        G.add_node(
            nid,
            node_type=NODE_TYPE_STORAGE,
            raw_value=s,
        )

    # -------- Edges product -- others (no edge_type) --------

    # product -- product_group
    df_g = df_meta.dropna(subset=["group"])
    for _, row in df_g.iterrows():
        prod_id = row["node_id"]
        g_val = str(row["group"])
        g_node = f"GROUP::{g_val}"
        if G.has_node(g_node):
            G.add_edge(prod_id, g_node)

    # product -- product_sub_group
    df_sg = df_meta.dropna(subset=["sub_group"])
    for _, row in df_sg.iterrows():
        prod_id = row["node_id"]
        sg_val = str(row["sub_group"])
        sg_node = f"SUBGROUP::{sg_val}"
        if G.has_node(sg_node):
            G.add_edge(prod_id, sg_node)

    # product -- plant
    df_p = df_meta.dropna(subset=["plant"])
    for _, row in df_p.iterrows():
        prod_id = row["node_id"]
        p_val = str(row["plant"])
        p_node = f"PLANT::{p_val}"
        if G.has_node(p_node):
            G.add_edge(prod_id, p_node)

    # product -- storage_location
    df_st = df_meta.dropna(subset=["storage_location"])
    for _, row in df_st.iterrows():
        prod_id = row["node_id"]
        st_val = str(row["storage_location"])
        st_node = f"STORAGE::{st_val}"
        if G.has_node(st_node):
            G.add_edge(prod_id, st_node)

    # -------- Save --------
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    print(f"[Homogeneous-5type] |V|={n_nodes}, |E|={n_edges}")

    out_gpickle = HOMO_DIR / "homogeneous_5node_types.gpickle"
    with open(out_gpickle, "wb") as f:
        pickle.dump(G, f)
    print(f"[Homogeneous-5type] Saved Graph (pickle) to {out_gpickle}")

    # Save node table & edge list
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
    print(f"[Homogeneous-5type] Saved node table to nodes_homogeneous_5type.parquet")

    edge_records = []
    for u, v in G.edges():
        edge_records.append({"src": u, "dst": v})
    df_edges = pd.DataFrame(edge_records)
    df_edges.to_parquet(HOMO_DIR / "edges_homogeneous_5type.parquet", index=False)
    print(f"[Homogeneous-5type] Saved edge list to edges_homogeneous_5type.parquet")

    return G

# ============================================================
# 3. Heterogeneous graph: 5 node types, labeled edge types
# ============================================================

def build_heterogeneous_5type_graph(df_meta: pd.DataFrame) -> nx.MultiDiGraph:
    """
    Heterogeneous graph:
      - 5 loại node:
          product, product_group, product_sub_group, plant, storage_location.
      - Edge có thuộc tính edge_type ∈ {
          'product_group', 'product_subgroup', 'product_plant', 'product_storage'
        }.
      - Mỗi cặp (product, type_node) xuất hiện tối đa 1 cạnh (dùng drop_duplicates).
    """
    G = nx.MultiDiGraph()

    # -------- 1) Product nodes --------
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

    # -------- 2) Other node types --------

    unique_groups = (
        df_meta["group"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    for g in unique_groups:
        nid = f"GROUP::{g}"
        G.add_node(
            nid,
            node_type=NODE_TYPE_PRODUCT_GROUP,
            raw_value=g,
        )

    unique_subgroups = (
        df_meta["sub_group"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    for sg in unique_subgroups:
        nid = f"SUBGROUP::{sg}"
        G.add_node(
            nid,
            node_type=NODE_TYPE_PRODUCT_SUBGR,
            raw_value=sg,
        )

    unique_plants = (
        df_meta["plant"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    for p in unique_plants:
        nid = f"PLANT::{p}"
        G.add_node(
            nid,
            node_type=NODE_TYPE_PLANT,
            raw_value=p,
        )

    unique_storages = (
        df_meta["storage_location"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    for s in unique_storages:
        nid = f"STORAGE::{s}"
        G.add_node(
            nid,
            node_type=NODE_TYPE_STORAGE,
            raw_value=s,
        )

    # -------- 3) Edges với edge_type (unique pair product–type_node) --------

    # product -- product_group
    df_g = df_meta.dropna(subset=["group"])[["node_id", "group"]].drop_duplicates()
    for _, row in df_g.iterrows():
        prod_id = row["node_id"]
        g_val = str(row["group"])
        g_node = f"GROUP::{g_val}"
        if G.has_node(g_node):
            G.add_edge(
                prod_id,
                g_node,
                edge_type=EDGE_TYPE_PRODUCT_GROUP,
            )

    # product -- product_sub_group
    df_sg = df_meta.dropna(subset=["sub_group"])[["node_id", "sub_group"]].drop_duplicates()
    for _, row in df_sg.iterrows():
        prod_id = row["node_id"]
        sg_val = str(row["sub_group"])
        sg_node = f"SUBGROUP::{sg_val}"
        if G.has_node(sg_node):
            G.add_edge(
                prod_id,
                sg_node,
                edge_type=EDGE_TYPE_PRODUCT_SUBGROUP,
            )

    # product -- plant
    df_p = df_meta.dropna(subset=["plant"])[["node_id", "plant"]].drop_duplicates()
    for _, row in df_p.iterrows():
        prod_id = row["node_id"]
        p_val = str(row["plant"])
        p_node = f"PLANT::{p_val}"
        if G.has_node(p_node):
            G.add_edge(
                prod_id,
                p_node,
                edge_type=EDGE_TYPE_PRODUCT_PLANT,
            )

    # product -- storage_location
    df_st = (
        df_meta.dropna(subset=["storage_location"])[["node_id", "storage_location"]]
        .drop_duplicates()
    )
    for _, row in df_st.iterrows():
        prod_id = row["node_id"]
        st_val = str(row["storage_location"])
        st_node = f"STORAGE::{st_val}"
        if G.has_node(st_node):
            G.add_edge(
                prod_id,
                st_node,
                edge_type=EDGE_TYPE_PRODUCT_STORAGE,
            )

    # -------- 4) Save --------
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    print(f"[Heterogeneous-5type] |V|={n_nodes}, |E|={n_edges}")

    out_gpickle = HETERO_DIR / "heterogeneous_5node_types.gpickle"
    with open(out_gpickle, "wb") as f:
        pickle.dump(G, f)
    print(f"[Heterogeneous-5type] Saved MultiDiGraph (pickle) to {out_gpickle}")

    # Save node table
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
    print(f"[Heterogeneous-5type] Saved node table to nodes_heterogeneous_5type.parquet")

    # Save edge list với edge_type
    edge_records = []
    for u, v, data in G.edges(data=True):
        edge_records.append(
            {
                "src": u,
                "dst": v,
                "edge_type": data.get("edge_type", "unknown"),
            }
        )
    df_edges = pd.DataFrame(edge_records)
    df_edges.to_parquet(HETERO_DIR / "edges_heterogeneous_5type.parquet", index=False)
    print(f"[Heterogeneous-5type] Saved edge list to edges_heterogeneous_5type.parquet")

    return G
# ============================================================
# 4. Projected product graphs
# ============================================================

def build_projected_graph(df_meta: pd.DataFrame, by_col: str, out_name: str) -> nx.Graph:
    """
    Product graph projected theo 1 thuộc tính:
      - Node: product (node_id, node_index, meta attrs).
      - Edge (untyped): giữa 2 product nếu cùng giá trị by_col.
    by_col ∈ {"group", "sub_group", "plant", "storage_location"}.
    """
    G = nx.Graph()

    # Add tất cả product nodes
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

    # Save pickle
    gpath = PROJ_DIR / f"{out_name}.gpickle"
    with open(gpath, "wb") as f:
        pickle.dump(G, f)
    print(f"[Projected] Saved graph (pickle) to {gpath}")

    # Save edge list + node table
    edges = []
    for u, v in G.edges():
        edges.append({"src": u, "dst": v})
    df_edges = pd.DataFrame(edges)
    df_edges.to_parquet(PROJ_DIR / f"{out_name}_edges.parquet", index=False)

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
    df_nodes = pd.DataFrame(nodes)
    df_nodes.to_parquet(PROJ_DIR / f"{out_name}_nodes.parquet", index=False)

    print(f"[Projected] Saved nodes/edges parquet for {out_name} to {PROJ_DIR}")
    return G

def build_all_projected_graphs(df_meta: pd.DataFrame):
    """
    4 projected product graphs:
      - same_group
      - same_subgroup
      - same_plant
      - same_storage
    """
    build_projected_graph(df_meta, by_col="group",             out_name="product_graph_same_group")
    build_projected_graph(df_meta, by_col="sub_group",         out_name="product_graph_same_subgroup")
    build_projected_graph(df_meta, by_col="plant",             out_name="product_graph_same_plant")
    build_projected_graph(df_meta, by_col="storage_location",  out_name="product_graph_same_storage")

    print(f"[Projected] Built all 4 projected product graphs in {PROJ_DIR}")

# ============================================================
# 5. Sanity check alignment với timeseries
# ============================================================

def sanity_check_alignment(df_meta: pd.DataFrame):
    """
    Kiểm tra alignment node_index với base_raw_{temporal_type}.parquet nếu tồn tại.
    Đảm bảo product nodes (node_type='product') khớp với timeseries nodes.
    """
    for temporal_type in ["unit", "weight"]:
        raw_path = PROC_DIR / f"base_raw_{temporal_type}.parquet"
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

# ============================================================
# 6. Main
# ============================================================

def main():
    df_meta = load_node_metadata()

    # 1) sanity check alignment product nodes với timeseries
    sanity_check_alignment(df_meta)

    # 2) build homogeneous graph với 5 loại node, 1 edge type
    build_homogeneous_5type_graph(df_meta)

    # 3) build heterogeneous graph với 5 loại node, labeled edges
    build_heterogeneous_5type_graph(df_meta)

    # 4) build 4 projected product graphs
    build_all_projected_graphs(df_meta)

    print("\n[build_graphs] Done building homogeneous + heterogeneous + projected graphs.")

if __name__ == "__main__":
    main()
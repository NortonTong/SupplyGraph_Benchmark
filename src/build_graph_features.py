# build_graph_features.py

import pandas as pd
import networkx as nx
import pickle
from pathlib import Path

from config.config import PROC_DIR

GRAPH_DIR   = PROC_DIR / "graphs"
HOMO_DIR    = GRAPH_DIR / "homogeneous_graphs"
HETERO_DIR  = GRAPH_DIR / "heterogeneous_graphs"
PROJ_DIR    = GRAPH_DIR / "projected_product_graphs"

def compute_graph_stats(
    G: nx.Graph,
    product_nodes: list[str],
    prefix: str = "",
    compute_betweenness: bool = False,
    compute_eigenvector: bool = False,
    compute_pagerank: bool = False,
) -> pd.DataFrame:
    """
    Tính graph statistics cho product_nodes trên graph G (undirected hoặc directed).
    prefix: prefix cho tên cột feature (ví dụ 'proj_group_', 'homo_').
    """
    # Nếu G là directed, dùng phiên bản vô hướng cho phần lớn centrality
    if isinstance(G, (nx.DiGraph, nx.MultiDiGraph)):
        G_und = nx.Graph()
        G_und.add_nodes_from(G.nodes(data=True))
        # gộp cạnh
        for u, v in G.edges():
            if not G_und.has_edge(u, v):
                G_und.add_edge(u, v)
    else:
        G_und = G

    # Đảm bảo product_nodes nằm trong graph
    product_nodes = [n for n in product_nodes if n in G_und.nodes()]
    if not product_nodes:
        return pd.DataFrame(columns=["node_id"])

    # Degree
    deg_dict = dict(G_und.degree(product_nodes))

    # Clustering (chỉ meaningful với undirected)
    clust_all = nx.clustering(G_und)
    clust_dict = {n: clust_all.get(n, 0.0) for n in product_nodes}

    # Betweenness (optional)
    if compute_betweenness:
        btw_all = nx.betweenness_centrality(G_und, normalized=True)
        btw_dict = {n: btw_all.get(n, 0.0) for n in product_nodes}
    else:
        btw_dict = {n: 0.0 for n in product_nodes}

    # Eigenvector (optional)
    if compute_eigenvector:
        try:
            eig_all = nx.eigenvector_centrality(G_und, max_iter=1000)
            eig_dict = {n: eig_all.get(n, 0.0) for n in product_nodes}
        except Exception:
            eig_dict = {n: 0.0 for n in product_nodes}
    else:
        eig_dict = {n: 0.0 for n in product_nodes}

    # PageRank (optional)
    if compute_pagerank:
        # PageRank cần directed; nếu không có, dùng phiên bản directed từ undirected
        if isinstance(G, (nx.DiGraph, nx.MultiDiGraph)):
            G_pr = G
        else:
            G_pr = nx.DiGraph()
            G_pr.add_nodes_from(G_und.nodes())
            for u, v in G_und.edges():
                G_pr.add_edge(u, v)
                G_pr.add_edge(v, u)
        pr_all = nx.pagerank(G_pr)
        pr_dict = {n: pr_all.get(n, 0.0) for n in product_nodes}
    else:
        pr_dict = {n: 0.0 for n in product_nodes}

    rows = []
    for n in product_nodes:
        rows.append(
            {
                "node_id": n,
                f"{prefix}deg": float(deg_dict.get(n, 0.0)),
                f"{prefix}clust": float(clust_dict.get(n, 0.0)),
                f"{prefix}btw": float(btw_dict.get(n, 0.0)),
                f"{prefix}eig": float(eig_dict.get(n, 0.0)),
                f"{prefix}pr": float(pr_dict.get(n, 0.0)),
            }
        )

    return pd.DataFrame(rows)

def build_graph_features_projected(
    compute_betweenness: bool = False,
    compute_eigenvector: bool = False,
    compute_pagerank: bool = False,
) -> pd.DataFrame:
    """
    Tính graph stats cho product nodes trên 4 projected graphs:
      - same_group
      - same_subgroup
      - same_plant
      - same_storage
    Rồi concat features lại thành 1 bảng.
    """
    # Load 1 graph để lấy danh sách product nodes (node_id)
    base_gpath = PROJ_DIR / "product_graph_same_group.gpickle"
    with open(base_gpath, "rb") as f:
        G0 = pickle.load(f)
    product_nodes = [n for n in G0.nodes() if not isinstance(n, str) or not ("::" in n)]

    # same_group
    with open(PROJ_DIR / "product_graph_same_group.gpickle", "rb") as f:
        G_group = pickle.load(f)
    df_g = compute_graph_stats(
        G_group,
        product_nodes,
        prefix="proj_group_",
        compute_betweenness=compute_betweenness,
        compute_eigenvector=compute_eigenvector,
        compute_pagerank=compute_pagerank,
    )

    # same_subgroup
    with open(PROJ_DIR / "product_graph_same_subgroup.gpickle", "rb") as f:
        G_sub = pickle.load(f)
    df_sg = compute_graph_stats(
        G_sub,
        product_nodes,
        prefix="proj_subgrp_",
        compute_betweenness=compute_betweenness,
        compute_eigenvector=compute_eigenvector,
        compute_pagerank=compute_pagerank,
    )

    # same_plant
    with open(PROJ_DIR / "product_graph_same_plant.gpickle", "rb") as f:
        G_pl = pickle.load(f)
    df_pl = compute_graph_stats(
        G_pl,
        product_nodes,
        prefix="proj_plant_",
        compute_betweenness=compute_betweenness,
        compute_eigenvector=compute_eigenvector,
        compute_pagerank=compute_pagerank,
    )

    # same_storage
    with open(PROJ_DIR / "product_graph_same_storage.gpickle", "rb") as f:
        G_st = pickle.load(f)
    df_st = compute_graph_stats(
        G_st,
        product_nodes,
        prefix="proj_storage_",
        compute_betweenness=compute_betweenness,
        compute_eigenvector=compute_eigenvector,
        compute_pagerank=compute_pagerank,
    )

    # Merge theo node_id
    df = df_g.merge(df_sg, on="node_id", how="outer")
    df = df.merge(df_pl, on="node_id", how="outer")
    df = df.merge(df_st, on="node_id", how="outer")

    # Fill NaN -> 0
    df = df.fillna(0.0)

    out_path = GRAPH_DIR / "graph_features_projected.parquet"
    df.to_parquet(out_path, index=False)
    print(f"[GraphFeat] Saved projected graph features to {out_path}")

    return df

def build_graph_features_homogeneous(
    compute_betweenness: bool = False,
    compute_eigenvector: bool = False,
    compute_pagerank: bool = False,
) -> pd.DataFrame:
    """
    Tính graph stats trên homogeneous_5node_types.gpickle cho product nodes.
    """
    gpath = HOMO_DIR / "homogeneous_5node_types.gpickle"
    with open(gpath, "rb") as f:
        G = pickle.load(f)

    # product nodes = node_type == 'product'
    product_nodes = [
        n for n, data in G.nodes(data=True) if data.get("node_type") == "product"
    ]

    df = compute_graph_stats(
        G,
        product_nodes,
        prefix="homo_",
        compute_betweenness=compute_betweenness,
        compute_eigenvector=compute_eigenvector,
        compute_pagerank=compute_pagerank,
    )

    # Join node_index để merge với tabular
    nodes_tbl = pd.read_parquet(HOMO_DIR / "nodes_homogeneous_5type.parquet")
    df = df.merge(
        nodes_tbl[["node_id", "node_index"]],
        on="node_id",
        how="left",
    )

    out_path = GRAPH_DIR / "graph_features_homogeneous.parquet"
    df.to_parquet(out_path, index=False)
    print(f"[GraphFeat] Saved homogeneous graph features to {out_path}")

    return df

def build_graph_features_heterogeneous(
    compute_betweenness: bool = False,
    compute_eigenvector: bool = False,
    compute_pagerank: bool = False,
) -> pd.DataFrame:
    """
    Tính graph stats trên heterogeneous_5node_types.gpickle cho product nodes.
    Ở đây ta bỏ edge_type, chỉ giữ cấu trúc adjacency (union).
    """
    gpath = HETERO_DIR / "heterogeneous_5node_types.gpickle"
    with open(gpath, "rb") as f:
        G_hetero = pickle.load(f)

    # product nodes
    product_nodes = [
        n for n, data in G_hetero.nodes(data=True) if data.get("node_type") == "product"
    ]

    # Dùng union (undirected) cho stats
    G_und = nx.Graph()
    G_und.add_nodes_from(G_hetero.nodes(data=True))
    for u, v in G_hetero.edges():
        if not G_und.has_edge(u, v):
            G_und.add_edge(u, v)

    df = compute_graph_stats(
        G_und,
        product_nodes,
        prefix="hetero_",
        compute_betweenness=compute_betweenness,
        compute_eigenvector=compute_eigenvector,
        compute_pagerank=compute_pagerank,
    )

    nodes_tbl = pd.read_parquet(HETERO_DIR / "nodes_heterogeneous_5type.parquet")
    df = df.merge(
        nodes_tbl[["node_id", "node_index"]],
        on="node_id",
        how="left",
    )

    out_path = GRAPH_DIR / "graph_features_heterogeneous.parquet"
    df.to_parquet(out_path, index=False)
    print(f"[GraphFeat] Saved heterogeneous graph features to {out_path}")

    return df

def main():
    # Projected (4 graph) features
    build_graph_features_projected(
        compute_betweenness=True,
        compute_eigenvector=True,
        compute_pagerank=True,
    )

    # Homogeneous features
    build_graph_features_homogeneous(
        compute_betweenness=True,
        compute_eigenvector=True,
        compute_pagerank=True,
    )

    # Heterogeneous features
    build_graph_features_heterogeneous(
        compute_betweenness=True,
        compute_eigenvector=True,
        compute_pagerank=True,
    )

if __name__ == "__main__":
    main()
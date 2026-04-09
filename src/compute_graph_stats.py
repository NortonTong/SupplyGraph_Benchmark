import networkx as nx
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
NODE_DIR = RAW_DIR / "Nodes"
EDGE_DIR = RAW_DIR / "Edges"
PROC_DIR = DATA_DIR / "processed" / "graph_stats"

import pandas as pd
import networkx as nx
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
EDGE_DIR = RAW_DIR / "Edges"
NODE_DIR = RAW_DIR / "Nodes"
PROC_DIR = DATA_DIR / "processed"


def compute_graph_stats(edge_file: Path,
                        node_index_file: Path,
                        edge_type: str,
                        out_path: Path) -> None:
    """
    edge_file: CSV with columns [Source, Target] or similar (indices).
    node_index_file: NodesIndex.csv
    edge_type: e.g. 'plant', 'product_group', 'sub_group', 'storage_location'
    out_path: where to save CSV with prefixed columns.

    Output columns:
      node_id,
      {edge_type}_deg,
      {edge_type}_clustering,
      {edge_type}_closeness,
      {edge_type}_betweenness
    """
    edges = pd.read_csv(edge_file)
    if {"node1", "node2"}.issubset(edges.columns):
        u_col, v_col = "node1", "node2"
    else:
        raise ValueError(f"Unknown edge file format: {edge_file}")

    G = nx.Graph()
    for _, row in edges.iterrows():
        G.add_edge(int(row[u_col]), int(row[v_col]))

    deg = dict(G.degree())
    clustering = nx.clustering(G)
    closeness = nx.closeness_centrality(G)
    betweenness = nx.betweenness_centrality(G, normalized=True)

    df_stats = pd.DataFrame({
        "node_index": list(deg.keys()),
        f"{edge_type}_deg": list(deg.values()),
        f"{edge_type}_clustering": [clustering[n] for n in deg.keys()],
        f"{edge_type}_closeness": [closeness[n] for n in deg.keys()],
        f"{edge_type}_betweenness": [betweenness[n] for n in deg.keys()],
    })

    df_idx = pd.read_csv(node_index_file).rename(
        columns={"Node": "node_id", "NodeIndex": "node_index"}
    )
    df_stats = df_stats.merge(df_idx, on="node_index", how="left")

    df_stats.to_csv(out_path, index=False)
    print(f"Saved graph stats for {edge_type} to {out_path}")

def main():
    node_index_file = NODE_DIR / "NodesIndex.csv"

    compute_graph_stats(
        edge_file=EDGE_DIR / "Edges (Plant).csv",
        node_index_file=node_index_file,
        edge_type="plant",
        out_path=PROC_DIR / "graph_stats" / "graph_stats_plant.csv",
    )

    compute_graph_stats(
        edge_file=EDGE_DIR / "Edges (Product Group).csv",
        node_index_file=node_index_file,
        edge_type="group",
        out_path=PROC_DIR / "graph_stats" / "graph_stats_product_group.csv",
    )

    compute_graph_stats(
        edge_file=EDGE_DIR / "Edges (Product Sub-Group).csv",
        node_index_file=node_index_file,
        edge_type="sub_group",
        out_path=PROC_DIR / "graph_stats" / "graph_stats_sub_group.csv",
    )

    compute_graph_stats(
        edge_file=EDGE_DIR / "Edges (Storage Location).csv",
        node_index_file=node_index_file,
        edge_type="storage_location",
        out_path=PROC_DIR / "graph_stats" / "graph_stats_storage_location.csv",
    )

if __name__ == "__main__":
    main()
import torch
from pathlib import Path
from config.config import PROC_DIR
from preprocess_gnn_need_check import load_node_metadata_for_gnn, build_pyg_data_for_view

GNN_DIR = PROC_DIR / "gnn"
GNN_DIR.mkdir(parents=True, exist_ok=True)


def main():
    df_meta = load_node_metadata_for_gnn()
    edge_types = ["plant", "product_group", "sub_group", "storage_location"]

    edges = {}
    for et in edge_types:
        data = build_pyg_data_for_view(df_meta, et)
        edges[et] = data.edge_index
        print(f"{et}: edge_index shape = {data.edge_index.shape}")

    out_path = GNN_DIR / "edge_index_views.pt"
    torch.save(edges, out_path)
    print(f"Saved edge_index views to {out_path}")


if __name__ == "__main__":
    main()
# build_edge_index_aligned.py
import torch
from pathlib import Path
from config.config import PROC_DIR
from preprocess_gnn import load_node_metadata_for_gnn, build_pyg_data_for_view

GNN_DIR = PROC_DIR / "gnn"
GNN_DIR.mkdir(parents=True, exist_ok=True)


def main():
    # Dùng đúng 40 node như trong preprocess_gnn.build_gnn_embeddings_all_views
    df_meta = load_node_metadata_for_gnn()
    df_meta = df_meta.drop_duplicates(subset=["node_id"]).reset_index(drop=True)

    edge_types = ["plant", "product_group", "sub_group", "storage_location"]

    edges = {}
    for et in edge_types:
        data = build_pyg_data_for_view(df_meta, et)  # data.x, data.edge_index, ...
        edge_index = data.edge_index.long()
        edges[et] = edge_index

        min_idx = int(edge_index.min()) if edge_index.numel() > 0 else None
        max_idx = int(edge_index.max()) if edge_index.numel() > 0 else None
        num_edges = edge_index.size(1)
        nodes_in_edges = torch.unique(edge_index)
        num_nodes_in_edges = nodes_in_edges.numel()

        print(f"\n=== Edge type: {et} ===")
        print(f"edge_index shape: {edge_index.shape}  (2, E)")
        print(f"min node index: {min_idx}")
        print(f"max node index: {max_idx}")
        print(f"#edges: {num_edges}")
        print(f"#unique nodes in edges: {num_nodes_in_edges}")
        print(f"#nodes in df_meta: {len(df_meta)}")

    out_path = GNN_DIR / "edge_index_views.pt"
    torch.save(edges, out_path)
    print(f"\nSaved edge_index views to {out_path}")


if __name__ == "__main__":
    main()
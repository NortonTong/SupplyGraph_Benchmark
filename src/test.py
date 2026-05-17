import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from matplotlib.cm import get_cmap

# Giả định hàm load_node_metadata đã được import hoặc định nghĩa
# from data_preprocessing_baselines import load_node_metadata

def visualize_projected_graph(graph_path, df_meta):
    # 1. Load Graph
    with open(graph_path, "rb") as f:
        G_full = pickle.load(f)
    
    print(f"Graph loaded: {G_full.number_of_nodes()} nodes, {G_full.number_of_edges()} edges")

    # 2. Chuẩn hóa dữ liệu Metadata
    df_meta = df_meta.copy()
    df_meta["node_id"] = df_meta["node_id"].astype(str)
    node2group = df_meta.set_index("node_id")["group"].to_dict()
    
    # Lấy danh sách group duy nhất (loại bỏ NaN)
    all_groups = sorted([g for g in set(node2group.values()) if pd.notna(g)])
    
    # 3. Thiết lập màu sắc (Sử dụng tab20 cho nhiều nhóm)
    cmap = get_cmap("tab20")
    group_colors = {group: cmap(i % 20) for i, group in enumerate(all_groups)}

    # 4. Tính toán Layout theo cụm (Grid layout cho các groups)
    pos = {}
    final_node_colors = []
    nodes_to_draw = []

    # Giả định muốn xếp các nhóm trên lưới (ví dụ 5 cột)
    n_cols = 5
    spacing = 5.0 

    for i, group in enumerate(all_groups):
        # Lọc node thuộc group hiện tại mà có tồn tại trong Graph
        group_nodes = [n for n in G_full.nodes() if node2group.get(n) == group]
        
        if not group_nodes:
            continue

        # Tạo subgraph để tính layout cục bộ
        subgraph = G_full.subgraph(group_nodes)
        
        # Tính layout cục bộ (Spring layout giúp các node trong nhóm giãn đều)
        local_pos = nx.spring_layout(subgraph, k=1.2, iterations=100, seed=42)

        # Dịch chuyển (offset) vị trí của cả nhóm trên lưới tọa độ
        dx = (i % n_cols) * spacing
        dy = (i // n_cols) * spacing
        
        for node, (x, y) in local_pos.items():
            pos[node] = (x + dx, y + dy)
            final_node_colors.append(group_colors[group])
            nodes_to_draw.append(node)

    # 5. Vẽ đồ thị
    plt.figure(figsize=(12, 10), dpi=150)
    
    # Chỉ vẽ những node đã được phân cụm và có vị trí
    G_sub = G_full.subgraph(nodes_to_draw)
    
    nx.draw(
        G_sub,
        pos=pos,
        node_color=final_node_colors,
        with_labels=False,
        node_size=60,
        edge_color="#cccccc", # Màu xám nhạt cho tinh tế
        width=0.4,
        alpha=0.9
    )

    plt.title("Projected Product Graph – Grouped by Category", fontsize=15, pad=20)
    plt.axis("off")
    
    # 6. Lưu và hiển thị
    output_path = "all_groups_clusters.png"
    plt.savefig(output_path, bbox_inches="tight")
    plt.show()
    print(f"Hình ảnh đã được lưu tại: {output_path}")

# Cách sử dụng:
# meta_data = load_node_metadata()
# visualize_projected_graph("path_to_your_graph.gpickle", meta_data)
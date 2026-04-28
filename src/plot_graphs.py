"""
plot_all_graphs.py

Script to visualize SCGraph-Bench graphs and save PNG files:
- Projected product graphs (4 views)
- Homogeneous 5-type graph (simple)
- Homogeneous 5-type graph (typed: node shapes + colors + edge styles)
- Heterogeneous 5-type graph (ego-subgraph)

Usage:
    python plot_all_graphs.py

Assumptions:
- The graphs have already been built by the preprocessing pipeline
  and saved under:
    PROC_DIR/graphs/projected_product_graphs
    PROC_DIR/graphs/homogeneous_graphs
    PROC_DIR/graphs/heterogeneous_graphs
"""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from config.config import PROC_DIR  # project-specific config for processed data root


# -------------------------------------------------------------------
# Directory configuration
# -------------------------------------------------------------------

# Root directory for all graph artifacts
GRAPH_DIR = PROC_DIR / "graphs"

# Projected product graphs (same group / subgroup / plant / storage)
PROJ_DIR = GRAPH_DIR / "projected_product_graphs"

# Homogeneous 5-type graph (product + 4 side node types in one Graph)
HOMO_DIR = GRAPH_DIR / "homogeneous_graphs"

# Heterogeneous 5-type graph (MultiDiGraph with edge_type)
HETERO_DIR = GRAPH_DIR / "heterogeneous_graphs"


# Node type constants (must match data_preprocessing_baselines.py)
NODE_TYPE_PRODUCT = "product"
NODE_TYPE_PRODUCT_GROUP = "product_group"
NODE_TYPE_PRODUCT_SUBGR = "product_sub_group"
NODE_TYPE_PLANT = "plant"
NODE_TYPE_STORAGE = "storage_location"


# -------------------------------------------------------------------
# Matplotlib helpers
# -------------------------------------------------------------------

def init_matplotlib():
    """Set common matplotlib defaults for all plots (paper-ready)."""
    # Typical 1-column width ~3.5 inch, 2-column ~7 inch
    plt.rcParams["figure.figsize"] = (3.5, 3.0)   # default; có thể override cho từng plot
    plt.rcParams["axes.titlesize"] = 8
    plt.rcParams["axes.labelsize"] = 7
    plt.rcParams["xtick.labelsize"] = 6
    plt.rcParams["ytick.labelsize"] = 6
    plt.rcParams["legend.fontsize"] = 6

# -------------------------------------------------------------------
# 1) Projected product graphs (4 views)
# -------------------------------------------------------------------

def plot_projected_graph(
    gpickle_path: Path,
    title: str,
    out_png: Path,
    max_nodes: int = 80,
    seed: int = 42,
):
    """
    Plot a projected product graph and save as PNG.

    The projected graphs connect products that share a given attribute:
    - same group
    - same subgroup
    - same plant
    - same storage location

    Parameters
    ----------
    gpickle_path : Path
        Path to the .gpickle file storing the NetworkX Graph.
    title : str
        Title to show on the figure.
    out_png : Path
        Output PNG path.
    max_nodes : int, optional
        Maximum number of nodes to display (for readability).
    seed : int, optional
        Random seed for the layout.
    """
    init_matplotlib()

    if not gpickle_path.exists():
        print(f"[ERROR][PROJECTED] File not found: {gpickle_path}")
        return

    with open(gpickle_path, "rb") as f:
        G = pickle.load(f)

    print(
        f"[INFO][PROJECTED] {gpickle_path.name} "
        f"|V|={G.number_of_nodes()}, |E|={G.number_of_edges()}"
    )

    # Take a small induced subgraph for plotting, to keep the figure readable
    nodes = list(G.nodes())
    if len(nodes) > max_nodes:
        nodes = nodes[:max_nodes]
    G_small = G.subgraph(nodes).copy()

    # Spring layout to roughly separate clusters
    pos = nx.spring_layout(G_small, seed=seed, k=0.5)

    plt.figure()
    nx.draw_networkx_nodes(G_small, pos, node_size=40, node_color="#1f77b4")
    nx.draw_networkx_edges(G_small, pos, alpha=0.4, width=0.5)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=250)
    plt.close()
    print(f"[SAVE][PROJECTED] {title} -> {out_png}")


# -------------------------------------------------------------------
# 2) Homogeneous 5-type graph (simple ego-plot)
# -------------------------------------------------------------------

def plot_homogeneous_5type_graph(
    gpickle_path: Path,
    out_png: Path,
    max_products: int = 30,
    seed: int = 42,
):
    """
    Plot a simple ego-subgraph of the homogeneous 5-type graph.

    The homogeneous graph contains:
    - product nodes
    - product_group nodes
    - product_sub_group nodes
    - plant nodes
    - storage_location nodes

    All are stored in a single NetworkX Graph with a 'node_type' attribute.

    This function:
    - selects a subset of product nodes,
    - takes their immediate neighbors,
    - visualizes the resulting subgraph with colors per node type.
    """
    init_matplotlib()

    if not gpickle_path.exists():
        print(f"[ERROR][HOMO] File not found: {gpickle_path}")
        return

    with open(gpickle_path, "rb") as f:
        G = pickle.load(f)

    print(
        f"[INFO][HOMO] {gpickle_path.name} "
        f"|V|={G.number_of_nodes()}, |E|={G.number_of_edges()}"
    )
    print("[INFO][HOMO] node_types =", set(d.get("node_type") for _, d in G.nodes(data=True)))

    # Select a subset of product nodes to serve as centers
    product_nodes = [
        n for n, d in G.nodes(data=True)
        if d.get("node_type") == NODE_TYPE_PRODUCT
    ]
    print("[INFO][HOMO] #product_nodes =", len(product_nodes))

    product_nodes = product_nodes[:max_products]

    # Collect neighbors of those products
    sample_nodes = set(product_nodes)
    for p in product_nodes:
        for nbr in G.neighbors(p):
            sample_nodes.add(nbr)

    G_small = G.subgraph(sample_nodes).copy()
    print(
        "[INFO][HOMO] ego-subgraph |V|=",
        G_small.number_of_nodes(),
        "|E|=",
        G_small.number_of_edges(),
    )

    # Color map by node type
    color_map = {
        NODE_TYPE_PRODUCT: "#1f77b4",
        NODE_TYPE_PRODUCT_GROUP: "#ff7f0e",
        NODE_TYPE_PRODUCT_SUBGR: "#2ca02c",
        NODE_TYPE_PLANT: "#d62728",
        NODE_TYPE_STORAGE: "#9467bd",
    }

    node_colors = []
    for _, data in G_small.nodes(data=True):
        nt = data.get("node_type", "unknown")
        node_colors.append(color_map.get(nt, "#7f7f7f"))

    pos = nx.spring_layout(G_small, seed=seed, k=0.6)

    plt.figure()
    nx.draw_networkx_nodes(G_small, pos, node_size=60, node_color=node_colors)
    nx.draw_networkx_edges(G_small, pos, alpha=0.4, width=0.5)
    plt.title("Homogeneous 5-type Graph – Sample Ego Network")
    plt.axis("off")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=250)
    plt.close()
    print(f"[SAVE][HOMO] Homogeneous 5-type -> {out_png}")


# -------------------------------------------------------------------
# 2b) Homogeneous 5-type graph (typed: shapes + styles)
# -------------------------------------------------------------------
def plot_homogeneous_5type_typed(
    gpickle_path: Path,
    out_png: Path,
    max_products: int = 40,
    per_type_limit: int = 5,
    max_nodes: int = 20,
    seed: int = 42,
):
    """
    Homogeneous 5-type graph visualization (small, paper-friendly).


    - Node types: product, product_group, product_sub_group, plant, storage_location
      (stored in 'node_type').
    - We:
      * choose a few product centers,
      * take their neighbors (ego-subgraph),
      * then subsample up to `per_type_limit` nodes per type,
      * and ensure total nodes <= max_nodes.
    """
    init_matplotlib()

    if not gpickle_path.exists():
        print(f"[ERROR][HOMO-TYPED] File not found: {gpickle_path}")
        return

    with open(gpickle_path, "rb") as f:
        G = pickle.load(f)

    print(
        f"[INFO][HOMO-TYPED] {gpickle_path.name} "
        f"|V|={G.number_of_nodes()}, |E|={G.number_of_edges()}"
    )
    print("[INFO][HOMO-TYPED] node_types =", set(d.get("node_type") for _, d in G.nodes(data=True)))

    # Choose a subset of product nodes as centers
    product_nodes = [
        n for n, d in G.nodes(data=True)
        if d.get("node_type") == NODE_TYPE_PRODUCT
    ]
    product_nodes = product_nodes[:max_products]

    # Ego-subgraph around these products
    sample_nodes = set(product_nodes)
    for p in product_nodes:
        for nbr in G.neighbors(p):
            sample_nodes.add(nbr)

    G_ego = G.subgraph(sample_nodes).copy()
    print(
        "[INFO][HOMO-TYPED] ego-subgraph |V|=",
        G_ego.number_of_nodes(),
        "|E|=",
        G_ego.number_of_edges(),
    )

    # Subsample nodes: up to per_type_limit nodes per node_type
    rng = np.random.default_rng(seed)
    nodes_by_type = {}
    for n, d in G_ego.nodes(data=True):
        nt = d.get("node_type", "unknown")
        nodes_by_type.setdefault(nt, []).append(n)

    selected_nodes = []
    for nt, nodes in nodes_by_type.items():
        rng.shuffle(nodes)
        selected_nodes.extend(nodes[:per_type_limit])

    # If we still exceed max_nodes, randomly drop some nodes
    if len(selected_nodes) > max_nodes:
        rng.shuffle(selected_nodes)
        selected_nodes = selected_nodes[:max_nodes]

    G_small = G_ego.subgraph(selected_nodes).copy()
    print(
        "[INFO][HOMO-TYPED] pruned subgraph |V|=",
        G_small.number_of_nodes(),
        "|E|=",
        G_small.number_of_edges(),
    )

    # Node type -> shape and color
    node_shapes = {
        NODE_TYPE_PRODUCT: "o",
        NODE_TYPE_PRODUCT_GROUP: "s",
        NODE_TYPE_PRODUCT_SUBGR: "D",
        NODE_TYPE_PLANT: "^",
        NODE_TYPE_STORAGE: "v",
    }
    node_colors = {
        NODE_TYPE_PRODUCT: "bisque",
        NODE_TYPE_PRODUCT_GROUP: "cyan",
        NODE_TYPE_PRODUCT_SUBGR: "lightgreen",
        NODE_TYPE_PLANT: "plum",
        NODE_TYPE_STORAGE: "lightcoral",
    }

    pos = nx.spring_layout(G_small, seed=seed, k=0.6)

    plt.figure(figsize=(3.2, 3.0))

    # Draw nodes grouped by node_type
    for nt, shape in node_shapes.items():
        nodes_nt = [n for n, d in G_small.nodes(data=True) if d.get("node_type") == nt]
        if not nodes_nt:
            continue
        nx.draw_networkx_nodes(
            G_small,
            pos,
            nodelist=nodes_nt,
            node_shape=shape,
            node_color=node_colors.get(nt, "gray"),
            label=nt,
            node_size=150,
        )

    # All edges same style
    nx.draw_networkx_edges(
        G_small,
        pos,
        style="solid",
        width=0.8,
        alpha=0.7,
        edge_color="gray",
    )

    # Optional labels only for products
    labels = {
        n: n for n, d in G_small.nodes(data=True)
        if d.get("node_type") == NODE_TYPE_PRODUCT
    }
    nx.draw_networkx_labels(G_small, pos, labels=labels, font_size=6)

    # Legend: node types only
    node_legend_handles = [
        plt.Line2D([0], [0], marker='o', color='black', label='Product',
                   markerfacecolor=node_colors[NODE_TYPE_PRODUCT], markersize=6),
        plt.Line2D([0], [0], marker='s', color='black', label='Product Group',
                   markerfacecolor=node_colors[NODE_TYPE_PRODUCT_GROUP], markersize=6),
        plt.Line2D([0], [0], marker='D', color='black', label='Product Subgroup',
                   markerfacecolor=node_colors[NODE_TYPE_PRODUCT_SUBGR], markersize=6),
        plt.Line2D([0], [0], marker='^', color='black', label='Plant',
                   markerfacecolor=node_colors[NODE_TYPE_PLANT], markersize=6),
        plt.Line2D([0], [0], marker='v', color='black', label='Storage Location',
                   markerfacecolor=node_colors[NODE_TYPE_STORAGE], markersize=6),
    ]

    plt.legend(handles=node_legend_handles, fontsize=6, loc='upper left')
    plt.title("Homogeneous 5-type Graph (small ego-subgraph)", fontsize=8)
    plt.axis("off")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=250)
    plt.close()
    print(f"[SAVE][HOMO-TYPED] -> {out_png}")
# -------------------------------------------------------------------
# 3) Heterogeneous 5-type graph (ego-subgraph)
# -------------------------------------------------------------------

def plot_heterogeneous_5type_graph(
    gpickle_path: Path,
    out_png: Path,
    max_products: int = 20,
    radius: int = 1,
    seed: int = 42,
):
    """
    Plot an ego-subgraph of the heterogeneous 5-type graph.

    The heterogeneous graph is stored as a NetworkX MultiDiGraph with:
    - node attribute 'node_type' (same set as homogeneous graph)
    - edge attribute 'edge_type' (one of:
        'product_group', 'product_subgroup', 'product_plant', 'product_storage')

    For visualization:
    - we convert it to a simple DiGraph, then to an undirected graph,
    - select a subset of product nodes and their neighbors within
      a given radius,
    - color nodes by type.

    Parameters
    ----------
    gpickle_path : Path
        Path to the heterogeneous_5node_types.gpickle file.
    out_png : Path
        Output PNG path.
    max_products : int, optional
        Maximum number of product centers to sample.
    radius : int, optional
        Radius for ego-graph around each center (1 or 2 recommended).
    seed : int, optional
        Random seed for layout.
    """
    init_matplotlib()

    if not gpickle_path.exists():
        print(f"[ERROR][HETERO] File not found: {gpickle_path}")
        return

    with open(gpickle_path, "rb") as f:
        G_multi: nx.MultiDiGraph = pickle.load(f)

    print(
        f"[INFO][HETERO] {gpickle_path.name} "
        f"|V|={G_multi.number_of_nodes()}, |E|={G_multi.number_of_edges()}"
    )
    print("[INFO][HETERO] node_types =", set(d.get("node_type") for _, d in G_multi.nodes(data=True)))
    print("[INFO][HETERO] edge_types =", set(d.get("edge_type") for _, _, d in G_multi.edges(data=True)))

    if G_multi.number_of_edges() == 0:
        print("[WARN][HETERO] no edges, skip plotting.")
        return

    # Convert to a simple DiGraph (merge parallel edges) then to undirected
    G_dir = nx.DiGraph()
    for u, v, data in G_multi.edges(data=True):
        if not G_dir.has_edge(u, v):
            G_dir.add_edge(u, v, **data)
    G_und = G_dir.to_undirected()

    print(
        "[INFO][HETERO] G_und |V|=",
        G_und.number_of_nodes(),
        "|E|=",
        G_und.number_of_edges(),
    )

    # Find product nodes
    product_nodes = [
        n for n, d in G_und.nodes(data=True)
        if d.get("node_type") == NODE_TYPE_PRODUCT
    ]
    print("[INFO][HETERO] #product_nodes =", len(product_nodes))

    if not product_nodes:
        # Fallback: if node_type is missing or mis-specified, take first nodes
        print("[WARN][HETERO] no product nodes found by node_type, falling back to first nodes.")
        product_nodes = list(G_und.nodes())

    product_nodes = product_nodes[:max_products]

    # Build ego-subgraph around these product nodes
    sample_nodes = set()
    for p in product_nodes:
        ego = nx.ego_graph(G_und, p, radius=radius)
        sample_nodes.update(ego.nodes())

    print("[INFO][HETERO] #nodes in ego-subgraph =", len(sample_nodes))

    if len(sample_nodes) == 0:
        print("[WARN][HETERO] ego-subgraph empty, skip plotting.")
        return

    G_small = G_und.subgraph(sample_nodes).copy()

    # Color map by node type (same as homogeneous)
    color_map = {
        NODE_TYPE_PRODUCT: "#1f77b4",
        NODE_TYPE_PRODUCT_GROUP: "#ff7f0e",
        NODE_TYPE_PRODUCT_SUBGR: "#2ca02c",
        NODE_TYPE_PLANT: "#d62728",
        NODE_TYPE_STORAGE: "#9467bd",
    }

    node_colors = []
    for _, data in G_small.nodes(data=True):
        nt = data.get("node_type", "unknown")
        node_colors.append(color_map.get(nt, "#7f7f7f"))

    pos = nx.spring_layout(G_small, seed=seed, k=0.6)

    plt.figure()
    nx.draw_networkx_nodes(G_small, pos, node_size=60, node_color=node_colors)
    nx.draw_networkx_edges(G_small, pos, alpha=0.4, width=0.5)
    plt.title(f"Heterogeneous 5-type Graph – Ego Subgraph (radius={radius})")
    plt.axis("off")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=250)
    plt.close()
    print(f"[SAVE][HETERO] Heterogeneous 5-type -> {out_png}")

def plot_heterogeneous_5type_typed(
    gpickle_path: Path,
    out_png: Path,
    max_products: int = 20,
    per_type_limit: int = 5,
    max_nodes: int = 20,
    radius: int = 1,
    seed: int = 42,
):
    """
    Heterogeneous 5-type graph visualization (small, typed).


    - Node types: product, product_group, product_sub_group, plant, storage_location
      (node attribute 'node_type').
    - Edge types: product_group, product_subgroup, product_plant, product_storage
      (edge attribute 'edge_type').
    - Build ego-subgraph around a few product nodes, then subsample up to
      `per_type_limit` nodes per type, capped at `max_nodes`.
    """
    init_matplotlib()

    if not gpickle_path.exists():
        print(f"[ERROR][HETERO-TYPED] File not found: {gpickle_path}")
        return

    with open(gpickle_path, "rb") as f:
        G_multi: nx.MultiDiGraph = pickle.load(f)

    print(
        f"[INFO][HETERO-TYPED] {gpickle_path.name} "
        f"|V|={G_multi.number_of_nodes()}, |E|={G_multi.number_of_edges()}"
    )
    print("[INFO][HETERO-TYPED] node_types in G_multi =", set(d.get("node_type") for _, d in G_multi.nodes(data=True)))
    print("[INFO][HETERO-TYPED] edge_types in G_multi =", set(d.get("edge_type") for _, _, d in G_multi.edges(data=True)))

    if G_multi.number_of_edges() == 0:
        print("[WARN][HETERO-TYPED] no edges, skip plotting.")
        return

    # Build directed graph preserving node attributes
    G_dir = nx.DiGraph()
    for n, data in G_multi.nodes(data=True):
        G_dir.add_node(n, **data)
    for u, v, data in G_multi.edges(data=True):
        if not G_dir.has_edge(u, v):
            G_dir.add_edge(u, v, **data)

    G_und = G_dir.to_undirected()

    # Product centers
    product_nodes = [
        n for n, d in G_und.nodes(data=True)
        if d.get("node_type") == NODE_TYPE_PRODUCT
    ]
    if not product_nodes:
        print("[WARN][HETERO-TYPED] no product nodes found, using first nodes.")
        product_nodes = list(G_und.nodes())
    product_nodes = product_nodes[:max_products]

    # Ego-subgraph
    sample_nodes = set()
    for p in product_nodes:
        ego = nx.ego_graph(G_und, p, radius=radius)
        sample_nodes.update(ego.nodes())

    if not sample_nodes:
        print("[WARN][HETERO-TYPED] ego-subgraph empty, skip plotting.")
        return

    G_ego = G_und.subgraph(sample_nodes).copy()

    # Subsample nodes by type
    rng = np.random.default_rng(seed)
    nodes_by_type = {}
    for n, d in G_ego.nodes(data=True):
        nt = d.get("node_type", "unknown")
        nodes_by_type.setdefault(nt, []).append(n)

    selected_nodes = []
    for nt, nodes in nodes_by_type.items():
        rng.shuffle(nodes)
        selected_nodes.extend(nodes[:per_type_limit])

    if len(selected_nodes) > max_nodes:
        rng.shuffle(selected_nodes)
        selected_nodes = selected_nodes[:max_nodes]

    G_small = G_ego.subgraph(selected_nodes).copy()
    print(
        "[INFO][HETERO-TYPED] pruned subgraph |V|=",
        G_small.number_of_nodes(),
        "|E|=",
        G_small.number_of_edges(),
    )

    # Node type -> shape + color
    node_shapes = {
        NODE_TYPE_PRODUCT: "o",
        NODE_TYPE_PRODUCT_GROUP: "s",
        NODE_TYPE_PRODUCT_SUBGR: "D",
        NODE_TYPE_PLANT: "^",
        NODE_TYPE_STORAGE: "v",
    }
    node_colors = {
        NODE_TYPE_PRODUCT: "bisque",
        NODE_TYPE_PRODUCT_GROUP: "cyan",
        NODE_TYPE_PRODUCT_SUBGR: "lightgreen",
        NODE_TYPE_PLANT: "plum",
        NODE_TYPE_STORAGE: "lightcoral",
    }

    # Edge type -> style
    edge_type_styles = {
        "product_group":    {"style": "solid",   "color": "black"},
        "product_subgroup": {"style": "dashed",  "color": "black"},
        "product_plant":    {"style": "dashdot", "color": "black"},
        "product_storage":  {"style": "dotted",  "color": "black"},
    }

    pos = nx.spring_layout(G_small, seed=seed, k=0.6)

    plt.figure(figsize=(3.2, 3.0))

    # Draw nodes grouped by type
    for nt, shape in node_shapes.items():
        nodes_nt = [n for n, d in G_small.nodes(data=True) if d.get("node_type") == nt]
        if not nodes_nt:
            continue
        nx.draw_networkx_nodes(
            G_small,
            pos,
            nodelist=nodes_nt,
            node_shape=shape,
            node_color=node_colors.get(nt, "gray"),
            label=nt,
            node_size=150,
        )

    # Group edges by edge_type
    edges_by_type = {et: [] for et in edge_type_styles.keys()}
    for u, v, data in G_small.edges(data=True):
        et = data.get("edge_type")
        if et in edges_by_type:
            edges_by_type[et].append((u, v))

    for et, edgelist in edges_by_type.items():
        if not edgelist:
            continue
        style_cfg = edge_type_styles[et]
        nx.draw_networkx_edges(
            G_small,
            pos,
            edgelist=edgelist,
            style=style_cfg["style"],
            edge_color=style_cfg["color"],
            width=1.0,
            alpha=0.9,
        )

    # Label only product nodes
    labels = {
        n: n for n, d in G_small.nodes(data=True)
        if d.get("node_type") == NODE_TYPE_PRODUCT
    }
    nx.draw_networkx_labels(G_small, pos, labels=labels, font_size=6)

    # Legends
    node_legend_handles = [
        plt.Line2D([0], [0], marker='o', color='black', label='Product',
                   markerfacecolor=node_colors[NODE_TYPE_PRODUCT], markersize=6),
        plt.Line2D([0], [0], marker='s', color='black', label='Product Group',
                   markerfacecolor=node_colors[NODE_TYPE_PRODUCT_GROUP], markersize=6),
        plt.Line2D([0], [0], marker='D', color='black', label='Product Subgroup',
                   markerfacecolor=node_colors[NODE_TYPE_PRODUCT_SUBGR], markersize=6),
        plt.Line2D([0], [0], marker='^', color='black', label='Plant',
                   markerfacecolor=node_colors[NODE_TYPE_PLANT], markersize=6),
        plt.Line2D([0], [0], marker='v', color='black', label='Storage Location',
                   markerfacecolor=node_colors[NODE_TYPE_STORAGE], markersize=6),
    ]

    edge_legend_handles = [
        plt.Line2D([0], [0], color='black', lw=1.5, linestyle='-',
                   label='Edge: product_group'),
        plt.Line2D([0], [0], color='black', lw=1.5, linestyle='--',
                   label='Edge: product_subgroup'),
        plt.Line2D([0], [0], color='black', lw=1.5, linestyle='-.',
                   label='Edge: product_plant'),
        plt.Line2D([0], [0], color='black', lw=1.5, linestyle=':',
                   label='Edge: product_storage'),
    ]

    plt.legend(handles=node_legend_handles + edge_legend_handles,
               fontsize=6, loc='upper left')

    plt.title("Heterogeneous 5-type Graph (small ego-subgraph)", fontsize=8)
    plt.axis("off")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=250)
    plt.close()
    print(f"[SAVE][HETERO-TYPED] -> {out_png}")

# -------------------------------------------------------------------
# Main entry point
# -------------------------------------------------------------------

def main():
    """Run all graph visualizations and save the resulting PNG files."""
    figs_dir = PROC_DIR / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    # 1) Projected graphs (4 views)
    plot_projected_graph(
        PROJ_DIR / "product_graph_same_group.gpickle",
        "Projected Product Graph – Same Group",
        figs_dir / "projected_same_group.png",
    )
    plot_projected_graph(
        PROJ_DIR / "product_graph_same_subgroup.gpickle",
        "Projected Product Graph – Same Subgroup",
        figs_dir / "projected_same_subgroup.png",
    )
    plot_projected_graph(
        PROJ_DIR / "product_graph_same_plant.gpickle",
        "Projected Product Graph – Same Plant",
        figs_dir / "projected_same_plant.png",
    )
    plot_projected_graph(
        PROJ_DIR / "product_graph_same_storage.gpickle",
        "Projected Product Graph – Same Storage Location",
        figs_dir / "projected_same_storage.png",
    )

  # Homogeneous (simple)
    plot_homogeneous_5type_graph(
        HOMO_DIR / "homogeneous_5node_types.gpickle",
        figs_dir / "homogeneous_5type_ego.png",
        max_products=30,
    )

    # Homogeneous (typed, small)
    plot_homogeneous_5type_typed(
        HOMO_DIR / "homogeneous_5node_types.gpickle",
        figs_dir / "homogeneous_5type_typed_small.png",
        max_products=40,
        per_type_limit=5,
        max_nodes=20,
    )

    # Heterogeneous (typed, small)
    plot_heterogeneous_5type_typed(
        HETERO_DIR / "heterogeneous_5node_types.gpickle",
        figs_dir / "heterogeneous_5type_typed_small.png",
        max_products=40,
        per_type_limit=5,
        max_nodes=20,
        radius=1,
    )
    print(f"[DONE] All figures saved under {figs_dir}")


if __name__ == "__main__":
    main()
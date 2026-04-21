import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, HeteroConv


class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = out_channels
        self.net = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# 1. ProjectedGINEncoder (single node type, projected product graph)
# ============================================================

class ProjectedGINEncoder(nn.Module):
    """
    Encoder cho projected product graph.
    - Input: x [N, F_in], edge_index [2, E]
    - Output: embedding [N, hidden_channels]
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        num_layers: int = 3,
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GINConv(MLP(in_channels, hidden_channels)))
        for _ in range(num_layers - 1):
            self.convs.append(GINConv(MLP(hidden_channels, hidden_channels)))

    def forward(self, x, edge_index):
        """
        x: [N, F_in]
        edge_index: [2, E]
        return: [N, hidden_channels]
        """
        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)
        return h


# ============================================================
# 2. HomogeneousFiveTypeGINEncoder
#    (treat all node types in one big graph, with type embedding)
# ============================================================

class HomogeneousFiveTypeGINEncoder(nn.Module):
    """
    Encoder đồng nhất 5 node-type.
    - Node types: ví dụ ['product', 'plant', 'group', 'subgroup', 'storage_location']
    - Input:
        x_dict: {node_type: [N_type, F_in]}
        edge_index: [2, E_total] trên graph flatten
    - Output:
        embedding [N_product, hidden_channels] cho node_type 'product'
    """

    def __init__(
        self,
        in_channels: int,
        num_nodes_dict: dict,
        node_type_order: list,
        hidden_channels: int = 128,
        num_layers: int = 3,
        node_type_emb_dim: int = 8,
    ):
        super().__init__()

        self.num_nodes_dict = num_nodes_dict
        self.node_type_order = node_type_order

        # Tính offsets cho từng node type trong vector concat
        offsets = {}
        offset = 0
        for nt in node_type_order:
            offsets[nt] = offset
            offset += num_nodes_dict[nt]
        self.register_buffer(
            "node_type_offsets",
            torch.tensor([offsets[nt] for nt in node_type_order], dtype=torch.long),
            persistent=False,
        )
        self.total_num_nodes = offset

        # node_type_id: [N_total] chứa index type (0..num_types-1) cho từng node
        node_type_id = torch.empty(self.total_num_nodes, dtype=torch.long)
        cur = 0
        for i, nt in enumerate(node_type_order):
            n = num_nodes_dict[nt]
            node_type_id[cur:cur + n] = i
            cur += n
        self.register_buffer("node_type_id", node_type_id, persistent=False)

        self.num_types = len(node_type_order)
        self.type_emb = nn.Embedding(self.num_types, node_type_emb_dim)

        # GIN layers trên graph đồng nhất, input = feature + type_emb
        self.convs = nn.ModuleList()
        self.convs.append(
            GINConv(MLP(in_channels + node_type_emb_dim, hidden_channels))
        )
        for _ in range(num_layers - 1):
            self.convs.append(GINConv(MLP(hidden_channels, hidden_channels)))

    def _concat_x_dict(self, x_dict: dict):
        xs = []
        for nt in self.node_type_order:
            xs.append(x_dict[nt])
        return torch.cat(xs, dim=0)

    def forward(self, x_dict: dict, edge_index):
        """
        x_dict: {node_type: [N_type, F_in]}
        edge_index: [2, E_total]
        return: [N_product, hidden_channels]
        """
        device = edge_index.device
        x_all = self._concat_x_dict(x_dict).to(device)        # [N_total, F_in]
        node_type_id = self.node_type_id.to(device)           # [N_total]
        type_emb = self.type_emb(node_type_id)                # [N_total, emb_dim]

        h = torch.cat([x_all, type_emb], dim=-1)              # [N_total, F_in+emb_dim]
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)

        # Cắt embedding cho node_type 'product'
        idx_prod_type = self.node_type_order.index("product")
        offset_prod = self.node_type_offsets[idx_prod_type].item()
        n_prod = self.num_nodes_dict["product"]
        h_prod = h[offset_prod:offset_prod + n_prod]
        return h_prod


# ============================================================
# 3. HeterogeneousGINEncoder (5-type)
# ============================================================

class HeterogeneousGINLayer(nn.Module):
    """
    Một layer Hetero GINConv trên nhiều edge types.
    """

    def __init__(self, in_channels_dict, out_channels, aggr: str = "sum"):
        super().__init__()

        # in_channels_dict:
        #   {
        #       "edge_types": List[(src_type, rel, dst_type)],
        #       node_type: in_channels,
        #   }
        edge_types = in_channels_dict["edge_types"]

        convs = {}
        for (src_type, rel, dst_type) in edge_types:
            in_ch = in_channels_dict[src_type]
            mlp = nn.Sequential(
                nn.Linear(in_ch, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels),
            )
            convs[(src_type, rel, dst_type)] = GINConv(mlp)

        self.convs = HeteroConv(convs, aggr=aggr)

    def forward(self, x_dict, edge_index_dict):
        # x_dict: {node_type: [N_type, F_in]}
        # edge_index_dict: {(src, rel, dst): [2, E]}
        h_dict = self.convs(x_dict, edge_index_dict)
        return h_dict


class HeterogeneousGINEncoder(nn.Module):
    """
    Encoder Heterogeneous GIN 5 node-type.
    - in_channels_dict:
        {
            "edge_types": [(src_type, rel, dst_type), ...],
            node_type: in_channels,
        }
    - Input:
        x_dict: {node_type: [N_type, F_in]}
        edge_index_dict: {(src_type, rel, dst_type): edge_index}
    - Output:
        embedding [N_product, hidden_channels] cho node_type 'product'
    """

    def __init__(
        self,
        in_channels_dict: dict,
        hidden_channels: int = 128,
        num_layers: int = 2,
    ):
        super().__init__()

        if "edge_types" not in in_channels_dict:
            raise ValueError(
                "in_channels_dict must contain key 'edge_types' listing edge types."
            )

        self.node_types = [nt for nt in in_channels_dict.keys() if nt != "edge_types"]

        # Projection từ input dim -> hidden dim cho từng node type
        self.node_in_proj = nn.ModuleDict()
        for nt in self.node_types:
            self.node_in_proj[nt] = nn.Linear(in_channels_dict[nt], hidden_channels)

        # Heterogeneous GIN layers
        self.layers = nn.ModuleList()
        in_chs = {
            "edge_types": in_channels_dict["edge_types"],
            **{nt: hidden_channels for nt in self.node_types},
        }
        for _ in range(num_layers):
            self.layers.append(HeterogeneousGINLayer(in_chs, hidden_channels))

    def forward(self, x_dict, edge_index_dict):
        """
        x_dict: {node_type: [N_type, F_in]}
        edge_index_dict: {(src_type, rel, dst_type): edge_index}
        return: [N_product, hidden_channels]
        """
        # initial projection
        h_dict = {}
        for nt in self.node_types:
            h_dict[nt] = F.relu(self.node_in_proj[nt](x_dict[nt]))

        # heterogeneous GIN layers
        for layer in self.layers:
            h_dict = layer(h_dict, edge_index_dict)
            for nt in h_dict.keys():
                h_dict[nt] = F.relu(h_dict[nt])

        # Trả về embedding cho node_type 'product'
        if "product" not in h_dict:
            raise KeyError("Expected node type 'product' in h_dict.")
        return h_dict["product"]
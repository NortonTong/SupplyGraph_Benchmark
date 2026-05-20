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

class ProjectedGINEncoder(nn.Module):
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
        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)
        return h

class HomogeneousFiveTypeGINEncoder(nn.Module):
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

        node_type_id = torch.empty(self.total_num_nodes, dtype=torch.long)
        cur = 0
        for i, nt in enumerate(node_type_order):
            n = num_nodes_dict[nt]
            node_type_id[cur:cur + n] = i
            cur += n
        self.register_buffer("node_type_id", node_type_id, persistent=False)

        self.num_types = len(node_type_order)
        self.type_emb = nn.Embedding(self.num_types, node_type_emb_dim)

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
        x_all = self._concat_x_dict(x_dict).to(device)        
        node_type_id = self.node_type_id.to(device)           
        type_emb = self.type_emb(node_type_id)                

        h = torch.cat([x_all, type_emb], dim=-1)              
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)
        idx_prod_type = self.node_type_order.index("product")
        offset_prod = self.node_type_offsets[idx_prod_type].item()
        n_prod = self.num_nodes_dict["product"]
        h_prod = h[offset_prod:offset_prod + n_prod]
        return h_prod

class HeterogeneousGINLayer(nn.Module):
    def __init__(self, in_channels_dict, out_channels, aggr: str = "sum"):
        super().__init__()
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
        h_dict = self.convs(x_dict, edge_index_dict)
        return h_dict


class HeterogeneousGINEncoder(nn.Module):
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

        self.node_in_proj = nn.ModuleDict()
        for nt in self.node_types:
            self.node_in_proj[nt] = nn.Linear(in_channels_dict[nt], hidden_channels)

        self.layers = nn.ModuleList()
        in_chs = {
            "edge_types": in_channels_dict["edge_types"],
            **{nt: hidden_channels for nt in self.node_types},
        }
        for _ in range(num_layers):
            self.layers.append(HeterogeneousGINLayer(in_chs, hidden_channels))

    def forward(self, x_dict, edge_index_dict):
        h_dict = {}
        for nt in self.node_types:
            h_dict[nt] = F.relu(self.node_in_proj[nt](x_dict[nt]))

        for layer in self.layers:
            h_dict = layer(h_dict, edge_index_dict)
            for nt in h_dict.keys():
                h_dict[nt] = F.relu(h_dict[nt])
        if "product" not in h_dict:
            raise KeyError("Expected node type 'product' in h_dict.")
        return h_dict["product"]
import torch
from torch import nn
from torch_geometric.nn import GINConv, HeteroConv
import torch.nn.functional as F


# ============================================================
# 1. GIN block dùng chung
# ============================================================

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
# 2. Projected GIN Regressor (single node type, projected graph)
# ============================================================

class ProjectedGINRegressor(nn.Module):
    """
    GIN cho projected product graph.
    Forward luôn trả logits trên scale training (z), KHÔNG inverse.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        num_layers: int = 3,
        is_softplus: bool = False,  # chỉ dùng để log/tag
        is_log1p: bool = False,
    ):
        super().__init__()
        if is_softplus and is_log1p:
            raise ValueError("Only one of is_softplus / is_log1p can be True.")

        self.is_softplus = is_softplus
        self.is_log1p = is_log1p

        self.convs = nn.ModuleList()
        self.convs.append(GINConv(MLP(in_channels, hidden_channels)))
        for _ in range(num_layers - 1):
            self.convs.append(GINConv(MLP(hidden_channels, hidden_channels)))

        self.out_lin = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index):
        """
        x: [N, F]
        edge_index: [2, E]
        return: [N] logits z trên scale training
        """
        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)

        head = self.out_lin(h).squeeze(-1)  # [N]
        return head  # logits


# ============================================================
# 3. Homogeneous 5-type GIN Regressor
#    (treat all node types in one big graph, with type embedding)
# ============================================================

class HomogeneousFiveTypeGINRegressor(nn.Module):
    """
    GIN đồng nhất 5 node-type, output logits z cho node 'product'.

    - in_channels: số chiều feature input cho từng node (same F_in cho mọi node-type)
    - num_nodes_dict: {node_type: num_nodes}
    - node_type_order: list các node_type theo thứ tự concat (phải trùng pipeline)
    """

    def __init__(
        self,
        in_channels: int,
        num_nodes_dict: dict,
        node_type_order: list,
        hidden_channels: int = 128,
        num_layers: int = 3,
        node_type_emb_dim: int = 8,
        is_softplus: bool = False,  # chỉ để log
        is_log1p: bool = False,
    ):
        super().__init__()
        if is_softplus and is_log1p:
            raise ValueError("Only one of is_softplus / is_log1p can be True.")

        self.num_nodes_dict = num_nodes_dict
        self.node_type_order = node_type_order
        self.is_softplus = is_softplus
        self.is_log1p = is_log1p

        # offsets cho từng node_type trong concat
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

        # node_type_id cho từng node global
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

        self.out_lin = nn.Linear(hidden_channels, 1)

    def _concat_x_dict(self, x_dict: dict):
        xs = []
        for nt in self.node_type_order:
            xs.append(x_dict[nt])
        return torch.cat(xs, dim=0)  # [N_total, F_in]

    def forward(self, x_dict: dict, edge_index):
        """
        x_dict: {node_type: [N_type, F_in]}
        edge_index: [2, E_total] với index global
        return: [N_product] logits z trên scale training
        """
        device = edge_index.device
        x_all = self._concat_x_dict(x_dict).to(device)         # [N_total, F_in]
        node_type_id = self.node_type_id.to(device)            # [N_total]
        type_emb = self.type_emb(node_type_id)                 # [N_total, emb_dim]

        h = torch.cat([x_all, type_emb], dim=-1)               # [N_total, F_in+emb_dim]

        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)

        head_all = self.out_lin(h).squeeze(-1)                 # [N_total] logits

        # Cắt output cho node_type 'product'
        idx_prod_type = self.node_type_order.index("product")
        offset_prod = self.node_type_offsets[idx_prod_type].item()
        n_prod = self.num_nodes_dict["product"]
        out_prod = head_all[offset_prod:offset_prod + n_prod]  # [N_product]
        return out_prod   # logits


# ============================================================
# 4. Heterogeneous GIN Regressor (5-type)
# ============================================================

class HeterogeneousGINLayer(nn.Module):
    def __init__(self, node_in_channels: dict, edge_types, out_channels, aggr="sum"):
        super().__init__()
        convs = {}
        for (src_type, rel, dst_type) in edge_types:
            in_ch = node_in_channels[src_type]
            mlp = nn.Sequential(
                nn.Linear(in_ch, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels),
            )
            convs[(src_type, rel, dst_type)] = GINConv(mlp)

        self.convs = HeteroConv(convs, aggr=aggr)

    def forward(self, x_dict, edge_index_dict):
        # trả về dict {node_type: h_new}, chỉ cho dst types
        return self.convs(x_dict, edge_index_dict)


class HeterogeneousGINRegressor(nn.Module):
    """
    Heterogeneous GIN 5 node-type.
    Output logits z cho node_type 'product'.

    in_channels_dict:
        {
            "edge_types": [(src_type, rel, dst_type), ...],
            node_type: in_channels,
        }
    """

    def __init__(
        self,
        in_channels_dict: dict,
        hidden_channels: int = 128,
        num_layers: int = 2,
        is_softplus: bool = False,  # chỉ để log
        is_log1p: bool = False,
    ):
        super().__init__()
        if is_softplus and is_log1p:
            raise ValueError("Only one of is_softplus / is_log1p can be True.")

        if "edge_types" not in in_channels_dict:
            raise ValueError(
                "in_channels_dict must contain key 'edge_types' listing edge types."
            )

        self.node_types = [nt for nt in in_channels_dict.keys() if nt != "edge_types"]
        self.is_softplus = is_softplus
        self.is_log1p = is_log1p

        # project input features của từng node-type về hidden_channels
        self.node_in_proj = nn.ModuleDict()
        for nt in self.node_types:
            self.node_in_proj[nt] = nn.Linear(in_channels_dict[nt], hidden_channels)

        edge_types = in_channels_dict["edge_types"]
        node_in_channels = {nt: hidden_channels for nt in self.node_types}

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                HeterogeneousGINLayer(node_in_channels, edge_types, hidden_channels)
            )

        self.out_lin = nn.Linear(hidden_channels, 1)

    def forward(self, x_dict, edge_index_dict):
        """
        x_dict: {node_type: [N_type, F_in]}
        edge_index_dict: {(src_type, rel, dst_type): edge_index}
        return: [N_product] logits z trên scale training
        """
        # init h_dict sau projection
        h_dict = {
            nt: F.relu(self.node_in_proj[nt](x_dict[nt]))
            for nt in self.node_types
        }

        # chạy qua các hetero layers, với reinjection node types bị mất
        for layer in self.layers:
            h_in = h_dict
            h_out = layer(h_in, edge_index_dict)  # chỉ chứa dst types

            new_h = {}
            for nt in self.node_types:
                if nt in h_out:
                    new_h[nt] = F.relu(h_out[nt])
                else:
                    # node-type chỉ làm source: giữ embedding cũ
                    new_h[nt] = h_in[nt]
            h_dict = new_h

        head = self.out_lin(h_dict["product"]).squeeze(-1)  # [N_product] logits
        return head

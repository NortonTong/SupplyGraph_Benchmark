import torch
from torch import nn
from torch_geometric.nn import GINConv
from torch_geometric.nn import HeteroConv
import torch.nn.functional as F

def apply_output_head(head: torch.Tensor, is_softplus: bool, is_log1p: bool) -> torch.Tensor:
    """
    head: [N], logit từ Linear.
    - is_softplus: y_hat = softplus(head)
    - is_log1p :  y_hat = expm1(head).clamp_min(0)
    - cả 2 False: y_hat = head
    """
    if is_softplus:
        return F.softplus(head)
    if is_log1p:
        return torch.expm1(head).clamp_min(0.0)
    return head
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
    - is_softplus=True : y_hat = softplus(head)      (>=0)
    - is_log1p=True   : y_hat = expm1(head).>=0
    - cả 2 False      : y_hat = head (raw)
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        num_layers: int = 3,
        is_softplus: bool = False,
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

        if self.is_softplus:
            self.softplus = nn.Softplus()

    def forward(self, x, edge_index):
        """
        x: [N, F]
        edge_index: [2, E]
        return: [N] y_hat trên scale gốc (raw/softplus/log1p-expm1)
        """
        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)

        head = self.out_lin(h).squeeze(-1)  # [N]

        if self.is_softplus:
            y_hat = self.softplus(head)
        elif self.is_log1p:
            y_hat = torch.expm1(head).clamp_min(0.0)
        else:
            y_hat = head

        return y_hat
    
#  ============================================================
# 3. Homogeneous 5-type GIN Regressor
#    (treat all node types in one big graph, with type embedding)
# ============================================================

class HomogeneousFiveTypeGINRegressor(nn.Module):
    """
    GIN đồng nhất 5 node-type, output y_hat trên scale gốc cho node 'product'.
    """

    def __init__(
        self,
        in_channels: int,
        num_nodes_dict: dict,
        node_type_order: list,
        hidden_channels: int = 128,
        num_layers: int = 3,
        node_type_emb_dim: int = 8,
        is_softplus: bool = False,
        is_log1p: bool = False,
    ):
        super().__init__()
        if is_softplus and is_log1p:
            raise ValueError("Only one of is_softplus / is_log1p can be True.")

        self.num_nodes_dict = num_nodes_dict
        self.node_type_order = node_type_order
        self.is_softplus = is_softplus
        self.is_log1p = is_log1p

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

        self.out_lin = nn.Linear(hidden_channels, 1)
        if self.is_softplus:
            self.softplus = nn.Softplus()

    def _concat_x_dict(self, x_dict: dict):
        xs = []
        for nt in self.node_type_order:
            xs.append(x_dict[nt])
        return torch.cat(xs, dim=0)

    def forward(self, x_dict: dict, edge_index):
        device = edge_index.device
        x_all = self._concat_x_dict(x_dict).to(device)
        node_type_id = self.node_type_id.to(device)
        type_emb = self.type_emb(node_type_id)

        h = torch.cat([x_all, type_emb], dim=-1)

        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)

        head_all = self.out_lin(h).squeeze(-1)  # [N_total]

        if self.is_softplus:
            head_all = self.softplus(head_all)
        elif self.is_log1p:
            head_all = torch.expm1(head_all).clamp_min(0.0)

        idx_prod_type = self.node_type_order.index("product")
        offset_prod = self.node_type_offsets[idx_prod_type].item()
        n_prod = self.num_nodes_dict["product"]
        out_prod = head_all[offset_prod:offset_prod + n_prod]
        return out_prod
    
# ============================================================
# 4. Heterogeneous GIN Regressor (5-type)
# ============================================================

class HeterogeneousGINLayer(nn.Module):
    def __init__(self, in_channels_dict, out_channels, aggr="sum"):
        super().__init__()
        from torch_geometric.nn import HeteroConv, GINConv

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
    
class HeterogeneousGINRegressor(nn.Module):
    """
    Heterogeneous GIN 5 node-type.
    Output y_hat trên scale gốc cho node_type 'product'.
    """

    def __init__(
        self,
        in_channels_dict: dict,
        hidden_channels: int = 128,
        num_layers: int = 2,
        is_softplus: bool = False,
        is_log1p: bool = False,
    ):
        super().__init__()
        if is_softplus and is_log1p:
            raise ValueError("Only one of is_softplus / is_log1p can be True.")

        self.node_types = [nt for nt in in_channels_dict.keys() if nt != "edge_types"]
        self.is_softplus = is_softplus
        self.is_log1p = is_log1p

        if "edge_types" not in in_channels_dict:
            raise ValueError(
                "in_channels_dict must contain key 'edge_types' listing edge types."
            )

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

        self.out_lin = nn.Linear(hidden_channels, 1)
        if self.is_softplus:
            self.softplus = nn.Softplus()

    def forward(self, x_dict, edge_index_dict):
        h_dict = {}
        for nt in self.node_types:
            h_dict[nt] = F.relu(self.node_in_proj[nt](x_dict[nt]))

        for layer in self.layers:
            h_dict = layer(h_dict, edge_index_dict)
            for nt in h_dict.keys():
                h_dict[nt] = F.relu(h_dict[nt])

        head = self.out_lin(h_dict["product"]).squeeze(-1)

        if self.is_softplus:
            y_hat = self.softplus(head)
        elif self.is_log1p:
            y_hat = torch.expm1(head).clamp_min(0.0)
        else:
            y_hat = head

        return y_hat
    
# ============================================================
# Helper: chuyển h_dict sang định dạng cần cho HeterogeneousGINLayer (nếu muốn tách)
# ============================================================

def x_dict_to_for_layer(h_dict):
    """
    HeteroConv forward(x_dict, edge_index_dict) yêu cầu x_dict: {node_type: [N_type, F]}.
    Hàm này hiện chỉ là identity, tách riêng để sau dễ sửa nếu cần.
    """
    return h_dict
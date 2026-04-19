import torch
from torch import nn
from torch_geometric.nn import GINConv
from torch_geometric.nn import HeteroConv
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
    GIN cho graph projected 1 node-type (product) với 1 edge_index.
    Có thể bật use_softplus_output để đảm bảo output dương.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        num_layers: int = 3,
        use_softplus_output: bool = False,
    ):
        super().__init__()
        self.use_softplus_output = use_softplus_output

        self.convs = nn.ModuleList()
        # layer 1
        self.convs.append(
            GINConv(MLP(in_channels, hidden_channels))
        )
        # các layer tiếp theo
        for _ in range(num_layers - 1):
            self.convs.append(
                GINConv(MLP(hidden_channels, hidden_channels))
            )

        self.out_lin = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index):
        """
        x: [N, F]
        edge_index: [2, E]
        return: [N]
        """
        h = x
        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)

        out = self.out_lin(h).squeeze(-1)  # [N]
        if self.use_softplus_output:
            # Softplus: z' = log(1 + exp(z))
            out = F.softplus(out)
        return out


# ============================================================
# 3. Homogeneous 5-type GIN Regressor
#    (treat all node types in one big graph, with type embedding)
# ============================================================

class HomogeneousFiveTypeGINRegressor(nn.Module):
    """
    GIN đồng nhất, gộp 5 node-type vào một graph lớn.
    Dùng embedding node-type để cho model phân biệt type.
    Output chỉ trên các node 'product'.

    num_nodes_dict: {node_type: N_type}
    node_type_order: list theo thứ tự concat để build global node index.
    """

    def __init__(
        self,
        in_channels: int,
        num_nodes_dict: dict,
        node_type_order: list,
        hidden_channels: int = 128,
        num_layers: int = 3,
        node_type_emb_dim: int = 8,
        use_softplus_output: bool = False,
    ):
        super().__init__()
        self.num_nodes_dict = num_nodes_dict
        self.node_type_order = node_type_order
        self.use_softplus_output = use_softplus_output

        # Tính tổng số node và offset cho từng type
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

        # mapping: global_index -> node_type_id (0..num_types-1)
        node_type_id = torch.empty(self.total_num_nodes, dtype=torch.long)
        cur = 0
        for i, nt in enumerate(node_type_order):
            n = num_nodes_dict[nt]
            node_type_id[cur:cur + n] = i
            cur += n
        self.register_buffer("node_type_id", node_type_id, persistent=False)

        self.num_types = len(node_type_order)
        self.type_emb = nn.Embedding(self.num_types, node_type_emb_dim)

        # GIN layers
        self.convs = nn.ModuleList()
        self.convs.append(
            GINConv(MLP(in_channels + node_type_emb_dim, hidden_channels))
        )
        for _ in range(num_layers - 1):
            self.convs.append(
                GINConv(MLP(hidden_channels, hidden_channels))
            )

        self.out_lin = nn.Linear(hidden_channels, 1)

    def _concat_x_dict(self, x_dict: dict):
        """
        x_dict: {node_type: [N_type, F]}
        Trả về:
          x_all: [N_total, F]
        """
        xs = []
        for nt in self.node_type_order:
            xs.append(x_dict[nt])
        x_all = torch.cat(xs, dim=0)
        return x_all

    def forward(self, x_dict: dict, edge_index):
        """
        x_dict: {node_type: [N_type, F]}
        edge_index: [2, E] trên global node index
        return: [N_product] (theo thứ tự product trong node_type_order)
        """
        device = edge_index.device
        # concat features
        x_all = self._concat_x_dict(x_dict).to(device)  # [N_total, F]
        node_type_id = self.node_type_id.to(device)     # [N_total]
        type_emb = self.type_emb(node_type_id)          # [N_total, D_type]

        h = torch.cat([x_all, type_emb], dim=-1)        # [N_total, F + D_type]

        for conv in self.convs:
            h = conv(h, edge_index)
            h = F.relu(h)

        out_all = self.out_lin(h).squeeze(-1)           # [N_total]
        if self.use_softplus_output:
            out_all = F.softplus(out_all)

        # Lấy lại chỉ phần product
        # product được assume có tên 'product' trong node_type_order
        idx_prod_type = self.node_type_order.index("product")
        offset_prod = self.node_type_offsets[idx_prod_type].item()
        n_prod = self.num_nodes_dict["product"]
        out_prod = out_all[offset_prod:offset_prod + n_prod]   # [N_product]
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
    Heterogeneous GIN trên graph 5 node-type.
    x_dict: {node_type: [N_type, F]}
    edge_index_dict: {(src_type, rel, dst_type): [2, E]}
    Output: chỉ trên node_type 'product'.
    """

    def __init__(
        self,
        in_channels_dict: dict,
        hidden_channels: int = 128,
        num_layers: int = 2,
        use_softplus_output: bool = False,
    ):
        super().__init__()
        self.node_types = [nt for nt in in_channels_dict.keys() if nt != "edge_types"]
        self.use_softplus_output = use_softplus_output

        # Đảm bảo có key "edge_types" trong in_channels_dict nếu dùng HeterogeneousGINLayer
        if "edge_types" not in in_channels_dict:
            raise ValueError("in_channels_dict must contain key 'edge_types' listing edge types.")

        # Mạng cho từng node_type
        self.node_in_proj = nn.ModuleDict()
        for nt in self.node_types:
            self.node_in_proj[nt] = nn.Linear(in_channels_dict[nt], hidden_channels)

        # Các layer Hetero GIN
        self.layers = nn.ModuleList()
        in_chs = {"edge_types": in_channels_dict["edge_types"], **{nt: hidden_channels for nt in self.node_types}}
        for _ in range(num_layers):
            self.layers.append(HeterogeneousGINLayer(in_chs, hidden_channels))

        self.out_lin = nn.Linear(hidden_channels, 1)

    def forward(self, x_dict, edge_index_dict):
        # project input features
        h_dict = {}
        for nt in self.node_types:
            h_dict[nt] = F.relu(self.node_in_proj[nt](x_dict[nt]))

        # Hetero layers
        for layer in self.layers:
            h_dict = layer(h_dict, edge_index_dict)
            for nt in h_dict.keys():
                h_dict[nt] = F.relu(h_dict[nt])

        out_prod = self.out_lin(h_dict["product"]).squeeze(-1)
        if self.use_softplus_output:
            out_prod = F.softplus(out_prod)
        return out_prod


# ============================================================
# Helper: chuyển h_dict sang định dạng cần cho HeterogeneousGINLayer (nếu muốn tách)
# ============================================================

def x_dict_to_for_layer(h_dict):
    """
    HeteroConv forward(x_dict, edge_index_dict) yêu cầu x_dict: {node_type: [N_type, F]}.
    Hàm này hiện chỉ là identity, tách riêng để sau dễ sửa nếu cần.
    """
    return h_dict
import torch
from torch import nn
from torch_geometric.nn import GINConv, HeteroConv
import torch.nn.functional as F

# Nếu file encoder nằm cùng package:
from models_gnn_encoder import (
    ProjectedGINEncoder,
    HomogeneousFiveTypeGINEncoder,
    HeterogeneousGINEncoder,
)


# ============================================================
# 1. GIN block dùng chung (giữ lại cho các chỗ còn dùng)
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
#    Dùng ProjectedGINEncoder làm encoder
# ============================================================


class ProjectedGINRegressor(nn.Module):
    """
    GIN cho projected product graph.
    Forward trả logits trên scale training (z).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 128,
        num_layers: int = 3,
        is_softplus: bool = False,  # chỉ để log/tag
        is_log1p: bool = False,
    ):
        super().__init__()
        if is_softplus and is_log1p:
            raise ValueError("Only one of is_softplus / is_log1p can be True.")

        self.is_softplus = is_softplus
        self.is_log1p = is_log1p

        # Encoder GIN (chia sẻ với phần export embeddings)
        self.encoder = ProjectedGINEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
        )

        # Head regression trên embedding
        self.out_lin = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index):
        """
        x: [N, F]
        edge_index: [2, E]
        return: [N] logits z trên scale training
        """
        h = self.encoder(x, edge_index)      # [N, hidden]
        head = self.out_lin(h).squeeze(-1)   # [N]
        return head


# ============================================================
# 3. Homogeneous 5-type GIN Regressor
#    Wrap HomogeneousFiveTypeGINEncoder + head
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

        self.is_softplus = is_softplus
        self.is_log1p = is_log1p

        # Encoder đồng nhất 5 node-type (dùng lại trong export embeddings)
        self.encoder = HomogeneousFiveTypeGINEncoder(
            in_channels=in_channels,
            num_nodes_dict=num_nodes_dict,
            node_type_order=node_type_order,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            node_type_emb_dim=node_type_emb_dim,
        )

        # Head regression từ embedding product -> scalar
        self.out_lin = nn.Linear(hidden_channels, 1)

    def forward(self, x_dict: dict, edge_index):
        """
        x_dict: {node_type: [N_type, F_in]}
        edge_index: [2, E_total] với index global
        return: [N_product] logits z trên scale training
        """
        # encoder trả về embedding cho node_type 'product': [N_product, hidden]
        h_prod = self.encoder(x_dict, edge_index)   # [N_product, hidden]
        head = self.out_lin(h_prod).squeeze(-1)     # [N_product]
        return head


# ============================================================
# 4. Heterogeneous GIN Regressor (5-type)
#    Wrap HeterogeneousGINEncoder + head
# ============================================================


class HeterogeneousGINLayer(nn.Module):
    """
    Giữ lại nếu bạn còn dùng riêng layer này ở chỗ khác.
    Ở đây Regressor sẽ dùng HeterogeneousGINEncoder nên class này
    có thể không cần thiết nữa, nhưng mình giữ để tránh vỡ import.
    """

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

        self.is_softplus = is_softplus
        self.is_log1p = is_log1p

        # Encoder Heterogeneous GIN (5 node-type), reuse trong export embeddings
        self.encoder = HeterogeneousGINEncoder(
            in_channels_dict=in_channels_dict,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
        )

        # Head regression cho node_type 'product'
        self.out_lin = nn.Linear(hidden_channels, 1)

    def forward(self, x_dict, edge_index_dict):
        """
        x_dict: {node_type: [N_type, F_in]}
        edge_index_dict: {(src_type, rel, dst_type): edge_index}
        return: [N_product] logits z trên scale training
        """
        # encoder trả embedding cho product nodes
        h_prod = self.encoder(x_dict, edge_index_dict)    # [N_product, hidden]
        head = self.out_lin(h_prod).squeeze(-1)           # [N_product]
        return head
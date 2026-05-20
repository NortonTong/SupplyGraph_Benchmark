import torch
from torch import nn
from torch_geometric.nn import GINConv, HeteroConv
import torch.nn.functional as F
from models_gnn_encoder import (
    ProjectedGINEncoder,
    HomogeneousFiveTypeGINEncoder,
    HeterogeneousGINEncoder,
)

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

class ProjectedGINRegressor(nn.Module):

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
        self.encoder = ProjectedGINEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
        )

        self.out_lin = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index):
        """
        x: [N, F]
        edge_index: [2, E]
        return: [N] logits z trên scale training
        """
        h = self.encoder(x, edge_index)    
        head = self.out_lin(h).squeeze(-1)   
        return head


class HomogeneousFiveTypeGINRegressor(nn.Module):
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

        self.is_softplus = is_softplus
        self.is_log1p = is_log1p

        self.encoder = HomogeneousFiveTypeGINEncoder(
            in_channels=in_channels,
            num_nodes_dict=num_nodes_dict,
            node_type_order=node_type_order,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            node_type_emb_dim=node_type_emb_dim,
        )

        self.out_lin = nn.Linear(hidden_channels, 1)

    def forward(self, x_dict: dict, edge_index):
        h_prod = self.encoder(x_dict, edge_index)   
        head = self.out_lin(h_prod).squeeze(-1)    
        return head

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
        return self.convs(x_dict, edge_index_dict)


class HeterogeneousGINRegressor(nn.Module):
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

        if "edge_types" not in in_channels_dict:
            raise ValueError(
                "in_channels_dict must contain key 'edge_types' listing edge types."
            )

        self.is_softplus = is_softplus
        self.is_log1p = is_log1p

        self.encoder = HeterogeneousGINEncoder(
            in_channels_dict=in_channels_dict,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
        )
        self.out_lin = nn.Linear(hidden_channels, 1)

    def forward(self, x_dict, edge_index_dict):
        h_prod = self.encoder(x_dict, edge_index_dict)    
        head = self.out_lin(h_prod).squeeze(-1)          
        return head
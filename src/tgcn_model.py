# models_tgcn.py
import torch
from torch import nn
from torch_geometric.nn import GCNConv


class TGCN(nn.Module):
    """
    X: [B, L, N, F]
    edge_index: [2, E]
    Output: [B, pre_len, N]
    """

    def __init__(
        self,
        num_nodes: int,
        in_channels: int,
        hidden_channels: int = 64,
        gcn_layers: int = 1,
        pre_len: int = 1,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.pre_len = pre_len

        # GCN stack (áp dụng cho từng bước thời gian)
        convs = []
        for i in range(gcn_layers):
            in_ch = in_channels if i == 0 else hidden_channels
            convs.append(GCNConv(in_ch, hidden_channels))
        self.convs = nn.ModuleList(convs)

        # GRU trên chuỗi thời gian, feature size = N * hidden_channels
        self.gru = nn.GRU(
            input_size=num_nodes * hidden_channels,
            hidden_size=num_nodes * hidden_channels,
            num_layers=1,
            batch_first=True,
        )

        # Linear ra pre_len * N
        self.fc = nn.Linear(num_nodes * hidden_channels, pre_len * num_nodes)

    def forward(self, x, edge_index):
        """
        x: [B, L, N, F]
        edge_index: [2, E]
        """
        B, L, N, F = x.shape
        assert N == self.num_nodes

        gcn_outputs = []
        for t in range(L):
            x_t = x[:, t]           # [B, N, F]
            x_t = x_t.reshape(B * N, F)
            for conv in self.convs:
                x_t = conv(x_t, edge_index)  # [B*N, H]
                x_t = torch.relu(x_t)
            x_t = x_t.reshape(B, N, self.hidden_channels)
            gcn_outputs.append(x_t)

        h_seq = torch.stack(gcn_outputs, dim=1)  # [B, L, N, H]
        h_seq = h_seq.reshape(B, L, N * self.hidden_channels)

        gru_out, _ = self.gru(h_seq)            # [B, L, N*H]
        last = gru_out[:, -1, :]                # [B, N*H]

        out = self.fc(last)                     # [B, pre_len * N]
        out = out.view(B, self.pre_len, N)      # [B, pre_len, N]
        return out
    
# models_tgcn.py (phiên bản multi-view)
import torch
from torch import nn
from torch_geometric.nn import GCNConv


class MultiViewTGCN(nn.Module):
    """
    X: [B, L, N, F]
    edge_index_dict: dict[str, edge_index], 4 view
    Output: [B, pre_len, N]
    """

    def __init__(
        self,
        num_nodes: int,
        in_channels: int,
        hidden_channels: int = 32,
        gcn_layers: int = 1,
        pre_len: int = 1,
        views=("plant", "product_group", "sub_group", "storage"),
        fusion="concat",  # "concat" hoặc "sum"
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.pre_len = pre_len
        self.views = views
        self.fusion = fusion

        # GCN stack per view
        self.view_convs = nn.ModuleDict()
        for v in views:
            convs = []
            for i in range(gcn_layers):
                in_ch = in_channels if i == 0 else hidden_channels
                convs.append(GCNConv(in_ch, hidden_channels))
            self.view_convs[v] = nn.ModuleList(convs)

        # fusion dim
        if fusion == "concat":
            fusion_dim = hidden_channels * len(views)
        else:  # "sum"
            fusion_dim = hidden_channels

        # GRU: feature size = N * fusion_dim
        self.gru = nn.GRU(
            input_size=num_nodes * fusion_dim,
            hidden_size=num_nodes * fusion_dim,
            num_layers=1,
            batch_first=True,
        )

        # Linear ra pre_len * N
        self.fc = nn.Linear(num_nodes * fusion_dim, pre_len * num_nodes)

    def forward(self, x, edge_index_dict):
        """
        x: [B, L, N, F]
        edge_index_dict: {view_name: edge_index}
        """
        B, L, N, F = x.shape
        assert N == self.num_nodes

        gcn_outputs = []
        for t in range(L):
            x_t = x[:, t]  # [B, N, F]
            x_t = x_t.reshape(B * N, F)

            view_feats = []
            for v in self.views:
                h_v = x_t
                for conv in self.view_convs[v]:
                    h_v = conv(h_v, edge_index_dict[v])  # [B*N, H]
                    h_v = torch.relu(h_v)
                h_v = h_v.reshape(B, N, self.hidden_channels)
                view_feats.append(h_v)                 # [B, N, H]

            if self.fusion == "concat":
                h_t = torch.cat(view_feats, dim=-1)     # [B, N, H*V]
            else:  # "sum"
                h_t = torch.stack(view_feats, dim=0).sum(dim=0)  # [B, N, H]

            gcn_outputs.append(h_t)

        h_seq = torch.stack(gcn_outputs, dim=1)         # [B, L, N, fusion_dim]
        fusion_dim = h_seq.shape[-1]
        h_seq = h_seq.reshape(B, L, N * fusion_dim)

        gru_out, _ = self.gru(h_seq)                   # [B, L, N*fusion_dim]
        last = gru_out[:, -1, :]                       # [B, N*fusion_dim]

        out = self.fc(last)                            # [B, pre_len * N]
        out = out.view(B, self.pre_len, N)             # [B, pre_len, N]
        return out
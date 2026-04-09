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
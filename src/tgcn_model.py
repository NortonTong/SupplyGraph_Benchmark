# tgcn_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class TGCNCell(nn.Module):
    """
    Một bước T-GCN như paper:
      - GCN để lấy spatial features.
      - GRU-like gates (r, u, c) để cập nhật hidden state.
    """
    def __init__(self, in_channels, gcn_hidden, gru_hidden):
        super().__init__()
        self.gcn = GCNConv(in_channels, gcn_hidden)

        # GRU gates: reset, update, candidate
        self.W_r = nn.Linear(gcn_hidden + gru_hidden, gru_hidden)
        self.W_u = nn.Linear(gcn_hidden + gru_hidden, gru_hidden)
        self.W_c = nn.Linear(gcn_hidden + gru_hidden, gru_hidden)

    def forward(self, x_t, h_prev, edge_index):
        """
        x_t:     [N, F_in]
        h_prev:  [N, H]  (hidden state trước đó)
        edge_index: [2, E]
        return:
          h_t: [N, H]
        """
        # 1) GCN cho spatial thông tin
        z_t = self.gcn(x_t, edge_index)      # [N, gcn_hidden]
        z_t = F.relu(z_t)

        # 2) GRU-like update
        # concat input và hidden trước đó
        xh = torch.cat([z_t, h_prev], dim=-1)  # [N, gcn_hidden + H]

        r_t = torch.sigmoid(self.W_r(xh))      # reset gate
        u_t = torch.sigmoid(self.W_u(xh))      # update gate

        xh_candidate = torch.cat([z_t, r_t * h_prev], dim=-1)
        c_t = torch.tanh(self.W_c(xh_candidate))  # candidate state

        h_t = u_t * h_prev + (1.0 - u_t) * c_t
        return h_t


class TGCN(nn.Module):
    """
    T-GCN đơn giản:
      - Input: chuỗi X_seq [T, N, F] (T bước lịch sử), một edge_index cố định.
      - Unroll TGCNCell theo thời gian để thu được h_T cho mỗi node.
      - Dùng h_T qua một Linear => dự đoán y_hat [N] cho horizon tương ứng.
    """
    def __init__(self, in_channels, gcn_hidden=64, gru_hidden=64, horizon=1):
        super().__init__()
        self.horizon = horizon
        self.cell = TGCNCell(in_channels, gcn_hidden, gru_hidden)
        self.out_layer = nn.Linear(gru_hidden, 1)

    def forward(self, x_seq, edge_index):
        """
        x_seq: [T, N, F]  (chuỗi input lịch sử)
        edge_index: [2, E]
        Output: [N] (dự đoán cho horizon tương ứng)
        """
        T, N, Fdim = x_seq.shape
        device = x_seq.device

        # init hidden state = 0 như repo gốc
        H = self.cell.W_r.out_features
        h = torch.zeros(N, H, device=device)

        # unroll theo thời gian
        for t in range(T):
            x_t = x_seq[t]         # [N, F]
            h = self.cell(x_t, h, edge_index)  # [N, H]

        # sau khi đi hết chuỗi, dùng h_T để dự đoán
        y_hat = self.out_layer(h).squeeze(-1)  # [N]
        return y_hat
import torch
import torch.nn as nn


class MHGAT(nn.Module):
    def __init__(self, dim=128, dropout=0.1):
        super().__init__()
        self.Wv = nn.Linear(dim, dim, bias=False)
        self.do = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(dim)

    def forward(self, U, V):
        msg = self.Wv(V)
        U_upd = self.ln(U + self.do(msg))
        return U_upd
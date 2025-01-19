import torch
from torch import nn as nn


class SideInformationEmbedding(nn.Module):
    """
    Side information embedding step input of the STDM Residual Blocks (time and conditional mask).
    emb_time: (1, L, T) -> repeat (N, L, T)
    m_cond x emb_time: (N, L, K) + (N, L, T) -> concat (N, L, K + T)

    N: batch size
    L: periods
    K: features/characteristics (129 in the paper)
    T: time dimension (64 in the paper)
    """

    def __init__(self):
        super().__init__()

    def forward(self, time, m_cond):
        n_repeats = m_cond.size(0)
        time = time.repeat(n_repeats, 1, 1)
        x = torch.cat([time, m_cond], dim=2)
        return x

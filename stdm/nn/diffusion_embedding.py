from torch import nn as nn


class DiffusionEmbedding(nn.Module):
    """
    Diffusion step input of the STDM Residual Blocks.
    t_diff: (N, 1, 1) -> Linear SiLU (N, 1, out/2) -> Linear SiLU (N, 1, out) -> repeat (N, L, out)
    Note: According to the paper there are two linear layers with SiLU activation. However, there was no mention of the
    intermediate dimension of the first linear layer. Therefore, we assume that the intermediate dimension is out/2.

    N: batch size
    L: periods

    Parameters
    ----------
    out, output dimension: (128 in the paper)

    """

    def __init__(self, output_dim: int):
        super().__init__()
        self.output_dim = output_dim
        self.fc1 = nn.Linear(1, output_dim // 2)
        self.fc2 = nn.Linear(output_dim // 2, output_dim)

    def forward(self, t, n_repeat):
        t = nn.SiLU()(self.fc1(t))
        t = nn.SiLU()(self.fc2(t))
        t = t.repeat(1, n_repeat, 1)
        return t

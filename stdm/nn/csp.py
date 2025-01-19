from torch import nn as nn


class CharacteristicSpecificEncoder(nn.Module):
    """
    Characteristic-specific Projection (CSP) Encoder, correspond to the feature input of the STDM model.
    (N, L, K) -> 1x1 groupconv LeakyReLU (N, L, K * E) -> 1x1 conv (N, L, dm)

    N: batch size
    L: periods

    Parameters
    ----------
    K, n_features: features/characteristics (129 in the paper)
    E, n_channels: number of channels (8 in the paper)
    dm, out_channels: output dimension (512 in the paper)
    """

    def __init__(self, n_features: int, n_channels: int, out_channels: int):
        super().__init__()
        self.group_conv = nn.Conv1d(
            in_channels=n_features,
            out_channels=n_channels * n_features,
            kernel_size=1,
            groups=n_features,
        )

        self.conv_1_1 = nn.Conv1d(
            in_channels=n_channels * n_features,
            out_channels=out_channels,
            kernel_size=1,
        )

    def forward(self, x_t):
        x_t = x_t.permute(0, 2, 1)
        x_t = self.group_conv(x_t)
        x_t = nn.LeakyReLU()(x_t)
        x_t = self.conv_1_1(x_t)
        x_t = x_t.permute(0, 2, 1)
        return x_t


class CharacteristicSpecificDecoder(nn.Module):
    """
    CSP Decoder, outputting of the STDM model (i.e., the predicted noise).
    (N, L, dm) -> 1x1 conv (N, L, K * E) -> LeakyReLU 1x1 groupconv (N, L, K)

    N: batch size
    L: periods

    Parameters
    ----------
    K, n_features: features/characteristics (129 in the paper)
    E, n_channels: number of channels (8 in the paper)
    dm, in_channels: input dimension (512 in the paper)
    """

    def __init__(self, in_channels: int, n_features: int, n_channels: int):
        super().__init__()
        self.conv_1_1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=n_channels * n_features,
            kernel_size=1,
        )

        self.group_conv = nn.Conv1d(
            in_channels=n_channels * n_features,
            out_channels=n_features,
            kernel_size=1,
            groups=n_features,
        )

    def forward(self, x_t):
        x_t = x_t.permute(0, 2, 1)
        x_t = self.conv_1_1(x_t)
        x_t = nn.LeakyReLU()(x_t)
        x_t = self.group_conv(x_t)
        x_t = x_t.permute(0, 2, 1)
        return x_t

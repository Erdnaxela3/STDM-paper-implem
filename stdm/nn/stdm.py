from torch import nn as nn

from stdm.nn.csp import CharacteristicSpecificEncoder, CharacteristicSpecificDecoder
from stdm.nn.diffusion_embedding import DiffusionEmbedding
from stdm.nn.sideinfo_embedding import SideInformationEmbedding


class STDMResidualBlock(nn.Module):
    """
    STDM Residual Block, the main building block of the STDM model. Duplicable.
    x_t: output of a CSP encoder (N, L, dm)
    t: output of a diffusion embedding (N, L, out)
    side_info: output of a side information embedding (N, L, K + T)

    Diffusion step and side information are converted to convoluted features entering the block.

    Note: Two linear layers that were not mentioned in the paper were added before the tanh and sigmoid activations.
    As the output of the previous step is (N, L, 2 * dm) and the input of the activation functions is (N, L, dm).

    Parameters
    ----------
    dm, csp_input_dim: input dimension of the CSP encoder (512 in the paper)
    out, diff_output_dim: output dimension of the diffusion embedding (128 in the paper)
    K, side_info_dim: features/characteristics (129 in the paper)
    T: time dimension (64 in the paper)
    """

    def __init__(self, csp_input_dim: int, diff_output_dim: int, side_info_dim: int):
        super().__init__()
        self.diff_conv = nn.Conv1d(
            in_channels=diff_output_dim,
            out_channels=csp_input_dim,
            kernel_size=1,
        )

        self.side_info_conv = nn.Conv1d(
            in_channels=side_info_dim,
            out_channels=2 * csp_input_dim,
            kernel_size=1,
        )

        self.attention = nn.TransformerEncoderLayer(
            d_model=csp_input_dim,
            nhead=8,
        )

        # FIXME not actual gcn, just a linear layer for the time being
        self.gcn = nn.Linear(csp_input_dim, 2 * csp_input_dim)

        self.conv = nn.Conv1d(
            in_channels=2 * csp_input_dim,
            out_channels=2 * csp_input_dim,
            kernel_size=1,
        )

        self.lin_tanh = nn.Linear(2 * csp_input_dim, csp_input_dim)
        self.lin_sigmoid = nn.Linear(2 * csp_input_dim, csp_input_dim)

        self.conv_reinput = nn.Conv1d(
            in_channels=csp_input_dim,
            out_channels=csp_input_dim,
            kernel_size=1,
        )

        self.conv_output = nn.Conv1d(
            in_channels=csp_input_dim,
            out_channels=csp_input_dim,
            kernel_size=1,
        )

    def forward(self, x, t, side_info):
        t = t.permute(0, 2, 1)
        t = self.diff_conv(t)
        t = t.permute(0, 2, 1)

        side_info = side_info.permute(0, 2, 1)
        side_info = self.side_info_conv(side_info)
        side_info = side_info.permute(0, 2, 1)

        x_from_start = x

        x = x + t
        x = self.attention(x)
        x = self.gcn(x)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        x = x + side_info

        h_tanh = self.lin_tanh(x)
        h_tanh = nn.Tanh()(h_tanh)

        h_sigmoid = self.lin_sigmoid(x)
        h_sigmoid = nn.Sigmoid()(h_sigmoid)
        x = h_tanh * h_sigmoid

        x = x.permute(0, 2, 1)
        reinput = self.conv_reinput(x)
        reinput = reinput.permute(0, 2, 1)
        reinput += x_from_start

        out = self.conv_output(x)
        out = out.permute(0, 2, 1)

        return reinput, out


class STDM(nn.Module):
    """
    Spatio-Temporal Diffusion Model (STDM) model.

    Parameters
    ----------
    K, n_features: features/characteristics (129 in the paper)
    E, n_channels: number of channels (8 in the paper)
    dm, csp_output_dim: output dimension of the CSP encoder (512 in the paper)
    out, diffusion_output_dim: output dimension of the diffusion embedding (128 in the paper)
    n_residual_blocks: number of residual blocks (4 in the paper)
    time_embed_dim: time embedding dimension (64 in the paper)
    """

    def __init__(
        self,
        n_features: int,
        n_channels: int,
        csp_output_dim: int,
        diffusion_output_dim: int = 128,
        n_residual_blocks: int = 4,
        time_embed_dim: int = 64,
    ):
        super().__init__()
        self.csp = CharacteristicSpecificEncoder(n_features, n_channels, csp_output_dim)
        self.diff = DiffusionEmbedding(diffusion_output_dim)
        self.sie = SideInformationEmbedding()

        self.res_blocks = nn.ModuleList(
            [
                STDMResidualBlock(
                    csp_input_dim=csp_output_dim,
                    diff_output_dim=diffusion_output_dim,
                    side_info_dim=n_features + time_embed_dim,
                )
                for _ in range(n_residual_blocks)
            ]
        )

        self.csd = CharacteristicSpecificDecoder(
            in_channels=csp_output_dim, n_features=n_features, n_channels=n_channels
        )

    def forward(self, x, t, emb_time, m_cond):
        x = self.csp(x)
        t = self.diff(t, n_repeat=x.size(1))
        side_info = self.sie(emb_time, m_cond)

        # compute first output of the residual block
        # then reuse the first output as input for the next residual
        # + sum up of the second output
        reinput, sum_out = self.res_blocks[0](x, t, side_info)
        for res_block in self.res_blocks[1:]:
            reinput, out = res_block(reinput, t, side_info)
            sum_out += out

        x = self.csd(x)
        return x

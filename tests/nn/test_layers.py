from stdm.nn import (
    STDM,
    STDMResidualBlock,
    CharacteristicSpecificEncoder,
    CharacteristicSpecificDecoder,
    DiffusionEmbedding,
    SideInformationEmbedding,
)
import torch

N, L, K, dc = 100, 60, 129, 8
dm = 512


def test_characteristic_specific_encoder():
    csp = CharacteristicSpecificEncoder(n_features=K, n_channels=dc, out_channels=dm)
    csp_in = torch.randn(N, L, K)
    out = csp(csp_in)
    assert out.shape == (N, L, dm)

    csp_in_2 = torch.randn(N * 2 + 3, L + 69, K)
    out_2 = csp(csp_in_2)
    assert out_2.shape == (N * 2 + 3, L + 69, dm)


def test_characteristic_specific_decoder():
    csp = CharacteristicSpecificDecoder(in_channels=dm, n_features=K, n_channels=dc)
    csp_in = torch.randn(N, L, dm)
    out = csp(csp_in)
    assert out.shape == (N, L, K)


def test_diffusion_embedding():
    diff = DiffusionEmbedding(output_dim=128)
    t = torch.randn(N, 1, 1)
    out = diff(t, n_repeat=L)
    assert out.shape == (N, L, 128)


def test_side_information_embedding():
    sie = SideInformationEmbedding()
    emb_time = torch.randn(1, L, 64)
    m_cond = torch.randn(N, L, K)
    out = sie(emb_time, m_cond)
    assert out.shape == (N, L, 64 + K)


def test_stdm_residual_block():
    stdm_res_block = STDMResidualBlock(csp_input_dim=dm, diff_output_dim=128, side_info_dim=K + 64)
    x = torch.randn(N, L, dm)
    t = torch.randn(N, L, 128)
    side_info = torch.randn(N, L, K + 64)
    reinput, out = stdm_res_block(x, t, side_info)
    assert reinput.shape == (N, L, dm)
    assert out.shape == (N, L, dm)


def test_stdm():
    stdm_model = STDM(n_features=K, n_channels=dc, csp_output_dim=dm, diffusion_output_dim=128)
    x = torch.randn(N, L, K)
    t = torch.randn(N, 1, 1)
    emb_time = torch.randn(1, L, 64)
    m_cond = torch.randn(N, L, K)
    predicted_noise = stdm_model(x, t, emb_time, m_cond)
    assert predicted_noise.shape == (N, L, K)

    new_n = N * 2 + 3
    new_l = L + 69

    x = torch.randn(new_n, new_l, K)
    t = torch.randn(new_n, 1, 1)
    emb_time = torch.randn(1, new_l, 64)
    m_cond = torch.randn(new_n, new_l, K)
    predicted_noise = stdm_model(x, t, emb_time, m_cond)
    assert predicted_noise.shape == (new_n, new_l, K)

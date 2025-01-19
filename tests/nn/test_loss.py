import torch

from stdm.nn import STDMDiffusionLoss


def test_stdmdiffusionloss():
    stdm_loss = STDMDiffusionLoss()
    mask = torch.Tensor([[0, 1, 0], [1, 0, 1]])
    noise = torch.Tensor([[1, 2, 3], [4, 5, 6]])
    predicted_noise = torch.Tensor([[-2, 2, 4], [4, 4, 19]])

    # masked_noise = ([1, 0, 3], [0, 5, 0])
    # masked_predicted_noise = ([-2, 0, 4], [0, 4, 0])
    # diff = ([3, 0, 1], [0, 1, 0])
    # squared_diff = ([9, 0, 1], [0, 1, 0])
    # n_data = number of non-masked data points = 3

    expected = (9 + 1 + 1) / 3
    loss = stdm_loss(mask, noise, predicted_noise)

    assert torch.isclose(loss, torch.tensor(expected))

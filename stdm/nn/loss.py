import torch
from torch import nn


class STDMDiffusionLoss(nn.Module):
    """
    Spatio-Temporal Diffusion Model (STDM) loss.
    Compute the diffusion loss given the mask, noise and predicted noise.
    L2 distance, conditioned on the mask.

    Parameters
    ----------
    mask : torch.Tensor
        The mask tensor. 1 if the value is observed, 0 otherwise.
    noise : torch.Tensor
        The noise tensor.
    predicted_noise : torch.Tensor
        The predicted noise tensor.

    Returns
    -------
    torch.Tensor
        The diffusion loss.
    """

    def __init__(self):
        super().__init__()

    def forward(self, mask: torch.Tensor, noise: torch.Tensor, predicted_noise: torch.Tensor) -> torch.Tensor:
        return (((1 - mask) * (noise - predicted_noise)) ** 2).sum() / mask.sum()

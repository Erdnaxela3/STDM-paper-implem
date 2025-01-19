import os

import pandas as pd
import pytest
import torch

from stdm.model.impute import compute_x0_t_hat, ddim_step, stdm_impute
from stdm.model.training import compute_alpha_t
from stdm.nn import STDM
from stdm.utils.data import OSAPDataset, OSAPDataLoader

from tests import filepath, file_exists, device


def test_compute_x_0_t_hat():
    x_t = torch.tensor([[[1.42]]])
    t = torch.tensor([[[2]]])
    predicted_noise = torch.tensor([[[0.22]]])

    x0_t_hat = compute_x0_t_hat(x_t, t, predicted_noise)

    assert x0_t_hat.shape == x_t.shape
    assert not torch.allclose(x0_t_hat, x_t)


def test_compute_x_t_prev():
    x_t = torch.tensor([[[1.42]]])
    t = torch.tensor([[[2]]])
    predicted_noise = torch.tensor([[[0.22]]])

    x_t_prev = ddim_step(x_t, t, predicted_noise, 1)

    assert x_t_prev.shape == x_t.shape
    assert not torch.allclose(x_t_prev, x_t)


@pytest.mark.skipif(not file_exists, reason="Data file not found")
def test_impute():
    df = pd.read_csv(filepath, nrows=100)
    dataset = OSAPDataset(df)

    model = STDM(
        n_features=len(df.columns) - 2, n_channels=8, diffusion_output_dim=128, csp_output_dim=512, time_embed_dim=10
    )

    dataloader = OSAPDataLoader(dataset, batch_size=1, shuffle_masks=False)
    model.to(device)

    x_0, emb_time, m_miss, m = next(iter(dataloader))
    x_0, emb_time, m_miss, m = x_0.to(device), emb_time.to(device), m_miss.to(device), m.to(device)

    noise = torch.randn_like(x_0)
    t = torch.randint(0, 50, (x_0.size(0), 1, 1), device=device).float()
    alpha_t = compute_alpha_t(t)
    x_t_mask = torch.sqrt(alpha_t) * x_0 + torch.sqrt(1 - alpha_t) * noise
    x_t = m * x_0 + (1 - m) * x_t_mask

    imputed_x_0 = stdm_impute(model=model, x_t=x_t, emb_time=emb_time, m=m, t_diff=50, n_steps=5)

    assert imputed_x_0.shape == x_0.shape

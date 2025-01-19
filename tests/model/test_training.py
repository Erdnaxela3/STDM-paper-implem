import pandas as pd
import pytest
import torch

from stdm.model.training import train_stdm, compute_alpha_t, compute_beta_t
from stdm.nn import STDM
from stdm.utils.data import OSAPDataset, OSAPDataLoader
from tests import filepath, file_exists, device


def test_compute_beta_t():
    N = 50
    t = torch.randint(0, 50, (N, 1, 1)).float()
    beta_t = compute_beta_t(t)
    assert beta_t.shape == t.shape


def test_compute_alpha_t_0():
    t = torch.tensor([[[0]]]).float()
    alpha_t = compute_alpha_t(t)
    assert alpha_t.shape == t.shape
    assert alpha_t.item() == 1.0


def test_compute_alpha_t():
    N = 50
    t = torch.randint(0, 50, (N, 1, 1)).float()
    alpha_t = compute_alpha_t(t)
    assert alpha_t.shape == t.shape


@pytest.mark.skipif(not file_exists, reason="Data file not found")
def test_stdm_training():
    model = STDM(n_features=209, n_channels=8, diffusion_output_dim=128, csp_output_dim=512, time_embed_dim=10)

    df = pd.read_csv(filepath, nrows=100)
    dataset = OSAPDataset(df)

    train_dataloader = OSAPDataLoader(dataset, batch_size=1)
    val_dataloader = OSAPDataLoader(dataset, batch_size=1)

    train_stdm(model, train_dataloader, val_dataloader, 2, 0.01, device, 2, save_model=False)

    assert True

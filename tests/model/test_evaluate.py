import pandas as pd

from stdm.nn import STDM
from stdm.utils.data import OSAPDataset, OSAPDataLoader

import pytest
from stdm.model.evaluate import evaluate_stdm, evaluate_mean_imputer

from tests import filepath, file_exists, device


@pytest.mark.skipif(not file_exists, reason="Data file not found")
@pytest.mark.parametrize("batch_size", [1, 2, 3, 4, 5, 6, 7])
def test_stdm_evaluate_run(batch_size):
    df = pd.read_csv(filepath, nrows=100)
    dataset = OSAPDataset(df)

    model = STDM(
        n_features=len(df.columns) - 2, n_channels=8, diffusion_output_dim=128, csp_output_dim=512, time_embed_dim=10
    )

    dataloader = OSAPDataLoader(dataset, batch_size=batch_size, shuffle_masks=False)

    model.to(device)

    losses = evaluate_stdm(model, dataloader)

    assert losses["mse"] >= 0.0
    assert losses["mae"] >= 0.0


@pytest.mark.skipif(not file_exists, reason="Data file not found")
@pytest.mark.parametrize("batch_size", [1, 2, 3, 4, 5, 6, 7])
def test_evaluate_mean_impute(batch_size):
    df = pd.read_csv(filepath, nrows=100)
    dataset = OSAPDataset(df)
    dataloader = OSAPDataLoader(dataset, batch_size=batch_size, shuffle_masks=False)

    losses = evaluate_mean_imputer(dataloader)

    assert losses["mse"] >= 0.0
    assert losses["mae"] >= 0.0

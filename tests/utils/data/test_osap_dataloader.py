import pandas as pd
import pytest

from stdm.utils.data import OSAPDataset, OSAPDataLoader
from tests import mock_data, filepath, file_exists

dataset = OSAPDataset(mock_data)


@pytest.mark.parametrize(
    "batch_size,dataloader_len,n_stocks_per_batch,n_period_per_batch",
    [
        (1, 6, [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]),
        (2, 3, [1, 2, 2], [2, 2, 2]),
        (3, 2, [2, 3], [3, 3]),
        (4, 2, [3, 2], [4, 2]),
        (5, 2, [4, 1], [5, 1]),
        (6, 1, [5], [6]),
        (7, 1, [5], [6]),
    ],
)
def test_osap_dataloader_batch_size(batch_size, dataloader_len, n_stocks_per_batch, n_period_per_batch):
    dataloader = OSAPDataLoader(dataset, batch_size=batch_size)
    assert len(dataloader) == dataloader_len

    for i, (x, time, m_miss, m) in enumerate(dataloader):
        assert x.shape == (n_stocks_per_batch[i], n_period_per_batch[i], 2)
        assert time.shape == (1, n_period_per_batch[i], 10)
        assert m_miss.shape == (n_stocks_per_batch[i], n_period_per_batch[i], 2)
        assert m.shape == (n_stocks_per_batch[i], n_period_per_batch[i], 2)


@pytest.mark.skipif(not file_exists, reason=f"OSAP datafile not found: {filepath}")
def test_dataloader_on_mock_data():
    df = pd.read_csv(filepath, nrows=100)
    dataset = OSAPDataset(df)
    dataloader = OSAPDataLoader(dataset, batch_size=1)

    for i, (x, time, m_miss, m) in enumerate(dataloader):
        pass

import numpy as np
import pandas as pd
import pytest
import torch

from stdm.utils.data import OSAPDataset
from tests import filepath, file_exists, mock_data


def test_osap_dataset():
    dataset = OSAPDataset(mock_data)

    x, time, m_miss = dataset[0]
    assert torch.allclose(
        x,
        torch.tensor(
            [
                [
                    [0.0, 0.0],  # stock 1, 201911
                ],
            ]
        ),
        equal_nan=True,
    )
    # assert torch.allclose(time, torch.tensor([201911], dtype=torch.int))
    assert time.size() == (1, 1, 10)
    assert torch.allclose(
        m_miss,
        torch.tensor(
            [
                [
                    [0, 0],  # stock 1, 201911
                ],
            ],
            dtype=torch.int32,
        ),
    )

    x, time, m_miss = dataset[3]
    assert torch.allclose(
        x,
        torch.tensor(
            [
                [
                    [0.3, 0.3],  # stock 3, 202003
                ],
            ]
        ),
        equal_nan=True,
    )
    # assert torch.allclose(time, torch.tensor([202003], dtype=torch.int))
    assert time.size() == (1, 1, 10)
    assert torch.allclose(
        m_miss,
        torch.tensor(
            [
                [
                    [1, 1],  # stock 3, 202003
                ],
            ],
            dtype=torch.int32,
        ),
    )

    x, time, m_miss = dataset[[201911, 202002]]
    assert torch.allclose(
        x,
        torch.tensor(
            [
                [
                    [0.0, 0.0],  # stock 1, 201911
                    [0.1, 0.0],  # stock 1, 202001
                    [0.0, 0.0],  # stock 1, 202002 no record
                ],
                [
                    [0.0, 0.0],  # stock 2, 201911 no record
                    [0.0, 0.0],  # stock 2, 202001 no record
                    [0.0, 0.2],  # stock 2, 202002
                ],
            ]
        ),
        equal_nan=True,
    )
    # assert torch.allclose(time, torch.tensor([201911, 202001, 202002], dtype=torch.int))
    assert time.size() == (1, 3, 10)
    assert torch.allclose(
        m_miss,
        torch.tensor(
            [
                [
                    [0, 0],  # stock 1, 201911
                    [1, 0],  # stock 1, 202001
                    [0, 0],  # stock 1, 202002 no record
                ],
                [
                    [0, 0],  # stock 2, 201911 no record
                    [0, 0],  # stock 2, 202001 no record
                    [0, 1],  # stock 2, 202002
                ],
            ],
            dtype=torch.int32,
        ),
    )


@pytest.mark.skipif(not file_exists, reason=f"OSAP datafile not found: {filepath}")
def test_on_mock_data():
    df = pd.read_csv(filepath, nrows=100)
    dataset = OSAPDataset(df)

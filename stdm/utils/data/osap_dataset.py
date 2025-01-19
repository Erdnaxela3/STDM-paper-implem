import logging
import math

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class OSAPDataset(Dataset):
    """
    Dataset for the OSAP model.
    The dataset must be a pandas DataFrame with the following columns (+ additional feature columns):
    - yyyymm: time identifier
    - permno: entity identifier

    The dataset is sorted by yyyymm to sample data in a time-ordered manner.
    Data is returned as a tuple (x, time, m_miss), where:
    - x: features # Note permnos were removed, no idea how, but the model performed much better without them
    - time: time identifier (yyyymm)
    - m_miss: missing mask for the features (1 if missing, 0 otherwise)
    """

    def __init__(self, data: pd.DataFrame):
        self.data = data

    def __len__(self):
        unique_yyyymms = sorted(self.data["yyyymm"].unique())
        return len(unique_yyyymms)

    def __getitem__(self, idx: list | int, **kwargs):
        unique_yyyymms = sorted(self.data["yyyymm"].unique())
        if isinstance(idx, int):
            idx = [unique_yyyymms[idx], unique_yyyymms[idx]]
        if isinstance(idx, list) and kwargs.get("is_indices", False):
            idx = [unique_yyyymms[idx[0]], unique_yyyymms[idx[1]]]

        logging.debug(f"{self.__class__.__name__}: Getting data from {idx[0]} to {idx[1]}")

        # Get all the rows between the two time periods
        data = self.data[(self.data["yyyymm"] >= idx[0]) & (self.data["yyyymm"] <= idx[1])]
        unique_permnos = sorted(data["permno"].unique())
        unique_yyyymms = sorted(data["yyyymm"].unique())

        logging.debug(f"{self.__class__.__name__}: records: {len(data)}, permnos: {len(unique_permnos)}")

        # Create a grid for all combinations of permno and yyyymm
        grid = pd.MultiIndex.from_product([unique_permnos, unique_yyyymms], names=["permno", "yyyymm"])
        grid_df = pd.DataFrame(index=grid).reset_index()

        merged = pd.merge(grid_df, data, how="left", on=["permno", "yyyymm"])

        features = [col for col in merged.columns if col not in ["yyyymm", "permno"]]

        pivoted = merged.pivot(index="permno", columns="yyyymm", values=features)

        tensor_data = []
        for permno in pivoted.index:
            permno_data = []
            for yyyymm in unique_yyyymms:
                row = [pivoted.at[permno, (feature, yyyymm)] for feature in features]
                permno_data.append(row)
            tensor_data.append(permno_data)

        x = torch.tensor(tensor_data, dtype=torch.float)

        dates = pd.to_datetime(unique_yyyymms, format="%Y%m")
        date_features = np.array(
            [
                [
                    [
                        dt.year,  # Year
                        dt.month,  # Month
                        dt.day,  # Day
                        dt.hour,  # Hour
                        dt.minute,  # Minute
                        dt.second,  # Second
                        dt.weekday(),  # Day of the week
                        dt.dayofyear,  # Day of the year
                        np.sin(2 * np.pi * dt.month / 12),  # Month as sine
                        np.cos(2 * np.pi * dt.month / 12),  # Month as cosine
                    ]
                    for dt in dates
                ]
            ]
        )

        time = torch.tensor(date_features, dtype=torch.float)

        m_miss = 1 - torch.isnan(x).to(torch.int32)

        x[torch.isnan(x)] = 0

        return x, time, m_miss

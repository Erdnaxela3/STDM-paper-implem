import os

import numpy as np
import pandas as pd
import torch

filepath = os.path.join(
    os.path.dirname(__file__),
    "../Data Release 2024.10/Firm Level Characteristics/Full Sets/signed_predictors_dl_wide.csv",
)
file_exists = os.path.exists(filepath)

device = "cuda" if torch.cuda.is_available() else "cpu"

mock_data = pd.DataFrame(
    {
        "permno": [1, 2, 3, 4, 5, 1],
        "yyyymm": [202001, 202002, 202003, 202004, 202005, 201911],
        "feat1": [0.1, np.nan, 0.3, np.nan, 0.5, np.nan],
        "feat2": [np.nan, 0.2, 0.3, 0.4, np.nan, np.nan],
    }
)

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from stdm.utils.data.preprocessing import (
    remove_outliers,
    remove_not_enough_months,
    yyyymm_month_diff,
    standard_scale,
    remove_nan_columns,
    remove_outlier_by_mean_std,
)


@pytest.mark.parametrize(
    "is_inplace",
    [True, False],
)
def test_remove_nan_rows(is_inplace):
    nan_map = np.tri(10, 10)
    nan_map = np.where(nan_map == 0, np.nan, nan_map)
    df = pd.DataFrame(nan_map)
    df.columns = [f"col_{i}" for i in range(10)]

    func_ret = remove_nan_columns(df, threshold=0.2, inplace=is_inplace)
    if not is_inplace:
        assert func_ret is not None, "remove_nan_columns should return a DataFrame if inplace=False"
        assert isinstance(func_ret, pd.DataFrame), "remove_nan_columns should return a DataFrame if inplace=False"
        df = func_ret
    else:
        assert func_ret is None, "remove_nan_columns should be an inplace operation with no return value"

    assert (df.isna().mean() <= 0.2).all()


@pytest.mark.parametrize(
    "is_inplace",
    [True, False],
)
def test_remove_nan_columns(is_inplace):
    nan_map = np.tri(10, 10)
    nan_map = np.where(nan_map == 0, np.nan, nan_map)
    df = pd.DataFrame(nan_map)
    df.columns = [f"col_{i}" for i in range(10)]

    func_ret = remove_nan_columns(df, threshold=0.2, inplace=is_inplace)
    if not is_inplace:
        assert func_ret is not None, "remove_nan_columns should return a DataFrame if inplace=False"
        assert isinstance(func_ret, pd.DataFrame), "remove_nan_columns should return a DataFrame if inplace=False"
        df = func_ret
    else:
        assert func_ret is None, "remove_nan_columns should be an inplace operation with no return value"

    assert "col_10" not in df.columns
    assert "col_9" not in df.columns
    assert (df.isna().mean() <= 0.2).all()


@pytest.mark.parametrize(
    "date1, date2, expected_diff",
    [
        (202001, 202002, 1),
        (202002, 202001, 1),
        (202001, 202001, 0),
        (202002, 202002, 0),
        (202102, 202002, 12),
        (202202, 202002, 24),
    ],
)
def test_yyyymm_month_diff_good(date1, date2, expected_diff):
    assert yyyymm_month_diff(date1, date2) == expected_diff


@pytest.mark.parametrize(
    "date1, date2",
    [
        (202000, 202000),
        (202000, 202001),
        (202001, 202000),
        (202001, 202013),
        (202013, 202001),
        (202013, 202014),
        (202040, 202001),
    ],
)
def test_yyyymm_month_diff_bad(date1, date2):
    with pytest.raises(AssertionError):
        yyyymm_month_diff(date1, date2)


@pytest.mark.parametrize(
    "is_inplace",
    [True, False],
)
def test_not_enough_months(is_inplace):
    enough_months = pd.DataFrame(
        {
            "permno": np.ones(60) * 10000,
            "yyyymm": np.arange(202001, 202061),
        }
    )

    not_enough_months = pd.DataFrame(
        {
            "permno": np.ones(59) * 20000,
            "yyyymm": np.arange(202001, 202060),
        }
    )

    both_good_and_bad = pd.concat([enough_months, not_enough_months])

    func_ret = remove_not_enough_months(both_good_and_bad, L=60, inplace=is_inplace)
    if not is_inplace:
        assert func_ret is not None, "remove_not_enough_months should return a DataFrame if inplace=False"
        assert isinstance(func_ret, pd.DataFrame), "remove_not_enough_months should return a DataFrame if inplace=False"
        both_good_and_bad = func_ret
    else:
        assert func_ret is None, "remove_not_enough_months should be an inplace operation with no return value"

    assert (both_good_and_bad.permno < 20000).all()


@pytest.mark.parametrize(
    "is_inplace",
    [True, False],
)
def test_remove_outliers_handmade_mean_std(is_inplace):
    mock_df = pd.DataFrame(
        {
            "permno": [10000] * 10,
            "yyyymm": np.arange(202001, 202011),
            "mock_feature": [5, 10, 10.1, 0, -0.01, 0.01, np.nan, 0, 0, 0],
            "mock_feature_2": [-1.1, -1, -0.9, -0.01, 0, 0.01, 0.9, 1, 1.1, 1.2],
        }
    )

    func_ret = remove_outlier_by_mean_std(mock_df, "mock_feature", 5, 1, n_std=5, inplace=is_inplace)
    if not is_inplace:
        assert func_ret is not None, "remove_outlier_by_mean_std should return a DataFrame if inplace=False"
        assert isinstance(
            func_ret, pd.DataFrame
        ), "remove_outlier_by_mean_std should return a DataFrame if inplace=False"
        mock_df = func_ret
    else:
        assert func_ret is None, "remove_outlier_by_mean_std should be an inplace operation with no return value"

    assert mock_df.shape[0] == 8
    assert mock_df[~mock_df["mock_feature"].isna()]["mock_feature"].between(0, 10).all()


@pytest.mark.parametrize(
    "is_inplace",
    [True, False],
)
def test_remove_outliers(is_inplace):
    n_samples = int(1e6)
    df = pd.DataFrame(
        {
            "permno": [10000] * n_samples,
            "yyyymm": np.arange(202001, 202001 + n_samples),
            "mock_feature": np.random.normal(0, 1, n_samples),
            "mock_feature_2": np.random.normal(10, 2, n_samples),
        }
    )
    df.loc[0, "mock_feature"] = -5.01
    df.loc[1, "mock_feature"] = -5
    df.loc[2, "mock_feature"] = 5
    df.loc[3, "mock_feature"] = 5.01
    df.loc[4, "mock_feature"] = np.nan

    df.loc[0, "mock_feature_2"] = -0.01
    df.loc[1, "mock_feature_2"] = 0
    df.loc[2, "mock_feature_2"] = 20
    df.loc[3, "mock_feature_2"] = 20.01
    df.loc[4, "mock_feature_2"] = np.nan

    # this should asymptotically be true
    mock_feat1_mean = df["mock_feature"].mean()
    mock_feat1_std = df["mock_feature"].std()
    assert np.abs(mock_feat1_std - 1) < 1e-2

    # this should asymptotically be true
    mock_feat2_mean = df["mock_feature_2"].mean()
    mock_feat2_std = df["mock_feature_2"].std()
    assert np.abs(mock_feat2_std - 2) < 1e-2

    # NaN values should not be removed
    assert df.isna().sum().sum() == 2

    exclude_columns = ["permno", "yyyymm"]

    func_ret = remove_outliers(df, exclude_columns=exclude_columns, inplace=is_inplace)

    if not is_inplace:
        assert func_ret is not None, "remove_outliers should return a DataFrame if inplace=False"
        assert isinstance(func_ret, pd.DataFrame), "remove_outliers should return a DataFrame if inplace=False"
        df = func_ret
    else:
        assert func_ret is None, "remove_outliers should be an inplace operation with no return value"

    feat1_non_nan = df[~df["mock_feature"].isna()]["mock_feature"]
    feat2_non_nan = df[~df["mock_feature_2"].isna()]["mock_feature_2"]
    assert feat1_non_nan.between(mock_feat1_mean - 5 * mock_feat1_std, mock_feat1_mean + 5 * mock_feat1_std).all()
    assert feat2_non_nan.between(mock_feat2_mean - 5 * mock_feat2_std, mock_feat2_mean + 5 * mock_feat2_std).all()


@pytest.mark.parametrize(
    "is_inplace",
    [True, False],
)
def test_standard_scale(is_inplace):
    n_samples = int(1e6)
    mock_df = pd.DataFrame(
        {
            "permno": [10000] * n_samples,
            "yyyymm": np.arange(202001, 202001 + n_samples),
            "mock_feature": np.random.normal(0, 1, n_samples),
            "mock_feature_2": np.random.normal(10, 2, n_samples),
        }
    )
    mock_df.loc[4, "mock_feature"] = np.nan
    mock_df.loc[4, "mock_feature_2"] = np.nan

    permno_before = mock_df["permno"].copy()

    exclude_columns = ["permno", "yyyymm"]

    scaler, returned_df = standard_scale(mock_df, exclude_columns=exclude_columns, inplace=is_inplace)

    assert isinstance(scaler, StandardScaler), "standard_scale should return a StandardScaler object"

    if not is_inplace:
        assert returned_df is not None, "standard_scale should return a DataFrame if inplace=False"
        mock_df = returned_df
    else:
        assert returned_df is None, "standard_scale should not return a DataFrame if inplace=True"

    # NaN values should not be removed
    assert mock_df.isna().sum().sum() == 2

    assert mock_df["permno"].equals(permno_before), "permno column should not be affected by standard scaling"

    for col in mock_df.columns:
        if col in exclude_columns:
            continue
        assert np.isclose(mock_df[col].mean(), 0, atol=1e-2)
        assert np.isclose(mock_df[col].std(), 1, atol=1e-2)

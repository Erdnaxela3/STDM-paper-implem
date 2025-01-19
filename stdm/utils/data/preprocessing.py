import logging

import pandas as pd
from sklearn.preprocessing import StandardScaler


def remove_nan_rows(df: pd.DataFrame, threshold: float = 2 / 3, inplace=True) -> pd.DataFrame | None:
    """
    Removes rows from a DataFrame that have more than threshold missing values.

    Parameters
    ----------
    df: pd.DataFrame
        The DataFrame to remove rows from.
    threshold: float, default=2/3
        The threshold of missing values to remove rows.
    inplace: bool, default=True
        Whether to perform the operation inplace.

    Returns
    -------
    cleaned_df: pd.DataFrame|None
        If inplace is None else returns the DataFrame with the rows removed.
    """
    initial_size = df.shape[1]

    cleaned_df = df if inplace else df.copy()

    missing = df.isna().mean(axis=1)
    rows_to_remove = missing[missing > threshold].index
    cleaned_df.drop(index=rows_to_remove, inplace=True)

    resulting_size = cleaned_df.shape[0]
    logging.info(f"Removed {initial_size - resulting_size} rows with more than {threshold:.2f} missing values")
    logging.info(f"Shape after removal: {cleaned_df.shape}")

    if not inplace:
        return cleaned_df


def remove_nan_columns(df: pd.DataFrame, threshold: float = 2 / 3, inplace=True) -> pd.DataFrame | None:
    """
    Removes columns from a DataFrame that have more than threshold missing values.

    Parameters
    ----------
    df: pd.DataFrame
        The DataFrame to remove columns from.
    threshold: float, default=2/3
        The threshold of missing values to remove columns.
    inplace: bool, default=True
        Whether to perform the operation inplace.

    Returns
    -------
    cleaned_df: pd.DataFrame|None
        If inplace is None else returns the DataFrame with the columns removed.
    """
    initial_size = df.shape[1]

    cleaned_df = df if inplace else df.copy()

    missing = df.isna().mean()
    columns_to_remove = missing[missing > threshold].index
    cleaned_df.drop(columns=columns_to_remove, inplace=True)

    resulting_size = cleaned_df.shape[1]
    logging.info(f"Removed {initial_size - resulting_size} columns with more than {threshold:.2f} missing values")
    logging.info(f"Shape after removal: {cleaned_df.shape}")

    if not inplace:
        return cleaned_df


def yyyymm_month_diff(date1: int, date2: int) -> int:
    """
    Returns the difference in months between two yyyymm dates.

    Parameters
    ----------
    date1 : int
        The first date in the format yyyymm.
    date2 : int
        The second date in the format yyyymm.

    Returns
    -------
    diff: int
        The difference in months between the two dates.

    Raises
    ------
    AssertionError
        If the month is not between 1 and 12.
    """
    date_1_year, date_1_month = divmod(date1, 100)
    date_2_year, date_2_month = divmod(date2, 100)

    assert date_1_month >= 1 and date_1_month <= 12, f"date_1_month: {date_1_month}, should be between 1 and 12"
    assert date_2_month >= 1 and date_2_month <= 12, f"date_2_month: {date_2_month}, should be between 1 and 12"

    year_diff = abs(date_2_year - date_1_year)
    month_diff = abs(date_2_month - date_1_month)
    diff = year_diff * 12 + month_diff
    return diff


def remove_not_enough_months(df: pd.DataFrame, L: int = 60, inplace=True) -> pd.DataFrame | None:
    """
    Removes all permnos that do not have at least L distinct months of data.

    Parameters
    ----------
    df: pd.DataFrame
        A DataFrame with a 'permno' column and a 'yyyymm' column.
    L: int, default=60
        The minimum number of months required (inclusive).
    inplace: bool, default=True
        Whether to perform the operation inplace.

    Returns
    -------
    cleaned_df: pd.DataFrame|None
        If inplace is None else returns the DataFrame with the permnos removed
    """
    initial_size = df.shape[0]
    cleaned_df = df if inplace else df.copy()

    permno_to_remove = df.groupby("permno").yyyymm.nunique() <= L
    permno_to_remove = permno_to_remove[permno_to_remove == True].index
    cleaned_df.drop(cleaned_df[cleaned_df["permno"].isin(permno_to_remove)].index, inplace=True)

    resulting_size = cleaned_df.shape[0]
    logging.info(f"Removed {initial_size - resulting_size} permnos with less than {L} months of data")
    logging.info(f"Shape after removal: {cleaned_df.shape}")

    if not inplace:
        return cleaned_df


def remove_outlier_by_mean_std(
    df: pd.DataFrame, column: str, mean: float, std: float, n_std: float = 5.0, inplace=True
) -> pd.DataFrame | None:
    """
    Removes outliers from a DataFrame based on a column.
    Outliers are defined as values that are n_std standard deviations away from the mean.
    NaN values are ignored and not removed.

    Parameters
    -----------
    df: pd.DataFrame
        The DataFrame to remove outliers from.
    column: str
        The column to remove outliers from.
    mean: float
        The mean of the column.
    std: float
        The standard deviation of the column.
    n_std: float
        The number of standard deviations to consider an outlier.
    inplace: bool, default=True
        Whether to perform the operation inplace.

    Returns
    -------
    cleaned_df: pd.DataFrame|None
        If inplace is None else returns the DataFrame with the outliers removed
    """
    cleaned_df = df if inplace else df.copy()
    cleaned_df.drop(
        cleaned_df[
            ~(cleaned_df[column].between(mean - n_std * std, mean + n_std * std) | cleaned_df[column].isna())
        ].index,
        inplace=True,
    )

    if not inplace:
        return cleaned_df


def remove_outliers(df: pd.DataFrame, n_std: float = 5.0, exclude_columns=None, inplace=True) -> pd.DataFrame | None:
    """
    Removes outliers from a DataFrame based on a column.
    Outliers are defined as values that are n_std standard deviations away from the mean.
    NaN values are ignored and not removed.

    Parameters
    -----------
    df: pd.DataFrame
        The DataFrame to remove outliers from.
    n_std: float
        The number of standard deviations to consider an outlier.
    exclude_columns: list, default=None
        A list of columns to exclude from outlier removal.
    inplace: bool, default=True
        Whether to perform the operation inplace.

    Returns
    -------
    cleaned_df: pd.DataFrame|None
        If inplace is None else returns the DataFrame with the outliers removed
    """
    if exclude_columns is None:
        exclude_columns = []

    cleaned_df = df if inplace else df.copy()
    initial_size = df.shape[0]

    for column in df.columns:
        if column in exclude_columns:
            continue
        std = df[column].std()
        mean = df[column].mean()
        remove_outlier_by_mean_std(cleaned_df, column, mean, std, n_std, inplace=True)

    resulting_size = cleaned_df.shape[0]
    logging.info(f"Removed {initial_size - resulting_size} outliers")
    logging.info(f"Shape after removal: {cleaned_df.shape}")

    if not inplace:
        return cleaned_df


def standard_scale(df: pd.DataFrame, exclude_columns=None, inplace=True) -> tuple[StandardScaler, pd.DataFrame | None]:
    """
    Standardizes the columns of a DataFrame.

    Parameters
    -----------
    df: pd.DataFrame
        The DataFrame to standardize.
    exclude_columns: list, default=None
        A list of columns to exclude from standardization.
    inplace: bool, default=True
        Whether to perform the operation inplace.

    Returns
    -------
    scaler: sklearn.preprocessing.StandardScaler
        The fitted StandardScaler object.
    cleaned_df: pd.DataFrame|None
        If inplace is None else returns the DataFrame with the columns standardized.
    """
    if exclude_columns is None:
        exclude_columns = []

    scaled_df = df.copy() if not inplace else df

    scaler = StandardScaler()
    column_to_scale = [col for col in df.columns if col not in exclude_columns]
    scaled_df[column_to_scale] = scaler.fit_transform(scaled_df[column_to_scale])
    logging.info("Columns have been standardized")

    if not inplace:
        return scaler, scaled_df
    return scaler, None

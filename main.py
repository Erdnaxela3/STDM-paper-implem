import argparse
import logging
import os

import numpy as np
import pandas as pd
import torch

from stdm.model.evaluate import evaluate_stdm, evaluate_mean_imputer
from stdm.model.training import train_stdm
from stdm.nn import STDM
from stdm.utils.data import OSAPDataset, OSAPDataLoader
from stdm.utils.data.preprocessing import (
    remove_nan_columns,
    remove_not_enough_months,
    remove_outliers,
    standard_scale,
)

logging.basicConfig(
    format="%(asctime)s %(levelname)s | %(message)s",
    level=logging.INFO,
    # level=logging.DEBUG,
    datefmt="%Y-%m-%d %H:%M:%S",
)

device = "cuda" if torch.cuda.is_available() else "cpu"


def main(
    data_file_path: str,
    nan_threshold: float,
    not_enough_months_L: int,
    outlier_n_std: float,
    train_start_month: int,
    train_end_month: int,
    val_start_month: int,
    val_end_month: int,
    test_start_month: int,
    test_end_month: int,
    train_model: bool,
    epochs: int,
    lr: float,
    train_batch_size: int,
    val_batch_size: int,
    test_batch_size: int,
    load_model_path: str,
    save_model_dir: str,
    save_datasets: bool = False,
    no_preprocessing: bool = False,
    train_pickle: str = "train.pkl",
    val_pickle: str = "val.pkl",
    test_pickle: str = "test.pkl",
    evaluate_only: bool = False,
):
    logging.info("Loading data")

    if not no_preprocessing:
        df = pd.read_csv(data_file_path)

        assert "permno" in df.columns, "permno column not found, please check the data file"
        assert "yyyymm" in df.columns, "yyyymm column not found, please check the data file"

        logging.info("Data loaded")
        logging.info(f"Data shape: {df.shape}")

        remove_nan_columns(df, threshold=nan_threshold, inplace=True)
        # remove_nan_rows(df, threshold=5/131, inplace=True)

        train = df[(df["yyyymm"] > train_start_month) & (df["yyyymm"] <= train_end_month)].copy()
        val = df[(df["yyyymm"] > val_start_month) & (df["yyyymm"] <= val_end_month)].copy()
        test = df[(df["yyyymm"] > test_start_month) & (df["yyyymm"] <= test_end_month)].copy()

        remove_not_enough_months(train, L=not_enough_months_L, inplace=True)
        remove_not_enough_months(val, L=not_enough_months_L, inplace=True)
        remove_not_enough_months(test, L=not_enough_months_L, inplace=True)

        remove_outliers(train, n_std=outlier_n_std, exclude_columns=["permno", "yyyymm"], inplace=True)
        remove_outliers(val, n_std=outlier_n_std, exclude_columns=["permno", "yyyymm"], inplace=True)
        remove_outliers(test, n_std=outlier_n_std, exclude_columns=["permno", "yyyymm"], inplace=True)

        scaler, _ = standard_scale(train, exclude_columns=["permno", "yyyymm"], inplace=True)
        column_to_scale = [col for col in df.columns if col not in ["permno", "yyyymm"]]
        val[column_to_scale] = scaler.transform(val[column_to_scale])
        test[column_to_scale] = scaler.transform(test[column_to_scale])
    else:
        if not evaluate_only:
            train = pd.read_pickle(train_pickle)
            val = pd.read_pickle(val_pickle)
        else:
            train = pd.DataFrame(columns=["permno", "yyyymm"])
            val = pd.DataFrame(columns=["permno", "yyyymm"])
        test = pd.read_pickle(test_pickle)

    if save_datasets:
        train.to_pickle("train.pkl")
        val.to_pickle("val.pkl")
        test.to_pickle("test.pkl")

    logging.info(f"Shape of train: {train.shape}")
    logging.info(f"Shape of val: {val.shape}")
    logging.info(f"Shape of test: {test.shape}")

    logging.info(f"{len(train['permno'].unique())} permnos in train, records: {train.shape}")
    logging.info(f"{len(val['permno'].unique())} permnos in val, records: {val.shape}")
    logging.info(f"{len(test['permno'].unique())} permnos in test, records: {test.shape}")

    n_features = test.shape[1] - 2
    stdm = STDM(n_features=n_features, n_channels=8, diffusion_output_dim=128, csp_output_dim=512, time_embed_dim=10)

    if train_model:
        train_loader = OSAPDataLoader(OSAPDataset(train), batch_size=train_batch_size, shuffle_masks=False)
        val_loader = OSAPDataLoader(OSAPDataset(val), batch_size=val_batch_size, shuffle_masks=True)

        logging.info("Training model")
        train_stdm(stdm, train_loader, val_loader, epochs, lr, device, save_model_dir=save_model_dir)
    elif load_model_path:
        stdm.load_state_dict(torch.load(load_model_path, weights_only=True, map_location=device))
        stdm.to(device)

    test_loader = OSAPDataLoader(OSAPDataset(test), batch_size=test_batch_size, shuffle_masks=False)

    stdm_losses = evaluate_stdm(stdm, test_loader, n_steps=5, save_original_data=True)
    mean_imputer_losses = evaluate_mean_imputer(test_loader)

    logging.info(f"STDM Results: MAE: {stdm_losses['mae']:.4f}, RMSE: {np.sqrt(stdm_losses['mse']):.4f}")
    logging.info(
        f"Mean Imputer Results: MAE: {mean_imputer_losses['mae']:.4f}, RMSE: {np.sqrt(mean_imputer_losses['mse']):.4f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # data preprocessing
    parser.add_argument("--no-preprocessing", action="store_true", help="Do not preprocess the data")
    parser.add_argument("--evaluate-only", action="store_true", help="Only load test data and evaluate the model")
    parser.add_argument("--save-datasets", action="store_true", help="Save the preprocessed datasets")
    parser.add_argument("--train-pickle", type=str, default="train.pkl", help="Path to the train dataset pickle")
    parser.add_argument("--val-pickle", type=str, default="val.pkl", help="Path to the validation dataset pickle")
    parser.add_argument("--test-pickle", type=str, default="test.pkl", help="Path to the test dataset pickle")

    parser.add_argument("--data-file-path", type=str, help="Path to the csv file containing the data")
    parser.add_argument(
        "--nan-threshold", type=float, default=2 / 3, help="Threshold to remove columns with NaN values"
    )
    parser.add_argument("--not-enough-months-L", type=int, default=60, help="Minimum number of months to keep a firm")
    parser.add_argument(
        "--outlier-n-std", type=float, default=5, help="Number of standard deviations to consider as outliers"
    )

    # data split
    parser.add_argument("--train-start-month", type=int, default=197001, help="Start month for the training set")
    parser.add_argument(
        "--train-end-month", type=int, default=199512, help="End month for the training set (inclusive)"
    )
    parser.add_argument("--val-start-month", type=int, default=199512, help="Start month for the validation set")
    parser.add_argument(
        "--val-end-month", type=int, default=200512, help="End month for the validation set (inclusive)"
    )
    parser.add_argument("--test-start-month", type=int, default=200512, help="Start month for the test set")
    parser.add_argument("--test-end-month", type=int, default=201912, help="End month for the test set (inclusive)")

    # model
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--epochs", type=int, default=400, help="Number of epochs for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for training")

    parser.add_argument(
        "--train-batch-size", type=int, default=2, help="Batch size for training (number of period per batch)"
    )
    parser.add_argument(
        "--val-batch-size", type=int, default=2, help="Batch size for validation (number of period per batch)"
    )
    parser.add_argument(
        "--test-batch-size", type=int, default=3, help="Batch size for testing (number of period per batch)"
    )

    parser.add_argument("--load-model-path", type=str, help="Path to the model to load")
    parser.add_argument("--save-model-dir", type=str, help="Directory to save the model", default="saved_models")

    # evaluation

    args = parser.parse_args()

    if args.train and args.load_model_path:
        raise ValueError("Cannot train and load a model at the same time")

    if args.train and not os.path.exists(args.save_model_dir):
        os.makedirs(args.save_model_dir)

    if args.load_model_path and not os.path.exists(args.load_model_path):
        raise ValueError(f"Model file not found: {args.load_model_path}")

    main(
        args.data_file_path,
        args.nan_threshold,
        args.not_enough_months_L,
        args.outlier_n_std,
        args.train_start_month,
        args.train_end_month,
        args.val_start_month,
        args.val_end_month,
        args.test_start_month,
        args.test_end_month,
        args.train,
        args.epochs,
        args.lr,
        args.train_batch_size,
        args.val_batch_size,
        args.test_batch_size,
        args.load_model_path,
        args.save_model_dir,
        args.save_datasets,
        args.no_preprocessing,
        args.train_pickle,
        args.val_pickle,
        args.test_pickle,
        args.evaluate_only,
    )

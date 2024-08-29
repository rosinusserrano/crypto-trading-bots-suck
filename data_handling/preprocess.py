import os
import pandas as pd
import torch
import numpy as np
from datetime import datetime
from constants.actions import Actions
from typing import Callable


def make_train_and_test(preprocess_fn: Callable,
                        preprocess_kwargs: dict,
                        start_testing_from: datetime,
                        symbol_filename: str | None = None,
                        verbose: bool = False):
    """Make training and testing data"""
    filenames = get_available_symbols(
        whole_filenames=True) if symbol_filename is None else [
            symbol_filename
        ]

    if verbose:
        print(f"Symbols: {filenames}")

    dataframe = pd.concat([pd.read_csv(f"data/{f}") for f in filenames])
    train, test = split_data(dataframe, start_testing_from)

    if verbose:
        print(f"Train data: {len(train)} entries")
        if len(train) > 0:
            print(f"\tlast: {train['timestamp'].iloc[-1]}")
        print(f"Test data: {len(test)} entries")
        print(f"\tfirst: {test['timestamp'].iloc[0]}")

    try:
        Xtrain, Ytrain = preprocess_fn(train, **preprocess_kwargs)
        if verbose:
            print(f"Train data: {Xtrain.shape, Ytrain.shape}")
    except Exception as e:
        print(e)
        Xtrain, Ytrain = None, None
        if verbose:
            print("No training data for symbol")

    try:
        Xtest, Ytest = preprocess_fn(test, **preprocess_kwargs)
        if verbose:
            print(f"Test data: {Xtest.shape, Ytest.shape}")
    except Exception as e:
        print(e)
        Xtest, Ytest = None, None
        if verbose:
            print("No testing data for symbol")

    return Xtrain, Ytrain, Xtest, Ytest


def get_available_symbols(whole_filenames: bool = False):
    """Get all available symbols"""
    if whole_filenames:
        return os.listdir("data")
    else:
        return [filename.split("-")[0] for filename in os.listdir("data")]


def split_data(data: pd.DataFrame, start_testing_from: datetime):
    """Split data into training and testing data"""
    data_train = data[pd.to_datetime(data["timestamp"]) < start_testing_from]
    data_test = data[pd.to_datetime(data["timestamp"]) >= start_testing_from]

    return data_train, data_test


def rates_of_change_regression(data: pd.DataFrame, window_size: int):
    """Preprocess data to predict rates of change"""
    rate_of_change = ((data["close"] - data["open"]) / data["open"]).to_numpy()

    X = rate_of_change[:-1]
    X = np.lib.stride_tricks.sliding_window_view(X, window_size)
    Y = rate_of_change[window_size:]

    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)

    return X, Y


def rates_of_change_categorical(data: pd.DataFrame, window_size: int,
                                fee: float):
    """Preprocess data to predict buy/sell/hold based on rates of change"""

    def action_index(y_regr):
        if y_regr > fee:
            return Actions.BUY.value

        if y_regr < -fee:
            return Actions.SELL.value

        return Actions.HOLD.value

    X, Y_regr = rates_of_change_regression(data, window_size)

    Y = np.apply_along_axis(action_index, 0,
                            Y_regr.detach().cpu().numpy()[None, :])
    Y = torch.tensor(Y, dtype=torch.long)

    return X, Y

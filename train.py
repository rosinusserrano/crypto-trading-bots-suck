"""Training module for the trading model."""
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Callable

import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm

import wandb

from models.base import TradingModule
from data_handling.preprocess import (make_train_and_test,
                                      rates_of_change_categorical,
                                      rates_of_change_regression,
                                      get_available_symbols)
from evaluation.simulate import simulate_trading
from models.convolutional_variational_regression import ConvolutionalVariationalRegression, ConvolutionalVariationalRegressionConfig
from models.fully_connected_cross_entropy_classifier import FullyConnectedCrossEntropyClassifier, FullyConnectedCrossEntropyClassifierConfig


@dataclass
class TrainConfig:
    """Training configuration"""
    epochs: int
    batch_size: int
    fee: float
    start_testing_from: datetime
    optimizer: torch.optim.Optimizer
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def train(model: TradingModule, config: TrainConfig):
    """Train the model"""
    print("Training the model with the following configuration:\n\n",
          asdict(config))

    # Initialize wandb
    wandb_run = wandb.init(project="bybit-trading-bot")
    wandb_run.config.update({
        "Training configuration": asdict(config),
        "Model configuration": asdict(model.config)
    })

    print("Wandb initialized successfully!")
    print(f"Wandb run name: {wandb_run.name}")

    # Load the data
    preprocess_fn = rates_of_change_categorical if model.config.output_type == "classification" else rates_of_change_regression

    preprocess_keywords = {"window_size": model.config.window_size}
    if model.config.output_type == "classification":
        preprocess_keywords["fee"] = config.fee

    Xtrain, Ytrain, Xtest, Ytest = make_train_and_test(
        preprocess_fn=preprocess_fn,
        preprocess_kwargs=preprocess_keywords,
        start_testing_from=config.start_testing_from,
        verbose=True)

    # Make pytorch dataset and dataloader
    dataset = TensorDataset(torch.tensor(Xtrain, dtype=torch.float32),
                            torch.tensor(Ytrain, dtype=torch.float32))
    dataloader = DataLoader(dataset,
                            batch_size=config.batch_size,
                            shuffle=True)

    # Create data for test runs
    # This is not really a different validation set
    # It's just the test set splitted by symbol so
    # that we can perform test runs for trading
    Yval_dict = {}
    Xval_dict = {}
    for filename in get_available_symbols(whole_filenames=True):
        symbol = filename.split('-')[0]
        _, _, X_tmp, Y_tmp = make_train_and_test(
            rates_of_change_regression,
            {'window_size': model.config.window_size},
            config.start_testing_from, filename)
        Xval_dict[symbol] = X_tmp
        Yval_dict[symbol] = Y_tmp

    best_avg_return = -torch.inf
    bext_avg_log_return = -torch.inf

    # Training loop
    for epoch in range(config.epochs):
        model.train()
        for Xbatch, Ybatch in tqdm(dataloader):
            config.optimizer.zero_grad()
            out = model(Xbatch)
            ypred = model.get_prediction(out)
            loss = config.loss_fn(ypred, Ybatch)
            loss.backward()
            config.optimizer.step()

        wandb.log({"Loss": loss.item()}, commit=False)
        print(f"Epoch {epoch}: Loss: {loss.item()}")

        # Validation / Testing
        with torch.no_grad():
            model.eval()

            test_preds = model(torch.tensor(Xtest, dtype=torch.float32))
            test_mean = model.get_prediction(test_preds)
            test_loss = config.loss_fn(
                test_mean, torch.tensor(Ytest, dtype=torch.float32))
            wandb.log({"Test loss": test_loss.item()}, commit=False)

            if epoch % 1 == 0:
                avg_final_return = 0
                avg_final_log_return = 0
                for symbol, _ in Xval_dict.items():
                    Xval = torch.tensor(Xval_dict[symbol], dtype=torch.float32)
                    Yval = torch.tensor(Yval_dict[symbol], dtype=torch.float32)

                    returns = simulate_trading(model, Xval, Yval, config.fee)
                    final_return = returns[-1]

                    wandb.log({f"final_return_{symbol}": final_return.item()},
                              commit=False)

                    avg_final_return += final_return
                    avg_final_log_return += torch.log(final_return)

                avg_final_return /= len(Xval_dict)
                avg_final_log_return /= len(Xval_dict)

                wandb.log({"avg_final_return": avg_final_return.item()},
                          commit=False)
                wandb.log(
                    {"avg_final_log_return": avg_final_log_return.item()},
                    commit=False)

                if avg_final_return > best_avg_return:
                    best_avg_return = avg_final_return
                    model.save(
                        f"{wandb_run.name}_best_avg_return_{best_avg_return.item():.2f}"
                    )
                if avg_final_log_return > bext_avg_log_return:
                    bext_avg_log_return = avg_final_log_return
                    model.save(
                        f"{wandb_run.name}_best_avg_log_return_{bext_avg_log_return.item():.2f}"
                    )

        wandb.log({"Epoch": epoch}, commit=True)

    print("Training completed successfully!")
    model.save(f"{wandb_run.name}_FINAL")
    print("Model saved successfully!")


if __name__ == "__main__":
    model_config = FullyConnectedCrossEntropyClassifierConfig(
        window_size=64,
        hidden_sizes=[64, 64, 64, 64, 128],
        confidence_threshold=0.3,
        dropout_rate=0.1,
        residual_connections=True)
    model = FullyConnectedCrossEntropyClassifier(model_config)

    train_config = TrainConfig(
        epochs=10000,
        batch_size=1024,
        fee=0.00055,
        start_testing_from=datetime.fromisoformat("2024-06-01T00:00:00"),
        optimizer=torch.optim.AdamW(model.parameters(), lr=0.001),
        loss_fn=torch.nn.MSELoss())

    train(model, train_config)

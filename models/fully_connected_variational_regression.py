"""Basic models for trading."""

from dataclasses import dataclass

import torch
from torch import nn

from models.base import TradingModule
from constants.actions import Actions
from models.base import BaseConfig


@dataclass
class FullyConnectedVariationalRegressionConfig(BaseConfig):
    """Fully connected variational regression configuration"""
    window_size: int
    hidden_sizes: list[int]
    dropout_rate: float = 0.1
    residual_connections: bool = True
    confidence_threshold: float = 0.1


class FullyConnectedVariationalRegression(TradingModule):
    """Fully connected variational regression model"""

    def __init__(
        self,
        config: FullyConnectedVariationalRegressionConfig,
    ):
        """Initialize the trading neural network"""
        super().__init__(config)
        self.config = config

        self.fc_input = nn.Linear(self.config.window_size, self.config.hidden_sizes[0])

        self.hidden_layers = nn.ModuleList()
        for i in range(len(config.hidden_sizes) - 1):
            self.hidden_layers.append(
                nn.Linear(config.hidden_sizes[i], config.hidden_sizes[i + 1]))

        self.fc_mean = nn.Linear(config.hidden_sizes[-1], 1)
        self.fc_logvar = nn.Linear(config.hidden_sizes[-1], 1)

        self.dropout = nn.Dropout(config.dropout_rate)

        self.activation = nn.SiLU()

    def forward(self, x):
        """Forward pass"""
        if self.config.residual_connections and self.config.window_size == self.config.hidden_sizes[
                0]:
            x = self.activation(self.fc_input(x)) + x
        else:
            x = self.activation(self.fc_input(x))

        if self.config.dropout_rate > 0:
            x = self.dropout(x)

        for i, layer in enumerate(self.hidden_layers):
            if self.config.residual_connections and self.config.hidden_sizes[
                    i] == self.config.hidden_sizes[i + 1]:
                x = self.activation(layer(x)) + x
            else:
                x = self.activation(layer(x))

            if self.config.dropout_rate > 0:
                x = self.dropout(x)

        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)

        z = torch.randn_like(mean)

        return (mean + z * torch.exp(logvar * 0.5)).squeeze(), mean, logvar

    def make_choice(self, prediction, fee) -> list[Actions]:
        """Predict the output"""
        _, mean, logvar = prediction

        std = torch.exp(logvar * 0.5)

        choices = []
        for i in range(len(mean)):
            if std[i] / torch.abs(mean[i]) > self.config.confidence_threshold:
                choices.append(Actions.HOLD)
            if mean[i] > fee:
                choices.append(Actions.BUY)
            if mean[i] < -fee:
                choices.append(Actions.SELL)
            choices.append(Actions.HOLD)

        return choices

"""Convolutional variational regression model"""

from dataclasses import dataclass
from typing import Literal
import torch
from torch import nn
from models.base import BaseConfig, TradingModule
from constants.actions import Actions


@dataclass
class ConvolutionalVariationalRegressionConfig(BaseConfig):
    """Convolutional variational regression configuration"""
    window_size: int
    output_type: Literal['regression', 'classification']
    hidden_channels: list[int]
    kernel_size: int
    padding: int
    dropout_rate: float = 0.1
    residual_connections: bool = True
    confidence_threshold: float = 0.1


class ConvolutionalVariationalRegression(TradingModule):
    """Fully connected variational regression model"""

    def __init__(
        self,
        config: ConvolutionalVariationalRegressionConfig,
    ):
        """Initialize the trading neural network"""
        super().__init__(config)
        self.config = config

        self.input_block = ConvolutionBlock(1, config.hidden_channels[0],
                                            config.kernel_size, config.padding,
                                            config.residual_connections)

        self.hidden_blocks = nn.ModuleList()

        for i in range(1, len(config.hidden_channels)):
            self.hidden_blocks.append(
                ConvolutionBlock(config.hidden_channels[i - 1],
                                 config.hidden_channels[i], config.kernel_size,
                                 config.padding, config.residual_connections))

        output_size = config.window_size / (2**(len(config.hidden_channels) -
                                                1))
        output_channels = config.hidden_channels[-1]
        assert int(
            output_size
        ) == output_size and output_size > 0, "Check dimensions and code, sorry for vage error message"
        num_features = int(output_channels * output_size)

        self.fc_mean = nn.Linear(num_features, 1)
        self.fc_logvar = nn.Linear(num_features, 1)

        self.dropout = nn.Dropout(config.dropout_rate)
        self.avgpool = nn.AvgPool1d(kernel_size=2, stride=2)
        self.activation = nn.SiLU()

    def forward(self, x):
        """Forward pass"""
        x = x.unsqueeze(1)
        x = self.input_block(x)
        if self.config.dropout_rate > 0:
            x = self.dropout(x)

        for block in self.hidden_blocks:
            x = self.avgpool(x)
            x = block(x)
            if self.config.dropout_rate > 0:
                x = self.dropout(x)

        x = x.view(x.size(0), -1)

        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)

        z = torch.randn_like(mean)

        return (mean + z * torch.exp(logvar * 0.5)).squeeze(), mean, logvar

    def get_prediction(self, output):
        """Get the actual prediction.
        
        Returns the random sampled variable for training
        and the mean for evaluation."""
        if self.training:
            return output[0]
        return output[1]

    def make_choice(self, prediction, fee):
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


class ConvolutionBlock(nn.Module):
    """Convolutional block"""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 padding: int,
                 residual_connections: bool = True,
                 activation: nn.Module = nn.SiLU()):
        super().__init__()

        self.residual_connections = residual_connections

        self.conv1 = nn.Conv1d(in_channels,
                               out_channels,
                               kernel_size=kernel_size,
                               padding=padding)
        self.conv2 = nn.Conv1d(out_channels,
                               out_channels,
                               kernel_size=kernel_size,
                               padding=padding)
        self.conv3 = nn.Conv1d(out_channels,
                               out_channels,
                               kernel_size=kernel_size,
                               padding=padding)

        self.shortcut = nn.Conv1d(
            in_channels, out_channels, kernel_size=1,
            padding=0) if out_channels != in_channels else nn.Identity()

        self.activation = activation

    def forward(self, x):
        """Forward pass"""
        out = self.activation(self.conv1(x))
        out = self.activation(self.conv2(out))
        out = self.activation(self.conv3(out))

        if self.residual_connections:
            out = out + self.shortcut(x)

        return out

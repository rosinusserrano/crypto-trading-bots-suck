from dataclasses import dataclass
import torch
from torch import nn

from models.base import TradingModule
from constants.actions import Actions


@dataclass
class FullyConnectedCrossEntropyClassifierConfig:
    """Fully connected cross entropy classifier configuration"""
    window_size: int
    hidden_sizes: list[int]
    dropout_rate: list[float] | float = 0.1
    residual_connections: bool = True
    confidence_threshold: float = 0.1
    output_type = "classification"


class FullyConnectedCrossEntropyClassifier(TradingModule):
    """Fully connected cross entropy classifier"""

    def __init__(self, config: FullyConnectedCrossEntropyClassifierConfig):
        super().__init__(config)
        self.config = config

        self.fc_input = nn.Linear(config.window_size, config.hidden_sizes[0])

        self.hidden_layers = nn.ModuleList()
        self.batch_norm_layers = nn.ModuleList()
        for i in range(len(config.hidden_sizes) - 1):
            self.hidden_layers.append(
                nn.Linear(config.hidden_sizes[i], config.hidden_sizes[i + 1]))
            self.batch_norm_layers.append(nn.BatchNorm1d(config.hidden_sizes[i + 1]))

        self.fc_output = nn.Linear(config.hidden_sizes[-1], len(Actions))

        self.activation = nn.SiLU()

    def forward(self, x):
        if self.config.residual_connections and self.config.window_size == self.config.hidden_sizes[
                0]:
            x = self.activation(self.fc_input(x)) + x
        else:
            x = self.activation(self.fc_input(x))

        if not (isinstance(self.config.dropout_rate, float) and self.config.dropout_rate == 0):
            x = nn.functional.dropout(x, self.config.dropout_rate[0])

        for i, layer in enumerate(self.hidden_layers):
            if self.config.residual_connections and self.config.hidden_sizes[
                    i] == self.config.hidden_sizes[i + 1]:
                x = self.activation(layer(x)) + x
            else:
                x = self.activation(layer(x))
            
            x = self.batch_norm_layers[i](x)

            if not (isinstance(self.config.dropout_rate, float) and self.config.dropout_rate == 0):
                x = nn.functional.dropout(x, self.config.dropout_rate[i+1], training=self.training)

        out = self.fc_output(x)

        return out

    def make_choice(self, prediction: torch.Tensor, fee) -> list[Actions]:
        softmax_predictions = nn.functional.softmax(prediction, dim=-1)

        sorted_predictions, _ = torch.sort(softmax_predictions, descending=True)

        diff_top2 = sorted_predictions[:, 0] - sorted_predictions[:, 1]

        choices = []
        for i in range(len(diff_top2)):
            if diff_top2[i] > self.config.confidence_threshold:
                choices.append(Actions(torch.argmax(softmax_predictions, dim=-1)[i].item()))
            else:
                choices.append(Actions.HOLD)

        return choices

from dataclasses import dataclass
from torch import nn

from models.base import TradingModule
from constants.actions import Actions


@dataclass
class FullyConnectedCrossEntropyClassifierConfig:
    """Fully connected cross entropy classifier configuration"""
    window_size: int
    hidden_sizes: list[int]
    dropout_rate: float = 0.1
    residual_connections: bool = True
    confidence_threshold: float = 0.1


class FullyConnectedCrossEntropyClassifier(TradingModule):
    """Fully connected cross entropy classifier"""

    def __init__(self, config: FullyConnectedCrossEntropyClassifierConfig):
        super().__init__(config)
        self.config = config

        self.fc_input = nn.Linear(config.window_size, config.hidden_sizes[0])

        self.hidden_layers = nn.ModuleList()
        for i in range(len(config.hidden_sizes) - 1):
            self.hidden_layers.append(
                nn.Linear(config.hidden_sizes[i], config.hidden_sizes[i + 1]))

        self.fc_output = nn.Linear(config.hidden_sizes[-1], len(Actions))

        self.dropout = nn.Dropout(config.dropout_rate)

        self.activation = nn.SiLU()

    def forward(self, x):
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

        out = self.fc_output(x)

        return out

    def make_choice(self, prediction, fee) -> Actions:
        sorted_predictions = prediction.argsort(descending=True)
        difference_top2 = prediction[sorted_predictions[0]] - prediction[
            sorted_predictions[1]]
        if difference_top2 > self.config.confidence_threshold:
            return Actions(sorted_predictions[0])
        return Actions.HOLD

"Random classifier model"

from dataclasses import dataclass

import torch
from torch import nn

from models.base import TradingModule
from constants.actions import Actions


class RandomClassifier(TradingModule):
    """Fully connected cross entropy classifier"""

    def forward(self, x):
        return torch.randn((x.shape[0], len(Actions)))

    def make_choice(self, prediction, fee) -> list[Actions]:
        softmax_predictions = nn.functional.softmax(prediction, dim=1)
        sorted_predictions = softmax_predictions.argsort(descending=True)
        return[Actions(action.item()) for action in sorted_predictions[:, 0]]

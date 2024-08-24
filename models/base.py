"""Base trading module"""
import os
from abc import abstractmethod
from dataclasses import asdict, dataclass
import json
from typing import Literal
import torch
from torch import nn

from constants.actions import Actions


@dataclass
class BaseConfig:
    """Base configuration"""
    window_size: int
    output_type: Literal["regression", "classification"]


class TradingModule(nn.Module):
    """Base trading module"""

    def __init__(self, config: BaseConfig):
        """Initialize the trading module"""
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, x):
        """Forward pass"""
        raise NotImplementedError

    @abstractmethod
    def make_choice(self, prediction, fee) -> Actions:
        """Predict the output"""
        raise NotImplementedError
    
    def get_prediction(self, output):
        """Get the prediction.
        
        Normally this will just return the output of the network,
        but for models that return multiple values, this can be
        overridden to return the actual prediction. In cases where
        the model uses different parts of the output for training
        and evaluation, this function can be used to return the
        correct part of the output.
        
        For example, in the variational models (god knows whether 
        they actually work), the network returns the random variables
        sampled using the reparametrization trick, the mean
        and the logvar. The prediction used for training is the
        random variable, but the mean is used for evaluation.
        """
        return output

    def get_dirname(self, model_id):
        """Get the directory name"""
        return f"model_state_dicts/{self.__class__.__name__}/{model_id}"

    def save(self, model_id):
        """Save the model"""
        if not os.path.exists(self.get_dirname(model_id)):
            os.makedirs(self.get_dirname(model_id))

        torch.save(self.state_dict(),
                   f"{self.get_dirname(model_id)}/weights.pth")

        with open(f"{self.get_dirname(model_id)}/config.json",
                  "w",
                  encoding="utf-8") as f:
            json.dump(asdict(self.config), f)

    def load(self, model_id):
        """Load the model"""
        with open(f"{self.get_dirname(model_id)}/config.json",
                  "r",
                  encoding="utf-8") as f:
            config = json.load(f)
            print("The config of the loaded model is:\n\n",
                  json.dumps(config, indent=2))

        self.load_state_dict(
            torch.load(f"{self.get_dirname(model_id)}/weights.pth"))

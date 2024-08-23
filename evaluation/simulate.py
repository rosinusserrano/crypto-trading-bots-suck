import torch
from models.base import TradingModule

from constants.actions import Actions


def simulate_trading(
    model: TradingModule,
    X: torch.Tensor,
    y: torch.Tensor,
    fee: float,
):
    """Simulate trading"""
    model.eval()

    with torch.no_grad():
        y_pred = model(X)
    
    choices = model.make_choice(y_pred, fee)

    returns = [1] + [
        get_return(rate_of_change, choice, fee)
        for rate_of_change, choice in zip(y, choices)
    ]

    cumprod = torch.cumprod(torch.tensor(returns), dim=0)

    return cumprod


def get_return(rate_of_change, choice: Actions, fee: float):
    """Get the return"""
    if choice == Actions.BUY:
        return 1 + rate_of_change - fee
    if choice == Actions.SELL:
        return 1 - rate_of_change - fee
    else:
        return 1

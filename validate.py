from datetime import datetime
import json

import matplotlib.pyplot as plt
import torch
from data_handling.preprocess import get_available_symbols, make_train_and_test, rates_of_change_regression
from evaluation.simulate import simulate_trading
from models import fully_connected_cross_entropy_classifier as fcXent

MODEL_ID = "soft-snow-86/best_avg_return_1.05"
MODEL_DIR = f"model_state_dicts/FullyConnectedCrossEntropyClassifier/{MODEL_ID}"

with open(f"{MODEL_DIR}/config.json", "r", encoding="utf-8") as file:
    config_json = json.load(file)

config = fcXent.FullyConnectedCrossEntropyClassifierConfig(**config_json)
config.confidence_threshold = 0.6

model = fcXent.FullyConnectedCrossEntropyClassifier(config)
model.load(MODEL_ID)

start_testing_from = datetime.fromisoformat("2024-07-01T00:00:00")
fee = 0.00055

Yval_dict = {}
Xval_dict = {}
for filename in get_available_symbols(whole_filenames=True):
    symbol = filename.split('-')[0]
    _, _, X_tmp, Y_tmp = make_train_and_test(
        rates_of_change_regression,
        {'window_size': model.config.window_size},
        start_testing_from, filename)
    Xval_dict[symbol] = X_tmp
    Yval_dict[symbol] = Y_tmp

final_return = torch.tensor(0.)
for symbol, _ in Xval_dict.items():
    Xval = Xval_dict[symbol]
    Yval = Yval_dict[symbol]
    returns = simulate_trading(model, Xval, Yval, fee)

    final_return += returns[-1] / len(Xval)

    plt.plot(returns.tolist(), label=symbol)

plt.title(f"Avg final return {final_return}")

plt.savefig("figure.png")

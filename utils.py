import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import json
from pathlib import Path
import argparse

from estimators.infoNCE import InfoNCE
from estimators.club import CLUB1
from estimators.mine import MINE


def plot_estimations(est, xlabel, title):

    sns.set_theme(style="whitegrid", context="talk")

    x = np.array(list(est.keys()))
    means = np.array([est[k]["mean"] for k in x])
    stds = np.array([est[k]["std"] for k in x])

    fig, ax = plt.subplots(figsize=(8, 5))

    sns.lineplot(x=x, y=means, marker="o", ax=ax)

    ax.fill_between(
        x,
        means - stds,
        means + stds,
        alpha=0.3
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("MI(X, Z)")
    ax.set_title(title)

    return fig


def build_estimator(name, config, device):

    match name:

        case "infonce":
            model = InfoNCE(
                x_dim=config["x_dim"],
                y_dim=config["y_dim"],
                hidden_dim=config["hidden_dim"],
                temperature=config["temperature"],
            )

        case "club":
            model = CLUB1(
                x_dim=config["x_dim"],
                y_dim=config["y_dim"],
                hidden_dim=config["hidden_dim"],
            )

        case "mine":
            model = MINE(
                x_dim=config["x_dim"],
                y_dim=config["y_dim"],
                hidden_dim=config["hidden_dim"],
                ema_decay=config["ema_decay"],
            )


        

        case _:
            raise ValueError(f"Unknown estimator: {name}")

    return model.to(device)


def build_config(exp_name: str, estimator_name: str):

    config_path = (
        Path("config")
        / estimator_name
        / f"{exp_name}.json"
    )

    if not config_path.exists():
        raise FileNotFoundError(
            f"No config found at {config_path}"
        )

    with open(config_path, "r") as f:
        config = json.load(f)

    return config

def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--estimator", type=str, required=True,
                        choices=["infonce", "club", "mine"],
                        help="Estimator to use. Available options are 'infonce', 'club', and 'mine'."
                        )
    return parser

def setup_logs(exp_name):
    logs_path = Path("logs")
    log_dir = logs_path / exp_name
    log_dir.mkdir(parents=True, exist_ok=True)

    return log_dir
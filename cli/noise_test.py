import argparse
import collections
import json
import random
import numpy as np
import os
import datetime

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.data import GausDropoutNetworkReprs
from utils import plot_estimations, build_estimator, build_config


# --------------------------------------------------
# Arguments
# --------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--estimator", 
                    type=str, 
                    default="infonce",
                    required=True)
args = parser.parse_args()


# --------------------------------------------------
# Experiment Name
# --------------------------------------------------

EXPERIMENT_NAME = "noise_consistency_test"

config = build_config(EXPERIMENT_NAME, args.estimator)

BS = config["batch_size"]
LR = config["learning_rate"]
GRAD_CLIP = config["grad_clip"]
X_SIZE = config["train_samples"]
VAL_SIZE = config["val_samples"]
DIM = config["dim"]
NOISE_LEVELS = config["noise_levels"]
SEED = config["seed"]
N_RUNS = config["n_runs"]
NOISE_SAMPLES = config["noise_samples"]


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    exp_name = (
        f"{args.estimator}_{EXPERIMENT_NAME}_"
        f"dim{DIM}_bs{BS}_runs{N_RUNS}_{timestamp}"
    )

    log_dir = os.path.join("logs", exp_name)
    os.makedirs(log_dir, exist_ok=True)

    # Save config
    with open(os.path.join(log_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    final_results = collections.OrderedDict()

    for NOISE in NOISE_LEVELS:

        print("\n===================================")
        print("Noise level:", NOISE)
        print("===================================")

        run_results = []

        for run in range(N_RUNS):

            print(f"\nRun {run+1}/{N_RUNS}")

            run_seed = SEED + run
            random.seed(run_seed)
            np.random.seed(run_seed)
            torch.manual_seed(run_seed)

            # ---------------- Dataset ----------------

            train_dataset = GausDropoutNetworkReprs(
                dim=DIM,
                noise=NOISE,
                num_samples=X_SIZE,
                noise_samples=NOISE_SAMPLES,
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=BS,
                shuffle=True,
                drop_last=True,
            )

            # ---------------- Model ----------------

            config_run = config.copy()
            config_run["x_dim"] = DIM
            config_run["y_dim"] = DIM

            model = build_estimator(
                args.estimator,
                config_run,
                device,
            )

            optimizer = torch.optim.Adam(
                model.parameters(),
                LR,
            )

            # ---------------- Training ----------------

            model.train()
            train_bar = tqdm(
                train_loader,
                desc=f"Train | noise={NOISE} | run={run+1}",
            )

            for X, Y in train_bar:

                X = X.to(device)
                Y = Y.to(device)

                optimizer.zero_grad()
                loss, mi_estimate = model(X, Y)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    GRAD_CLIP,
                )

                optimizer.step()

                train_bar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    mi=f"{mi_estimate.item():.4f}",
                )

            # ---------------- Validation ----------------

            val_dataset = GausDropoutNetworkReprs(
                dim=DIM,
                noise=NOISE,
                num_samples=VAL_SIZE,
                noise_samples=NOISE_SAMPLES,
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=BS,
                shuffle=False,
                drop_last=True,
            )

            model.eval()
            val_mi_total = 0.0
            num_batches = 0

            with torch.no_grad():
                for X, Y in val_loader:

                    X = X.to(device)
                    Y = Y.to(device)

                    _, mi_estimate = model(X, Y)

                    val_mi_total += mi_estimate.item()
                    num_batches += 1

            val_mi = val_mi_total / num_batches
            run_results.append(val_mi)

            print(
                f"Validation MI (run {run+1}): {val_mi:.4f}"
            )

        # ---------------- Aggregate ----------------

        mean_mi = float(np.mean(run_results))
        std_mi = float(np.std(run_results))

        final_results[NOISE] = {
            "mean": mean_mi,
            "std": std_mi,
            "runs": run_results,
        }

        print(
            f"\n>>> Noise={NOISE} | "
            f"Mean MI={mean_mi:.4f} | Std={std_mi:.4f}"
        )

    # ---------------- Save Results ----------------

    results_path = os.path.join(log_dir, "results.json")

    with open(results_path, "w") as f:
        json.dump(final_results, f, indent=4)

    # ---------------- Plot ----------------

    plot_estimations(
        final_results,
        "Noise",
        f"{args.estimator} MI Estimate (Mean ± Std)",
    )

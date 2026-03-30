import argparse
import collections
import json
import random
import numpy as np
import datetime
from pathlib import Path
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from synthetic_data_generators import GausDropoutEmbedded
from utils import *

if __name__ == "__main__":
    exp_args_parser = setup_parser()
    args = exp_args_parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # --------------------------------------------------
    # Experiment Config
    # --------------------------------------------------
    EXPERIMENT_NAME = "embedding_consistency_test"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = (
        f"{args.estimator}_{EXPERIMENT_NAME}_{timestamp}"
    )

    config = build_config(EXPERIMENT_NAME, args.estimator)

    BS = config["batch_size"]
    LR = config["learning_rate"]
    GRAD_CLIP = config["grad_clip"]
    TRAIN_SAMPLES = config["train_samples"]
    VAL_SAMPLES = config["val_samples"]
    DIM = config["dim"]
    SEED = config["seed"]
    N_RUNS = config["n_runs"]
    # NOISE_SAMPLES = config["noise_samples"]
    NOISE = config["noise"]
    NOISE_SAMPLES = config["noise_samples"]

    # only in this experiemnt type
    # additional dimensions that do not carry any information in X
    # adding dimensions should not change the MI
    ADD_DIMS = config["add_dims"]

    log_dir = setup_logs(exp_name)
    log_file = log_dir / (exp_name + "_config.json")
    # Save config, as a duplicate, but allows for double checking if something was changed compared to the saved one
    with log_file.open("w") as f:
        json.dump(config, f, indent=4)
    # -------------------------------------------------

    final_results = collections.OrderedDict()

    for ADDDIM in ADD_DIMS:

        print("\n===================================")
        print("Added dim:", ADDDIM)
        print("===================================")

        run_results = []

        for run in range(N_RUNS):

            print(f"\nRun {run+1}/{N_RUNS}")

            run_seed = SEED + run
            random.seed(run_seed)
            np.random.seed(run_seed)
            torch.manual_seed(run_seed)

            # ---------------- Dataset ----------------

            dataset = GausDropoutEmbedded(
                dim=DIM,
                noise=NOISE,
                num_samples=TRAIN_SAMPLES + VAL_SAMPLES,
                add_dim=ADDDIM,
                noise_samples=NOISE_SAMPLES
            )

            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset,
                [TRAIN_SAMPLES * NOISE_SAMPLES, VAL_SAMPLES * NOISE_SAMPLES],
                generator=torch.Generator().manual_seed(SEED)
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=BS,
                shuffle=True,
                drop_last=True,
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=BS,
                shuffle=False,
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
            
            #y_dim depends on ADDDIM
            config_run = config.copy()
            config_run["x_dim"] = DIM
            config_run["y_dim"] = DIM + ADDDIM

            model_addim = build_estimator(
                args.estimator,
                config_run,
                device,
            )

            optimizer = torch.optim.AdamW(
                model.parameters(),
                LR,
                weight_decay=0.01
            )

            optimizer_addim = torch.optim.AdamW(
                model_addim.parameters(),
                LR,
                weight_decay=0.01
            )

            if GRAD_CLIP is None:
                clip_fn = lambda params: None
            else:
                clip_fn = lambda params: torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP)


            # ---------------- Training ----------------

            
            train_bar = tqdm(
                train_loader,
                desc=f"Train | add_dim={ADDDIM} | run={run+1}",
            )

            for X, Y, Y_emb in train_bar:

                X = X.to(device)
                Y = Y.to(device)
                Y_emb = Y_emb.to(device)

                # estimator on X Y 
                model.train()
                loss, _ = model(X, Y)
                optimizer.zero_grad()
                loss.backward()
                clip_fn(model.parameters())
                optimizer.step()

                # estimator on X Y_emb
                model_addim.train()
                loss_addim, _ = model_addim(X, Y_emb)
                optimizer_addim.zero_grad()
                loss_addim.backward()
                clip_fn(model_addim.parameters())
                optimizer_addim.step()

                train_bar.set_postfix(
                    loss_xy=f"{loss.item():.4f}",
                    loss_emb=f"{loss_addim.item():.4f}"
                )


            # ---------------- Validation ----------------

            model.eval()
            model_addim.eval()
            val_mi_total = 0.0
            val_mi_total_addim = 0.0
            num_batches = 0

            with torch.no_grad():
                for X, Y, Y_emb in train_loader:

                    X = X.to(device)
                    Y = Y.to(device)
                    Y_emb = Y_emb.to(device)

                    _, mi_estimate = model(X, Y)
                    _, mi_estimate_addim = model_addim(X, Y_emb)

                    val_mi_total += mi_estimate.item()
                    val_mi_total_addim += mi_estimate_addim.item()
                    num_batches += 1

            val_mi = val_mi_total / num_batches
            val_mi_addim = val_mi_total_addim / num_batches

            ratio = val_mi / val_mi_addim
            
            run_results.append({
                "mi_xy": val_mi,
                "mi_emb": val_mi_addim,
                "ratio": ratio
            })


            print(
                f"Run {run+1} | MI(X,Y)={val_mi:.4f} | "
                f"MI(X,Y_emb)={val_mi_addim:.4f} | "
                f"Ratio={ratio:.4f}"
            )

        # ---------------- Aggregate ----------------

        mean_ratio = float(np.mean([r["ratio"] for r in run_results]))
        std_ratio = float(np.std([r["ratio"] for r in run_results]))


        final_results[ADDDIM] = {
            "mean": mean_ratio,
            "std": std_ratio,
            "runs": run_results,
        }

        print(
            f"\n>>> ADDDIM={ADDDIM} | "
            f"Mean MI ratio={mean_ratio:.4f} | Std={std_ratio:.4f}"
        )

    # ---------------- Save Results ----------------

    results_path = log_dir / (exp_name + "_results.json")

    with open(results_path, "w") as f:
        json.dump(final_results, f, indent=4)

    # ---------------- Plot ----------------

    figure = plot_estimations(
        final_results,
        "Added Dimensionality",
        f"{args.estimator} Ratio of MI Estimate (Mean ± Std)",
    )

    plt.tight_layout()
    plt.savefig(log_dir / (exp_name + "_results.png"), dpi=200, format="png")
    plt.show()

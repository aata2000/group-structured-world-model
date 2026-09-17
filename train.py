"""Train a group-structured world model on the synthetic Z4 + pick-up environment.

Usage:
    python train.py
    python train.py --epochs 1000 --seed 0
"""
from __future__ import annotations

import argparse
import random

import numpy as np
import torch
import torch.nn.functional as F

from src.environment import N_ACTIONS, OBS_DIM, sample_trajectories
from src.model import WorldModel

LATENT_DIM = 8
BATCH_SIZE = 256
HORIZON = 8
LR = 1e-3
COMMITMENT_WEIGHT = 0.01  # entropy penalty that sharpens gates toward 0/1


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    model = WorldModel(obs_dim=OBS_DIM, latent_dim=LATENT_DIM, n_actions=N_ACTIONS)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    log_every = max(1, args.epochs // 20)

    for epoch in range(1, args.epochs + 1):
        obs, actions = sample_trajectories(BATCH_SIZE, HORIZON, rng)

        model.train()
        preds = model.rollout(obs[:, 0], actions)
        targets = obs[:, 1:]
        pred_loss = sum(
            F.mse_loss(preds[t], targets[:, t]) for t in range(HORIZON)
        ) / HORIZON

        # Binary entropy of the gate probabilities: pushes each (action, dim)
        # gate toward a confident 0 or 1 instead of sitting at 0.5. No labels
        # tell the model which actions are group-like -- this only sharpens
        # whatever split the prediction loss has already found.
        p = model.transition.gate_probs()
        eps = 1e-6
        commitment_loss = -(p * torch.log(p + eps) + (1 - p) * torch.log(1 - p + eps)).mean()

        loss = pred_loss + COMMITMENT_WEIGHT * commitment_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % log_every == 0:
            print(
                f"epoch {epoch:5d}  pred_loss {pred_loss.item():.5f}  "
                f"mean_gate {p.mean().item():.3f}  "
                f"commit_loss {commitment_loss.item():.5f}"
            )

    torch.save(
        {"model_state_dict": model.state_dict(), "latent_dim": LATENT_DIM},
        "checkpoint.pt",
    )
    print("Saved checkpoint to checkpoint.pt")


if __name__ == "__main__":
    main()

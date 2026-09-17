"""Train a recurrent group-structured world model on the partially observed
DoorKey-like environment.

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

from src.environment import N_ACTIONS, N_GROUP_ACTIONS, OBS_DIM, STATE_DIM, sample_trajectories
from src.model import WorldModel

LATENT_DIM = 8
BATCH_SIZE = 256
HORIZON = 10
LR = 1e-3


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

    model = WorldModel(
        obs_dim=OBS_DIM,
        state_dim=STATE_DIM,
        latent_dim=LATENT_DIM,
        n_actions=N_ACTIONS,
        n_group_actions=N_GROUP_ACTIONS,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    log_every = max(1, args.epochs // 20)

    for epoch in range(1, args.epochs + 1):
        states, obs_seq, actions = sample_trajectories(BATCH_SIZE, HORIZON, rng)

        model.train()
        preds = model.teacher_forced(obs_seq, actions)
        targets = states[:, 1:]
        loss = sum(
            F.mse_loss(preds[t], targets[:, t]) for t in range(HORIZON)
        ) / HORIZON

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % log_every == 0:
            print(f"epoch {epoch:5d}  loss {loss.item():.5f}")

    torch.save(
        {"model_state_dict": model.state_dict(), "latent_dim": LATENT_DIM},
        "checkpoint.pt",
    )
    print("Saved checkpoint to checkpoint.pt")


if __name__ == "__main__":
    main()

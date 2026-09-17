"""Evaluate a trained group-structured world model.

Checks that the model isn't just fitting one-step prediction -- it also
inspects the algebraic structure the network discovered: does turn_left have
cyclic order 4, and is turn_right its inverse?

Usage:
    python validate.py
    python validate.py --checkpoint checkpoint.pt
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from src.environment import ACTION_NAMES, N_ACTIONS, OBS_DIM, sample_trajectories
from src.model import WorldModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=str, default="checkpoint.pt")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--horizon", type=int, default=8)
    return p.parse_args()


def load_model(path: str) -> WorldModel:
    ckpt = torch.load(path, map_location="cpu")
    model = WorldModel(obs_dim=OBS_DIM, latent_dim=ckpt["latent_dim"], n_actions=N_ACTIONS)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def one_step_mse(model: WorldModel, obs: torch.Tensor, actions: torch.Tensor) -> float:
    with torch.no_grad():
        pred = model.step(obs[:, 0], actions[:, 0])
    return F.mse_loss(pred, obs[:, 1]).item()


def rollout_accuracy(model: WorldModel, obs: torch.Tensor, actions: torch.Tensor) -> float:
    """Fraction of trajectories whose closed-loop rollout matches the true
    observation (all fields, rounded to the nearest integer) at the final step."""
    with torch.no_grad():
        preds = model.rollout(obs[:, 0], actions)
    matched = (preds[-1].round() == obs[:, -1]).all(dim=-1)
    return matched.float().mean().item()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    model = load_model(args.checkpoint)
    obs, actions = sample_trajectories(args.batch_size, args.horizon, rng)

    mse = one_step_mse(model, obs, actions)
    acc = rollout_accuracy(model, obs, actions)
    print(f"One-step prediction MSE: {mse:.5f}")
    print(f"{args.horizon}-step rollout accuracy: {acc:.3f}\n")

    with torch.no_grad():
        W = model.transition.W()               # (n_actions, D, D)
        gate_p = model.transition.gate_probs()  # (n_actions, D)
    D = W.shape[-1]
    I = torch.eye(D)

    print(f"{'Action':<12} {'mean gate':>10} {'||W^4-I||':>12}")
    for a, name in enumerate(ACTION_NAMES):
        w4 = torch.linalg.matrix_power(W[a], 4)
        resid = (w4 - I).norm().item()
        print(f"{name:<12} {gate_p[a].mean().item():>10.3f} {resid:>12.4f}")

    left = ACTION_NAMES.index("turn_left")
    right = ACTION_NAMES.index("turn_right")
    inv_err = (W[right] @ W[left] - I).norm().item()
    print(f"\nInverse error ||W_right W_left - I||: {inv_err:.4f}")


if __name__ == "__main__":
    main()

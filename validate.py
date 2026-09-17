"""Evaluate a trained recurrent group-structured world model.

Reports two different things and does not conflate them:

  1. Prediction/rollout accuracy -- does the model work at all.
  2. Operator diagnostics on the *learned* W_turn_left / W_turn_right --
     whether the Lie-parametrized operators the model was free to shape
     happen to satisfy the algebraic relations turning implies (order 4,
     mutual inverses). The group/interaction split itself is fixed by
     construction (see src/model.py); this only checks whether training
     pushed the group operators toward the "correct" rotations, which is
     not guaranteed -- read the printed numbers, don't assume they're small.

Usage:
    python validate.py
    python validate.py --checkpoint checkpoint.pt
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from src.environment import ACTION_NAMES, N_ACTIONS, N_GROUP_ACTIONS, OBS_DIM, STATE_DIM, sample_trajectories
from src.model import WorldModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=str, default="checkpoint.pt")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--horizon", type=int, default=10)
    return p.parse_args()


def load_model(path: str) -> WorldModel:
    ckpt = torch.load(path, map_location="cpu")
    model = WorldModel(
        obs_dim=OBS_DIM,
        state_dim=STATE_DIM,
        latent_dim=ckpt["latent_dim"],
        n_actions=N_ACTIONS,
        n_group_actions=N_GROUP_ACTIONS,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def teacher_forced_accuracy(model: WorldModel, states, obs_seq, actions):
    """One-step-at-a-time accuracy: the encoder re-observes the (partial) real
    trajectory at every step, as during training."""
    with torch.no_grad():
        preds = model.teacher_forced(obs_seq, actions)
    targets = states[:, 1:]
    mse = sum(F.mse_loss(preds[t], targets[:, t]) for t in range(actions.shape[1])) / actions.shape[1]
    final_match = (preds[-1].round() == targets[:, -1]).all(dim=-1).float().mean().item()
    return mse.item(), final_match


def latent_rollout_accuracy(model: WorldModel, states, obs_seq, actions):
    """Pure imagination: only obs_seq[:, 0] is ever encoded; every later state
    comes from composing the transition on z alone, with no further
    observations. Tests whether has_key/door_open -- never re-observed --
    stay correct purely through latent composition of the operators."""
    with torch.no_grad():
        preds = model.rollout(obs_seq[:, 0], actions)
    targets = states[:, 1:]
    final_match = (preds[-1].round() == targets[:, -1]).all(dim=-1).float().mean().item()
    return final_match


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    model = load_model(args.checkpoint)
    states, obs_seq, actions = sample_trajectories(args.batch_size, args.horizon, rng)

    tf_mse, tf_acc = teacher_forced_accuracy(model, states, obs_seq, actions)
    ro_acc = latent_rollout_accuracy(model, states, obs_seq, actions)

    print(f"Teacher-forced next-state MSE:        {tf_mse:.5f}")
    print(f"Teacher-forced final-state accuracy:  {tf_acc:.3f}")
    print(f"Latent-rollout final-state accuracy:  {ro_acc:.3f}  "
          f"(pure imagination over {args.horizon} steps, obs seen once)\n")

    with torch.no_grad():
        W = model.transition.W()  # (N_GROUP_ACTIONS, D, D)
    D = W.shape[-1]
    I = torch.eye(D)

    print(f"{'Action':<12} {'||W^4-I||':>12}")
    for a in range(N_GROUP_ACTIONS):
        w4 = torch.linalg.matrix_power(W[a], 4)
        resid = (w4 - I).norm().item()
        print(f"{ACTION_NAMES[a]:<12} {resid:>12.4f}")

    left = ACTION_NAMES.index("turn_left")
    right = ACTION_NAMES.index("turn_right")
    inv_err = (W[right] @ W[left] - I).norm().item()
    print(f"\nInverse error ||W_right @ W_left - I||: {inv_err:.4f}")


if __name__ == "__main__":
    main()

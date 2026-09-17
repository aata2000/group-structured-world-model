"""Tiny synthetic environment for testing group-structured world models.

State: (orientation, has_key)
    orientation in {0, 1, 2, 3}   -- a Z4 rotation
    has_key     in {0, 1}

Actions:
    0 = turn_left    orientation = (orientation - 1) mod 4   -- group action
    1 = turn_right   orientation = (orientation + 1) mod 4   -- group action, inverse of turn_left
    2 = pick_up      has_key = 1                             -- idempotent, non-invertible

turn_left and turn_right generate the cyclic group Z4: four consecutive turns
in either direction return to the starting orientation, and turn_right undoes
turn_left. pick_up has no inverse and collapses two states (has_key = 0 or 1)
to one -- it is not a group action, and a well-fit model should route it
through the free residual branch rather than an orthogonal operator.
"""
from __future__ import annotations

import numpy as np
import torch

N_ACTIONS = 3
OBS_DIM = 5  # 4 (one-hot orientation) + 1 (has_key)
ACTION_NAMES = ["turn_left", "turn_right", "pick_up"]


def step(orientation: np.ndarray, has_key: np.ndarray, action: np.ndarray):
    """Vectorized transition. All arguments are int arrays of the same shape."""
    orientation = orientation.copy()
    has_key = has_key.copy()
    left = action == 0
    right = action == 1
    pick = action == 2
    orientation[left] = (orientation[left] - 1) % 4
    orientation[right] = (orientation[right] + 1) % 4
    has_key[pick] = 1
    return orientation, has_key


def encode(orientation: np.ndarray, has_key: np.ndarray) -> np.ndarray:
    """(orientation, has_key) -> (N, OBS_DIM) float observation."""
    n = orientation.shape[0]
    obs = np.zeros((n, OBS_DIM), dtype=np.float32)
    obs[np.arange(n), orientation] = 1.0
    obs[:, 4] = has_key
    return obs


def sample_trajectories(batch_size: int, horizon: int, rng: np.random.Generator):
    """Generate `batch_size` random-action trajectories of length `horizon`.

    Returns:
        obs:     (B, T+1, OBS_DIM) float32
        actions: (B, T) int64
    """
    orientation = rng.integers(0, 4, size=batch_size)
    has_key = rng.integers(0, 2, size=batch_size)
    actions = rng.integers(0, N_ACTIONS, size=(batch_size, horizon))

    obs_seq = [encode(orientation, has_key)]
    for t in range(horizon):
        orientation, has_key = step(orientation, has_key, actions[:, t])
        obs_seq.append(encode(orientation, has_key))

    obs = np.stack(obs_seq, axis=1)
    return torch.from_numpy(obs), torch.from_numpy(actions).long()

"""Tiny partially observed DoorKey-like environment.

True state: (orientation, has_key, door_open)
    orientation in {0, 1, 2, 3}   -- a Z4 rotation
    has_key     in {0, 1}
    door_open   in {0, 1}

Actions:
    0 = turn_left    orientation = (orientation - 1) mod 4   -- movement
    1 = turn_right   orientation = (orientation + 1) mod 4   -- movement
    2 = pick_up      has_key = 1                             -- interaction
    3 = open_door    door_open = 1 if has_key else unchanged -- interaction

turn_left/turn_right generate the cyclic group Z4 and are each other's
inverse. pick_up and open_door are not invertible: pick_up collapses
"holding" vs "not holding" the key to the same outcome, and open_door's
effect depends on has_key, which a linear operator cannot condition on.

Partial observability: the observation given to the encoder at each step is
NOT the true state -- it never includes has_key or door_open. It is only
the current orientation plus the action taken on the *previous* step:

    obs_t = [one_hot(orientation_t), one_hot(action_{t-1} or NONE)]

So the only way to know whether the key has been picked up, or the door
opened, is to have tracked the action history -- there is nothing else in
the input to read it off from. This is what gives the GRU encoder in
model.py an actual job to do: `has_key`/`door_open` are supervised only at
the decoder output, never handed to the encoder directly.
"""
from __future__ import annotations

import numpy as np
import torch

N_ACTIONS = 4
N_GROUP_ACTIONS = 2  # turn_left, turn_right
ACTION_NAMES = ["turn_left", "turn_right", "pick_up", "open_door"]
NONE_ACTION = N_ACTIONS  # placeholder "no previous action" token

STATE_DIM = 6  # 4 (one-hot orientation) + has_key + door_open
OBS_DIM = 4 + (N_ACTIONS + 1)  # one-hot orientation + one-hot last action (or NONE)


def step(orientation: np.ndarray, has_key: np.ndarray, door_open: np.ndarray, action: np.ndarray):
    """Vectorized transition. All arguments are int arrays of the same shape."""
    orientation = orientation.copy()
    has_key = has_key.copy()
    door_open = door_open.copy()

    left = action == 0
    right = action == 1
    pick = action == 2
    open_ = action == 3

    orientation[left] = (orientation[left] - 1) % 4
    orientation[right] = (orientation[right] + 1) % 4
    has_key[pick] = 1
    door_open[open_ & (has_key == 1)] = 1

    return orientation, has_key, door_open


def encode_state(orientation: np.ndarray, has_key: np.ndarray, door_open: np.ndarray) -> np.ndarray:
    """(orientation, has_key, door_open) -> (N, STATE_DIM) float, the full true state."""
    n = orientation.shape[0]
    s = np.zeros((n, STATE_DIM), dtype=np.float32)
    s[np.arange(n), orientation] = 1.0
    s[:, 4] = has_key
    s[:, 5] = door_open
    return s


def encode_obs(orientation: np.ndarray, last_action: np.ndarray) -> np.ndarray:
    """(orientation, last_action) -> (N, OBS_DIM) float, the partial observation.

    last_action uses NONE_ACTION for "no previous action" (episode start).
    """
    n = orientation.shape[0]
    obs = np.zeros((n, OBS_DIM), dtype=np.float32)
    obs[np.arange(n), orientation] = 1.0
    obs[np.arange(n), 4 + last_action] = 1.0
    return obs


def sample_trajectories(batch_size: int, horizon: int, rng: np.random.Generator):
    """Generate `batch_size` random-action trajectories of length `horizon`.

    Every episode starts without the key and with the door closed, so
    whether/when each trajectory picks up the key and opens the door varies
    with the random actions -- that variation is what the GRU has to track.

    Returns:
        states:  (B, T+1, STATE_DIM) float32 -- full true state, for supervision
        obs_seq: (B, T, OBS_DIM)     float32 -- partial observation before each action
        actions: (B, T)              int64
    """
    orientation = rng.integers(0, 4, size=batch_size)
    has_key = np.zeros(batch_size, dtype=np.int64)
    door_open = np.zeros(batch_size, dtype=np.int64)
    actions = rng.integers(0, N_ACTIONS, size=(batch_size, horizon))

    states = [encode_state(orientation, has_key, door_open)]
    obs_seq = []
    last_action = np.full(batch_size, NONE_ACTION, dtype=np.int64)

    for t in range(horizon):
        obs_seq.append(encode_obs(orientation, last_action))
        orientation, has_key, door_open = step(orientation, has_key, door_open, actions[:, t])
        states.append(encode_state(orientation, has_key, door_open))
        last_action = actions[:, t]

    states = np.stack(states, axis=1)
    obs_seq = np.stack(obs_seq, axis=1)
    return (
        torch.from_numpy(states),
        torch.from_numpy(obs_seq),
        torch.from_numpy(actions).long(),
    )

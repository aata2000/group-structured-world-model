"""Recurrent group-structured world model.

Architecture (mirrors geometric_v16_recurrent.py):

    obs_t (partial observation)
        -> obs_net (MLP)
        -> GRUCell(h_{t-1})
        -> to_latent (Linear)
        -> z_t

obs_t never reveals the environment's hidden flags directly (see
environment.py) -- only the current orientation and the action just taken.
The GRU is what lets the model track hidden state (e.g. "have I picked up
the key yet?") across a sequence of such partial observations.

Each action then transitions z_t -> z_{t+1}. Which branch an action uses is
fixed in advance by its id, exactly like geometric_v16's `is_group = a < 3`
-- it is not learned or discovered:

  - movement actions (id < n_group_actions) go through a Lie-algebra
    parametrized operator W_a = exp(skew(A_raw_a)), which is always an
    exact rotation in SO(D);
  - interaction actions (id >= n_group_actions) go through a residual MLP
    conditioned on a one-hot action id, since a non-invertible state change
    (e.g. picking up a key) cannot be represented by any invertible linear
    map.

Planning/rollout encodes only the first observation; every subsequent state
is produced by composing the transition directly on z, with no further
observations and no decode/re-encode round trip (`WorldModel.rollout`).
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RecurrentEncoder(nn.Module):
    """(obs_t, h_{t-1}) -> (z_t, h_t)."""

    def __init__(self, obs_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.obs_net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.to_latent = nn.Linear(hidden_dim, latent_dim)

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def forward(self, obs: torch.Tensor, h: torch.Tensor):
        feat = self.obs_net(obs)
        h_next = self.gru(feat, h)
        z = self.to_latent(h_next)
        return z, h_next


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, state_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class GroupStructuredTransition(nn.Module):
    """Per-action transition with a fixed (not learned) group/interaction split.

    Actions with id < n_group_actions use the Lie-group operator W_a; the
    rest use the interaction MLP. `n_actions` counts all actions so the
    interaction MLP's action-conditioning one-hot covers every id, even
    though only the interaction ones ever reach it with a nonzero weight.
    """

    def __init__(
        self,
        latent_dim: int,
        n_actions: int,
        n_group_actions: int,
        hidden_dim: int = 64,
        w_init_scale: float = 0.01,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_actions = n_actions
        self.n_group_actions = n_group_actions

        # Lie-algebra parametrization: W_a = exp(skew(A_raw_a)) in SO(D).
        # Storing A_raw (unconstrained) instead of W_a directly means the
        # skew-symmetric projection and matrix exponential guarantee W_a is
        # an exact rotation at every step of training, with no orthogonality
        # penalty needed. Only group actions get one of these.
        self.A_raw = nn.Parameter(
            torch.randn(n_group_actions, latent_dim, latent_dim) * w_init_scale
        )

        # Interaction branch: one MLP shared across all non-group actions,
        # conditioned on a one-hot action id. Unlike the group branch this
        # is not residual and not invertible -- exactly what's needed for a
        # state change like "pick up the key", which collapses distinct
        # inputs (already holding the key or not) toward the same output.
        self.interaction_net = nn.Sequential(
            nn.Linear(latent_dim + n_actions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def W(self) -> torch.Tensor:
        """(n_group_actions, D, D) rotation matrices, recomputed from A_raw each call."""
        A = 0.5 * (self.A_raw - self.A_raw.transpose(-1, -2))
        return torch.matrix_exp(A)

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        is_group = (action < self.n_group_actions).to(z.dtype).unsqueeze(-1)  # (B, 1)

        # Interaction branch (computed for every sample; masked out below).
        onehot = F.one_hot(action, self.n_actions).to(z.dtype)
        z_interact = self.interaction_net(torch.cat([z, onehot], dim=-1))

        # Group branch. Clamp so interaction-action ids (>= n_group_actions)
        # stay in bounds when indexing W; their contribution is masked out
        # by is_group regardless of what this computes.
        group_id = action.clamp(max=self.n_group_actions - 1)
        Wa = self.W()[group_id]                                  # (B, D, D)
        z_group = torch.einsum("bij,bj->bi", Wa, z)

        return is_group * z_group + (1.0 - is_group) * z_interact


class WorldModel(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        state_dim: int,
        latent_dim: int,
        n_actions: int,
        n_group_actions: int,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.encoder = RecurrentEncoder(obs_dim, hidden_dim, latent_dim)
        self.transition = GroupStructuredTransition(
            latent_dim, n_actions, n_group_actions, hidden_dim
        )
        self.decoder = Decoder(latent_dim, state_dim, hidden_dim)

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return self.encoder.init_hidden(batch_size, device)

    def teacher_forced(self, obs_seq: torch.Tensor, actions: torch.Tensor) -> List[torch.Tensor]:
        """Sequential teacher forcing over a real trajectory.

        obs_seq[:, t] is the partial observation available just before
        action t. At each step the encoder re-derives z_t from that single
        observation plus the running GRU belief h, then applies the
        transition for actions[:, t]. Returns a list of T predicted states,
        each to be compared against the true state *after* that action.
        """
        B, T = actions.shape
        h = self.init_hidden(B, obs_seq.device)
        preds: List[torch.Tensor] = []
        for t in range(T):
            z, h = self.encoder(obs_seq[:, t], h)
            z_next = self.transition(z, actions[:, t])
            preds.append(self.decoder(z_next))
        return preds

    def rollout(
        self,
        obs0: torch.Tensor,
        actions: torch.Tensor,
        h: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """Closed-loop *latent* imagination: obs0 is encoded once, and every
        subsequent step composes the transition directly on z -- no further
        observations, no decode/re-encode round trip."""
        if h is None:
            h = self.init_hidden(obs0.shape[0], obs0.device)
        z, _ = self.encoder(obs0, h)
        preds: List[torch.Tensor] = []
        for t in range(actions.shape[1]):
            z = self.transition(z, actions[:, t])
            preds.append(self.decoder(z))
        return preds

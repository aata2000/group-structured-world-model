"""Group-structured world model.

Each action `a` is represented by two candidate transformations of the latent
state:

  - a linear operator  W_a = exp(A_a),  A_a = 1/2 (A_raw_a - A_raw_a^T)
    skew-symmetric, so W_a always lies in SO(D) -- an exact rotation, by
    construction, for any A_raw_a;
  - a free residual network f_theta(z, a) for actions that have no group
    structure (e.g. a non-invertible "pick up").

A learned per-(action, latent-dimension) gate decides, independently for each
coordinate, whether that coordinate is updated by the group operator or by
the free branch:

    z_{t+1} = s_a * (W_a z_t) + (1 - s_a) * (z_t + f_theta(z_t, a))

s_a is a straight-through hard Bernoulli sample of sigmoid(gate_logits[a]):
hard 0/1 in the forward pass, but gradients flow through the underlying
probability. Nothing tells the model in advance which actions are group-like
-- it has to discover the split from prediction error alone.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def matrix_exp(A: torch.Tensor, n_squarings: int = 4, n_terms: int = 8) -> torch.Tensor:
    """exp(A) via scale-and-square + Taylor series.

    Pure matmul, so it runs identically on CPU, CUDA, and MPS. For
    skew-symmetric A with ||A||_F <= pi (true for the rotation generators
    here), n_squarings=4 and n_terms=8 give relative error < 1e-6.
    """
    D = A.shape[-1]
    X = A / float(2 ** n_squarings)
    I = torch.eye(D, device=A.device, dtype=A.dtype).expand_as(A)
    term = X.clone()
    result = I + term
    for k in range(2, n_terms + 1):
        term = term @ X / k
        result = result + term
    for _ in range(n_squarings):
        result = result @ result
    return result


class Encoder(nn.Module):
    def __init__(self, obs_dim: int, latent_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, obs_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class GroupStructuredTransition(nn.Module):
    """Per-action transition: a gated mixture of a group operator and a free MLP."""

    def __init__(
        self,
        latent_dim: int,
        n_actions: int,
        free_hidden_dim: int = 64,
        gate_init_logit: float = 0.0,
        w_init_scale: float = 0.01,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_actions = n_actions

        # Lie-algebra parametrization: W_a = exp(skew(A_raw_a)) in SO(D).
        # Storing A_raw (unconstrained) instead of W_a directly means the
        # skew-symmetric projection and matrix exponential guarantee W_a is
        # an exact rotation at every step of training, with no orthogonality
        # penalty needed.
        self.A_raw = nn.Parameter(
            torch.randn(n_actions, latent_dim, latent_dim) * w_init_scale
        )

        # Per-(action, dim) gate logit; sigmoid(gate_logits[a, d]) = P(latent
        # dim d is routed through the group operator for action a).
        gates = torch.full((n_actions, latent_dim), float(gate_init_logit))
        self.gate_logits = nn.Parameter(gates + torch.randn_like(gates) * 0.01)

        # Free branch for non-group actions, conditioned on a one-hot action id.
        self.free_net = nn.Sequential(
            nn.Linear(latent_dim + n_actions, free_hidden_dim),
            nn.ReLU(),
            nn.Linear(free_hidden_dim, free_hidden_dim),
            nn.ReLU(),
            nn.Linear(free_hidden_dim, latent_dim),
        )

    def W(self) -> torch.Tensor:
        """(n_actions, D, D) rotation matrices, recomputed from A_raw each call."""
        A = 0.5 * (self.A_raw - self.A_raw.transpose(-1, -2))
        return matrix_exp(A)

    def gate_probs(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_logits)

    def _sample_gates(self, p: torch.Tensor) -> torch.Tensor:
        """Straight-through hard Bernoulli sample: hard 0/1 forward, d/dp = 1 backward."""
        if self.training:
            hard = (torch.rand_like(p) < p).float()
        else:
            hard = (p > 0.5).float()
        return hard.detach() + (p - p.detach())

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        Wa = self.W()[action]                                   # (B, D, D)
        z_group = torch.einsum("bij,bj->bi", Wa, z)

        onehot = F.one_hot(action, self.n_actions).to(z.dtype)
        z_free = z + self.free_net(torch.cat([z, onehot], dim=-1))

        p = torch.sigmoid(self.gate_logits[action])             # (B, D)
        s = self._sample_gates(p)
        return s * z_group + (1.0 - s) * z_free


class WorldModel(nn.Module):
    def __init__(self, obs_dim: int, latent_dim: int, n_actions: int, hidden_dim: int = 64):
        super().__init__()
        self.encoder = Encoder(obs_dim, latent_dim, hidden_dim)
        self.transition = GroupStructuredTransition(latent_dim, n_actions, hidden_dim)
        self.decoder = Decoder(latent_dim, obs_dim, hidden_dim)

    def step(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        z = self.encoder(obs)
        z_next = self.transition(z, action)
        return self.decoder(z_next)

    def rollout(self, obs0: torch.Tensor, actions: torch.Tensor) -> list[torch.Tensor]:
        """Closed-loop imagination: each step re-encodes the model's own previous
        prediction, never the real environment. `actions` is (B, T) long;
        returns a list of T predicted observations."""
        obs = obs0
        preds = []
        for t in range(actions.shape[1]):
            obs = self.step(obs, actions[:, t])
            preds.append(obs)
        return preds

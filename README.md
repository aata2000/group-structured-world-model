# Group-Structured World Models

A small PyTorch implementation of a recurrent world model where movement
actions are represented by learned operators in SO(D) and interaction
actions go through a separate residual network.

At each step, a recurrent encoder (MLP -> GRUCell -> linear) maps a partial
observation to a latent state `z_t`. Movement actions (`turn_left`,
`turn_right`) then act on `z_t` through a Lie-algebra parametrized rotation
`W_a = exp(skew(A_a))`; interaction actions (`pick_up`, `open_door`) act
through a residual MLP instead, since they are not invertible. Which branch
an action uses is fixed by its id -- this is *not* learned or discovered;
see "What this does and doesn't show" below.

## Demo environment

A tiny, partially observed DoorKey-like environment. True state:
`(orientation, has_key, door_open)`.

- `turn_left` / `turn_right` rotate `orientation` and generate Z4 -- each
  is the other's inverse.
- `pick_up` sets `has_key`; `open_door` sets `door_open`, but only if
  `has_key` is already set. Neither is invertible.

The observation the encoder actually sees at each step is **partial**:
just the current orientation and the action taken on the previous step --
never `has_key` or `door_open` directly. The GRU has to track those from
the action history alone; they're supervised only at the decoder output.
This is what gives the recurrence something to do (see `src/environment.py`).

## What this does and doesn't show

- The group/interaction split (which actions use `W_a` vs. the interaction
  MLP) is **fixed by construction**, mirroring the original design. This
  repo does not attempt to discover that split unsupervised.
- Training is sequential teacher forcing: at every step the encoder
  re-observes the real (partial) trajectory, so `z_t` is always freshly
  grounded before a single transition is applied and decoded.
- `validate.py` also reports a *latent rollout*: the observation is encoded
  once, then every later state comes from composing the transition
  directly on `z`, with no further observations. In practice this collapses
  after the first step in this setup -- teacher forcing never asks the model
  to feed a transition's own output back into another transition call, so
  nothing trains it to be stable under repeated composition. This is a real
  and expected property of the setup, not a bug; run `validate.py` and read
  the numbers rather than assuming either mode works well.
- `validate.py` also prints `||W_turn_left^4 - I||` and
  `||W_turn_right @ W_turn_left - I||`. These check whether the *learned*
  rotation happened to converge to the algebraic relations turning implies
  (order 4, mutual inverses) -- it is not guaranteed, and the printed
  numbers should be read as a diagnostic, not a claimed result.

## Model

For each movement action:

    A_a = 1/2 (A_raw - A_raw^T)
    W_a = exp(A_a)

so `W_a` lies in SO(D) exactly, via `torch.matrix_exp`.

## Usage

Install dependencies:

    pip install -r requirements.txt

Train:

    python train.py

Evaluate:

    python validate.py

See `notebooks/demo.ipynb` for a worked example, including a side-by-side
of teacher-forced vs. latent rollout on the same trajectory.

## Background

This repository is a cleaned-up, standalone version of a class project for
*Deep Learning for Doctoral Students* at the University of Zurich (UZH),
isolating the core idea -- action-conditioned SO(D) operators with a
recurrent belief state over partial observations -- from the full
experimental codebase.

## Language

Python / PyTorch

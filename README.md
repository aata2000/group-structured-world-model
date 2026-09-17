# Group-Structured World Models

A small PyTorch implementation exploring whether a learned world model can
discover algebraic structure in actions directly from trajectories. 

The model learns action-dependent latent transformations. Each action gets
a linear operator in SO(D) plus a free residual network, mixed by a learned
per-(action, latent-dimension) gate -- so, in principle, actions with true
group structure can be routed through the linear operator while others fall
back to the free branch, with no labels telling the model which is which.

## Demo environment

The included synthetic environment contains:

- left/right rotations, which form Z4;
- a non-invertible pick-up action.

**Caveat:** this environment only has 8 discrete states, and `validate.py`
reports the actual learned gate values and operator residuals every run --
in this minimal setup the gates do not reliably binarize into a clean
group/non-group split (a sufficiently expressive decoder can fit a
non-invertible transition through a rotation operator too, when the state
space is this small). The model does learn the dynamics correctly (see the
one-step MSE and rollout accuracy `validate.py` prints), and the
architecture and gating mechanism are real and functional; treat any
particular run's gate/order numbers as a diagnostic, not a guaranteed
discovery result. See `notebooks/demo.ipynb` for a worked example run.

## Model

For each action:

    A_a = 1/2 (A_raw - A_raw^T)
    W_a = exp(A_a)

so W_a lies in SO(D).

A learned per-dimension gate determines whether each latent coordinate is
updated by W_a or by a free residual network.

## Usage

Install dependencies:

    pip install -r requirements.txt

Train:

    python train.py

Evaluate:

    python validate.py

See `notebooks/demo.ipynb` for an interactive walkthrough of a trained model.

## Background

This repository is a cleaned and simplified implementation based on a class project completed for Deep Learning for Doctoral Students at the University of Zurich (UZH).

## Language

Python / PyTorch

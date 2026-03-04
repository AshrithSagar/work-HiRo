"""
Plotting utils
=======
"""
# src/pacer/plotting.py

# pyright: standard

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from typingkit.numpy._typed.helpers import TWO

from pacer.base import (
    PACER,
    Actions,
    Demonstrations,
    DimAction,
    NumBins,
    NumDemos,
    NumPoints,
    PACERBCTrainer,
    PhasesCollection,
    TrustValuesCollection,
)


def plot_trajectories(
    demonstrations: Demonstrations[Any, Any, TWO, DimAction],
    *,
    title: str = "Demonstration trajectories",
) -> None:
    """Plot 2D state trajectories."""
    plt.figure()
    for i, demo in enumerate(demonstrations):
        states = np.array(demo.states)
        plt.plot(states[:, 0], states[:, 1], label=f"Demo {i}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.legend()
    plt.axis("equal")
    plt.tight_layout()
    plt.show()


def plot_phases(
    phases: PhasesCollection[NumDemos, NumPoints],
    *,
    title: str = "Estimated phases",
) -> None:
    """Plot tau_{i,t} for each demonstration."""
    plt.figure(figsize=(8, 4))
    for i, tau in enumerate(phases):
        plt.plot(tau, label=f"Demo {i}", alpha=0.8)
    plt.xlabel("Time index t")
    plt.ylabel(r"Phase $\tau$")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_trust_values(
    trust_values: TrustValuesCollection[NumDemos, NumPoints],
    *,
    title: str = "Trust values",
) -> None:
    """Plot w_{i,t}."""
    plt.figure(figsize=(8, 4))
    for i, w in enumerate(trust_values):
        plt.plot(w, label=f"Demo {i}", alpha=0.8)
    plt.xlabel("Time index t")
    plt.ylabel("Trust w")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_action_comparison(
    original: Actions[NumPoints, DimAction],
    pseudo: Actions[NumPoints, DimAction],
    *,
    title: str = "Original vs Pseudo actions",
) -> None:
    """Compare action vectors over time (per dimension)."""
    _original = np.array(original)
    _pseudo = np.array(pseudo)
    dim = _original.shape[1]

    fig, _axes = plt.subplots(dim, 1, figsize=(8, 3 * dim))
    axes = [_axes] if dim == 1 else _axes

    for d in range(dim):
        axes[d].plot(_original[:, d], label="Original", alpha=0.8)  # type: ignore
        axes[d].plot(_pseudo[:, d], label="Pseudo", alpha=0.8)  # type: ignore
        axes[d].set_ylabel(f"Action dim {d}")  # type: ignore
        axes[d].legend()  # type: ignore

    axes[-1].set_xlabel("Time index t")  # type: ignore
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_ribbon_action_field(
    pacer: PACER[NumBins, NumDemos, NumPoints, TWO, TWO],
    *,
    title: str = "Ribbon Median Action Field",
    scale: float = 1.0,
) -> None:
    """
    Visualize median action vectors at each bin.
    Assumes state_dim == 2 and action_dim == 2.
    """
    xs = []
    ys = []
    us = []
    vs = []

    for bin in pacer.bins:
        token = bin.ribbon_token
        state = token.median_state
        action = token.median_action

        xs.append(state[0])
        ys.append(state[1])
        us.append(action[0])
        vs.append(action[1])

    plt.figure(figsize=(6, 6))
    plt.quiver(xs, ys, us, vs, angles="xy", scale_units="xy", scale=scale)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.axis("equal")
    plt.tight_layout()
    plt.show()


def full_diagnostic(
    trainer: PACERBCTrainer[NumBins, NumDemos, NumPoints, TWO, TWO],
) -> None:
    plot_trajectories(trainer.demonstrations)
    plot_phases(trainer.phase_estimator.estimate_phases())
    plot_trust_values(trainer.trust_values)
    plot_ribbon_action_field(trainer.pacer)
    plot_action_comparison(
        trainer.demonstrations.demos[0].actions,
        trainer.pseudo_labels[0],
        title="Demo 0: Action refinement",
    )

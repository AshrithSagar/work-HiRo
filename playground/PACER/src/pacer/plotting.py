"""
Plotting utils
=======
"""
# src/pacer/plotting.py
# pyright: standard

import matplotlib.pyplot as plt
import numpy as np
from typed_numpy._typed.helpers import TWO

from .base import PACER, Actions, Array1D, BinHandler, Demonstrations, DimAction


def plot_trajectories(
    demonstrations: Demonstrations[TWO, DimAction],
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
    phases: list[Array1D[int]],
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
    trust_values: list[list[float]],
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
    original: Actions[DimAction],
    pseudo: Actions[DimAction],
    *,
    title: str = "Original vs Pseudo actions",
) -> None:
    """Compare action vectors over time (per dimension)."""
    _original = np.array(original)
    _pseudo = np.array(pseudo)
    dim = _original.shape[1]

    fig, axes = plt.subplots(dim, 1, figsize=(8, 3 * dim))
    if dim == 1:
        axes = [axes]

    for d in range(dim):
        axes[d].plot(_original[:, d], label="Original", alpha=0.8)
        axes[d].plot(_pseudo[:, d], label="Pseudo", alpha=0.8)
        axes[d].set_ylabel(f"Action dim {d}")
        axes[d].legend()

    axes[-1].set_xlabel("Time index t")
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_ribbon_action_field(
    binner: BinHandler[TWO, TWO],
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

    for bin in binner.bins:
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


def full_diagnostic(pacer: PACER[TWO, TWO]) -> None:
    plot_trajectories(pacer.demonstrations)
    plot_phases(pacer.phase_estimator.estimate_phases())
    plot_trust_values(pacer.trust_values)
    plot_ribbon_action_field(pacer.binner)
    plot_action_comparison(
        pacer.demonstrations.demos[0].actions,
        pacer.pseudo_labels[0],
        title="Demo 0: Action refinement",
    )

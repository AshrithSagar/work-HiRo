"""
Plotting utils
=======
"""
# src/pacer/plotting.py

# pyright: standard

import matplotlib.pyplot as plt
from typingkit.numpy._typed.helpers import TWO

from pacer.base import Actions, Demonstrations, States, StatesCollection
from pacer.pacer import Bins, TrustValuesCollection
from pacer.phase.base import PhasesCollection
from pacer.typings import DimAction, DimState, NumBins, NumDemos, NumPoints


def plot_trajectories(
    demonstrations: Demonstrations[NumDemos, NumPoints, TWO, TWO],
    *,
    title: str = "Demonstration trajectories",
) -> None:
    """Plot 2D state trajectories."""
    plt.figure()
    for i, demo in enumerate(demonstrations):
        plt.plot(demo.states.coord(0), demo.states.coord(1), label=f"Demo {i}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.legend()
    plt.axis("equal")
    plt.margins(0.05)
    plt.tight_layout()


def plot_states_and_actions(
    demonstrations: Demonstrations[NumDemos, NumPoints, TWO, TWO],
    *,
    title: str = "Demonstrations",
    demo_indices: list[int] | None = None,
    action_scale: float = 1.0,
    action_step: int = 1,
) -> None:
    """Plot states and actions for the 2D case."""
    plt.figure()

    for i, demo in enumerate(demonstrations):
        if demo_indices is not None:
            if i not in demo_indices:
                continue

        xs = demo.states.coord(0)
        ys = demo.states.coord(1)

        (line,) = plt.plot(xs, ys, label=f"Demo {i}", linewidth=2)
        color = line.get_color()
        plt.scatter(xs[0], ys[0], color=color, marker="o", s=15)

        ax = demo.actions.coord(0)
        ay = demo.actions.coord(1)

        xs_q = xs[::action_step]
        ys_q = ys[::action_step]
        ax_q = ax[::action_step]
        ay_q = ay[::action_step]

        plt.quiver(
            xs_q,
            ys_q,
            ax_q,
            ay_q,
            angles="xy",
            scale_units="xy",
            scale=1.0 / action_scale,
            color=color,
            width=0.003,
            headwidth=3,
            headlength=4,
            headaxislength=3.5,
            alpha=0.5,
        )

        # Expand limits to include arrow tips
        plt.scatter(xs_q + ax_q * action_scale, ys_q + ay_q * action_scale, alpha=0)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.legend()
    plt.axis("equal")
    plt.margins(0.05)
    plt.tight_layout()


def plot_states(
    states_collection: StatesCollection[NumDemos, NumPoints, TWO],
    *,
    title: str = "Demonstration trajectories",
) -> None:
    """Plot 2D state trajectories."""
    plt.figure()
    for i, states in enumerate(states_collection):
        plt.plot(states.coord(0), states.coord(1), label=f"Demo {i}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.legend()
    plt.axis("equal")
    plt.margins(0.05)
    plt.tight_layout()


def plot_phases(
    phases: PhasesCollection[NumDemos, NumPoints],
    *,
    title: str = "Estimated phases",
) -> None:
    """Plot tau_{i,t} for each demonstration."""
    plt.figure(figsize=(8, 4))
    for i, tau in phases.items():
        plt.plot(tau, label=f"Demo {i}", alpha=0.8)
    plt.xlabel("Time index t")
    plt.ylabel(r"Phase $\tau$")
    plt.title(title)
    plt.legend()
    plt.tight_layout()


def plot_trust_values(
    trust_values: TrustValuesCollection[NumDemos, NumPoints],
    *,
    title: str = "Trust values",
) -> None:
    """Plot w_{i,t}."""
    plt.figure(figsize=(8, 4))
    for i, w in trust_values.items():
        plt.plot(w, label=f"Demo {i}", alpha=0.8)
    plt.xlabel("Time index t")
    plt.ylabel("Trust w")
    plt.title(title)
    plt.legend()
    plt.tight_layout()


def plot_action_comparison(
    original: Actions[NumPoints, DimAction],
    pseudo: Actions[NumPoints, DimAction],
    *,
    title: str = "Original vs Pseudo actions",
) -> None:
    """Compare action vectors over time (per dimension)."""
    _original = original.numpy()
    _pseudo = pseudo.numpy()
    dim = _original.shape[1]

    fig, _axes = plt.subplots(dim, 1, figsize=(8, 3 * dim))
    axes = [_axes] if dim == 1 else _axes

    for d in range(dim):
        axes[d].plot(_original[:, d], label="Original", alpha=0.8)
        axes[d].plot(_pseudo[:, d], label="Pseudo", alpha=0.8)
        axes[d].set_ylabel(f"Action dim {d}")
        axes[d].legend()

    axes[-1].set_xlabel("Time index t")
    fig.suptitle(title)
    plt.tight_layout()


def plot_state_comparison(
    original: States[NumPoints, DimState],
    pseudo: States[NumPoints, DimState],
    *,
    title: str = "Original vs Pseudo states",
) -> None:
    """Compare state vectors over time (per dimension)."""
    _original = original.numpy()
    _pseudo = pseudo.numpy()
    dim = _original.shape[1]

    fig, _axes = plt.subplots(dim, 1, figsize=(8, 3 * dim))
    axes = [_axes] if dim == 1 else _axes

    for d in range(dim):
        axes[d].plot(_original[:, d], label="Original", alpha=0.8)
        axes[d].plot(_pseudo[:, d], label="Pseudo", alpha=0.8)
        axes[d].set_ylabel(f"State dim {d}")
        axes[d].legend()

    axes[-1].set_xlabel("Time index t")
    fig.suptitle(title)
    plt.tight_layout()


def plot_ribbon_action_field(
    bins: Bins[NumBins, NumDemos, NumPoints, TWO, TWO],
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

    for bin in bins:
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

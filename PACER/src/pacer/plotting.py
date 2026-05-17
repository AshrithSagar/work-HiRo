"""
Plotting utils
=======
"""
# src/pacer/plotting.py

# pyright: reportUnknownMemberType = false

## ── Imports ──────────────────────────────────────────────────────────────────

from dataclasses import KW_ONLY, dataclass, field
from pathlib import Path
from typing import Generic

import matplotlib.pyplot as plt
import numpy as np
from typingkit.numpy._typed.helpers import TWO

from pacer import console
from pacer.base import (
    Actions,
    ActionsCollection,
    Demonstrations,
    States,
    StatesCollection,
)
from pacer.pacer import Bins, PACERResult, TrustValuesCollection
from pacer.phase.base import PhasesCollection
from pacer.typings import DemoIndex, DimAction, DimState, NumBins, NumDemos, NumPoints

## ── Plotting ─────────────────────────────────────────────────────────────────


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
    demo_indices: DemoIndex | list[DemoIndex] | None = None,
    action_scale: float = 1.0,
    action_step: int = 1,
) -> None:
    """Plot states and actions for the 2D case."""
    plt.figure()

    for i, demo in enumerate(demonstrations):
        match demo_indices:
            case DemoIndex():
                if i != demo_indices:
                    continue
            case list():
                if i not in demo_indices:
                    continue
            case None:
                pass

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


def plot_states_before_after(
    original: StatesCollection[NumDemos, NumPoints, TWO],
    pseudo: StatesCollection[NumDemos, NumPoints, TWO],
    *,
    title: str = "States comparision",
) -> None:
    """Overlay original and refined trajectories."""
    plt.figure(figsize=(7, 7))
    for i, (orig, new) in enumerate(zip(original, pseudo)):
        plt.plot(orig.coord(0), orig.coord(1), "--", alpha=0.5, label=f"Original {i}")
        plt.plot(new.coord(0), new.coord(1), label=f"Refined {i}")
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


def plot_action_correction_magnitude(
    original: ActionsCollection[NumDemos, NumPoints, TWO],
    pseudo: ActionsCollection[NumDemos, NumPoints, TWO],
    *,
    title: str = "Action Correction Magnitude",
) -> None:
    """Plot ||a_pseudo - a_original|| over time."""
    plt.figure(figsize=(8, 4))
    for i, (orig, new) in enumerate(zip(original, pseudo)):
        delta = np.linalg.norm(new.numpy() - orig.numpy(), axis=1)
        plt.plot(delta, label=f"Demo {i}")
    plt.xlabel("Time index")
    plt.ylabel("Correction magnitude")
    plt.title(title)
    plt.legend()
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


## ── PACER Visualisation ──────────────────────────────────────────────────────


@dataclass
class PACERVisualisationConfig:
    show: bool = True
    save_dir: Path | str | None = None

    trajectories: bool = True
    phases: bool = True
    trust_values: bool = True

    states_before_after: bool = True
    action_comparison: bool = True
    state_comparison: bool = True

    ribbon_action_field: bool = True
    action_correction_magnitude: bool = True


@dataclass
class PACERVisualiser(Generic[NumBins, NumDemos, NumPoints]):
    demonstrations: Demonstrations[NumDemos, NumPoints, TWO, TWO]
    pacer_result: PACERResult[NumBins, NumDemos, NumPoints, TWO, TWO]
    _: KW_ONLY
    config: PACERVisualisationConfig = field(default_factory=PACERVisualisationConfig)

    def render(self) -> None:
        if self.config.trajectories:
            plot_trajectories(self.demonstrations)

        if self.config.phases:
            plot_phases(self.pacer_result.phases)

        if self.config.trust_values:
            plot_trust_values(
                self.pacer_result.action_trust_values,
                title="Action trust values",
            )

            if self.pacer_result.state_trust_values is not None:
                plot_trust_values(
                    self.pacer_result.state_trust_values,
                    title="State trust values",
                )

        if self.config.ribbon_action_field:
            plot_ribbon_action_field(self.pacer_result.bins)

        if self.config.action_comparison:
            plot_action_comparison(
                self.demonstrations[0].actions,
                self.pacer_result.pseudo_labels.actions[0],
                title="Demo 0: Action refinement",
            )

        if self.config.states_before_after:
            if self.pacer_result.pseudo_labels.states is not None:
                plot_states_before_after(
                    self.demonstrations.states,
                    self.pacer_result.pseudo_labels.states,
                )

        if self.config.state_comparison:
            if self.pacer_result.pseudo_labels.states is not None:
                plot_state_comparison(
                    self.demonstrations[0].states,
                    self.pacer_result.pseudo_labels.states[0],
                    title="Demo 0: State refinement",
                )

        if self.config.action_correction_magnitude:
            plot_action_correction_magnitude(
                self.demonstrations.actions,
                self.pacer_result.pseudo_labels.actions,
            )

        if self.config.save_dir is not None:
            save_dir = Path(self.config.save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

            fignums = plt.get_fignums()
            with console.status("Saving figures...", spinner="dots"):
                for i in fignums:
                    fig = plt.figure(i)
                    fig.savefig(
                        save_dir / f"figure_{i}.png",
                        dpi=300,
                        bbox_inches="tight",
                    )
            console.print(
                f"[grey53]Saved {len(fignums)} figures to[/grey53] "
                + f"[italic grey69]{save_dir.absolute()}[/italic grey69]"
            )

        if self.config.show:
            plt.show()


## ─────────────────────────────────────────────────────────────────────────────

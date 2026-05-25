"""
(Legacy) Plotting utils
=======
"""
# src/pacer/plotting/legacy.py

# pyright: reportUnknownMemberType = false

## ── Imports ──────────────────────────────────────────────────────────────────

from collections.abc import Iterable, Mapping
from dataclasses import KW_ONLY, dataclass, field
from pathlib import Path
from typing import Literal

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import (  # type: ignore[import-untyped]  # pyright: ignore[reportMissingTypeStubs]
    Axes3D,
)
from mpl_toolkits.mplot3d.art3d import (  # type: ignore[import-untyped]  # pyright: ignore[reportMissingTypeStubs]
    Line3DCollection,
)
from typingkit.core import RuntimeGeneric
from typingkit.numpy._typed.helpers import TWO

from pacer import console
from pacer.base import (
    Actions,
    ActionsCollection,
    Demonstrations,
    StateActionPairs,
    States,
    StatesCollection,
)
from pacer.pacer import PACERResult
from pacer.pacer.analysis import (
    CorrectionMagnitudeAnalyser,
    CorrectionMagnitudeAnalysis,
    ResidualAnalyser,
    SmoothnessAnalyser,
    SmoothnessAnalysis,
)
from pacer.pacer.base import (
    MetricSeries,
    MetricValue,
    Residual,
    ResidualsCollection,
    TrustValue,
    TrustValuesCollection,
)
from pacer.pacer.binning import Bins
from pacer.phase import PhasesCollection
from pacer.typings import (
    DemoIndex,
    DimAction,
    DimState,
    NumBins,
    NumDemos,
    NumPoints,
    Vector,
    npDType,
)

## ── Plotting ─────────────────────────────────────────────────────────────────


def ensure_fig_ax(
    *,
    fig: Figure | None = None,
    ax: Axes | None = None,
    figsize: tuple[float, float] = (6, 6),
) -> tuple[Figure, Axes]:
    """Ensure figure + axis exist."""
    if fig is not None and ax is not None:
        return fig, ax
    created_fig, created_ax = plt.subplots(figsize=figsize)
    return created_fig, created_ax


def ensure_fig_axes(
    *,
    fig: Figure | None = None,
    axes: Iterable[Axes] | np.ndarray | None = None,
    nrows: int = 1,
    figsize: tuple[float, float] = (8, 4),
) -> tuple[Figure, list[Axes]]:
    """Ensure figure + axes exist."""
    if fig is not None and axes is not None:
        return fig, list(axes)
    created_fig, created_axes = plt.subplots(nrows, 1, figsize=figsize)
    if nrows == 1:
        return created_fig, [created_axes]
    return created_fig, list(created_axes)


# ──────────────────────────────────────────────────────────────────────────────


def plot_trajectories(
    demonstrations: Demonstrations[NumDemos, NumPoints, TWO, TWO],
    *,
    fig: Figure | None = None,
    ax: Axes | None = None,
    title: str = "Demonstration trajectories",
) -> None:
    """Plot 2D state trajectories."""
    fig, ax = ensure_fig_ax(fig=fig, ax=ax, figsize=(6, 6))
    for i, demo in enumerate(demonstrations):
        ax.plot(demo.states.coord(0), demo.states.coord(1), label=f"Demo {i}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.legend()
    ax.axis("equal")
    ax.margins(0.05)
    fig.tight_layout()


def plot_states_and_actions(
    demonstrations: Demonstrations[NumDemos, NumPoints, TWO, TWO],
    *,
    fig: Figure | None = None,
    ax: Axes | None = None,
    title: str = "Demonstrations",
    demo_indices: DemoIndex | list[DemoIndex] | None = None,
    action_scale: float = 1.0,
    action_step: int = 1,
) -> None:
    """Plot states and actions for the 2D case."""
    fig, ax = ensure_fig_ax(fig=fig, ax=ax, figsize=(6, 6))

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

        (line,) = ax.plot(xs, ys, label=f"Demo {i}", linewidth=2)
        color = line.get_color()
        ax.scatter(xs[0], ys[0], color=color, marker="o", s=15)

        ax_ = demo.actions.coord(0)
        ay_ = demo.actions.coord(1)

        xs_q = xs[::action_step]
        ys_q = ys[::action_step]
        ax_q = ax_[::action_step]
        ay_q = ay_[::action_step]

        ax.quiver(
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
        ax.scatter(xs_q + ax_q * action_scale, ys_q + ay_q * action_scale, alpha=0)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.legend()
    ax.axis("equal")
    ax.margins(0.05)
    fig.tight_layout()


def plot_states(
    states_collection: StatesCollection[NumDemos, NumPoints, TWO],
    *,
    fig: Figure | None = None,
    ax: Axes | None = None,
    title: str = "Demonstration trajectories",
) -> None:
    """Plot 2D state trajectories."""
    fig, ax = ensure_fig_ax(fig=fig, ax=ax, figsize=(6, 6))
    for i, states in enumerate(states_collection):
        ax.plot(states.coord(0), states.coord(1), label=f"Demo {i}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.legend()
    ax.axis("equal")
    ax.margins(0.05)
    fig.tight_layout()


def plot_states_before_after(
    original: StatesCollection[NumDemos, NumPoints, TWO],
    pseudo: StatesCollection[NumDemos, NumPoints, TWO],
    *,
    fig: Figure | None = None,
    ax: Axes | None = None,
    title: str = "States comparision",
) -> None:
    """Overlay original and refined trajectories."""
    fig, ax = ensure_fig_ax(fig=fig, ax=ax, figsize=(7, 7))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (orig, new) in enumerate(zip(original, pseudo)):
        color = colors[i % len(colors)]
        ax.plot(
            orig.coord(0),
            orig.coord(1),
            "--",
            alpha=0.5,
            color=color,
            label=f"Original {i}",
        )
        ax.plot(new.coord(0), new.coord(1), color=color, label=f"Refined {i}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.legend()
    ax.axis("equal")
    ax.margins(0.05)
    fig.tight_layout()


def plot_actions_before_after(
    states: States[NumPoints, TWO],
    original_actions: Actions[NumPoints, TWO],
    pseudo_actions: Actions[NumPoints, TWO],
    *,
    fig: Figure | None = None,
    ax: Axes | None = None,
    title: str = "Actions Before vs After",
    step: int = 1,
    action_scale: float = 1.0,
    show_corrections: bool = False,
    original_color: str = "tab:blue",
    pseudo_color: str = "tab:orange",
    trajectory_color: str = "grey",
) -> None:
    """Overlay original and pseudo actions on top of trajectories."""
    fig, ax = ensure_fig_ax(fig=fig, ax=ax, figsize=(7, 7))

    xs = states.coord(0)
    ys = states.coord(1)
    ox = original_actions.coord(0)
    oy = original_actions.coord(1)
    px = pseudo_actions.coord(0)
    py = pseudo_actions.coord(1)

    # Subsample
    xs_q = xs[::step]
    ys_q = ys[::step]
    ox_q = ox[::step]
    oy_q = oy[::step]
    px_q = px[::step]
    py_q = py[::step]

    # Background trajectories
    ax.plot(xs, ys, color=trajectory_color, alpha=0.3, linewidth=2, zorder=0)

    # Original actions
    ax.quiver(
        xs_q,
        ys_q,
        ox_q,
        oy_q,
        angles="xy",
        scale_units="xy",
        scale=1.0 / action_scale,
        color=original_color,
        alpha=0.65,
        width=0.003,
        label="Original",
        zorder=2,
    )

    # Pseudo actions
    ax.quiver(
        xs_q,
        ys_q,
        px_q,
        py_q,
        angles="xy",
        scale_units="xy",
        scale=1.0 / action_scale,
        color=pseudo_color,
        alpha=0.65,
        width=0.003,
        label="Pseudo",
        zorder=3,
    )

    # Correction vectors
    if show_corrections:
        dx = px_q - ox_q
        dy = py_q - oy_q
        ax.quiver(
            xs_q,
            ys_q,
            dx,
            dy,
            angles="xy",
            scale_units="xy",
            scale=1.0 / action_scale,
            color="red",
            alpha=0.5,
            width=0.002,
            label="Correction",
            zorder=4,
        )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.axis("equal")
    ax.margins(0.05)
    ax.legend()
    fig.tight_layout()


def plot_phases(
    phases: PhasesCollection[NumDemos, NumPoints],
    *,
    fig: Figure | None = None,
    ax: Axes | None = None,
    title: str = "Estimated phases",
) -> None:
    """Plot tau_{i,t} for each demonstration."""
    fig, ax = ensure_fig_ax(fig=fig, ax=ax, figsize=(8, 4))
    for i, tau in phases.items():
        ax.plot(tau, label=f"Demo {i}", alpha=0.8)
    ax.set_xlabel("Time index t")
    ax.set_ylabel(r"Phase $\tau$")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()


def plot_trust_values(
    trust_values: TrustValuesCollection[NumDemos, NumPoints],
    *,
    fig: Figure | None = None,
    ax: Axes | None = None,
    title: str = "Trust values",
) -> None:
    """Plot w_{i,t}."""
    fig, ax = ensure_fig_ax(fig=fig, ax=ax, figsize=(8, 4))
    for i, w in trust_values.items():
        ax.plot(w, label=f"Demo {i}", alpha=0.8)
    ax.set_xlabel("Time index t")
    ax.set_ylabel("Trust w")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()


def plot_action_comparison(
    original: Actions[NumPoints, DimAction],
    pseudo: Actions[NumPoints, DimAction],
    *,
    fig: Figure | None = None,
    axes: Iterable[Axes] | np.ndarray | None = None,
    title: str = "Original vs Pseudo actions",
) -> None:
    """Compare action vectors over time (per dimension)."""
    _original = original.numpy()
    _pseudo = pseudo.numpy()
    dim = _original.shape[1]

    fig, axes = ensure_fig_axes(fig=fig, axes=axes, nrows=dim, figsize=(8, 3 * dim))
    for d in range(dim):
        axes[d].plot(_original[:, d], label="Original", alpha=0.8)
        axes[d].plot(_pseudo[:, d], label="Pseudo", alpha=0.8)
        axes[d].set_ylabel(f"Action dim {d}")
        axes[d].legend()

    axes[-1].set_xlabel("Time index t")
    fig.suptitle(title)
    fig.tight_layout()


def plot_state_comparison(
    original: States[NumPoints, DimState],
    pseudo: States[NumPoints, DimState],
    *,
    fig: Figure | None = None,
    axes: Iterable[Axes] | np.ndarray | None = None,
    title: str = "Original vs Pseudo states",
) -> None:
    """Compare state vectors over time (per dimension)."""
    _original = original.numpy()
    _pseudo = pseudo.numpy()
    dim = _original.shape[1]

    fig, axes = ensure_fig_axes(fig=fig, axes=axes, nrows=dim, figsize=(8, 3 * dim))
    for d in range(dim):
        axes[d].plot(_original[:, d], label="Original", alpha=0.8)
        axes[d].plot(_pseudo[:, d], label="Pseudo", alpha=0.8)
        axes[d].set_ylabel(f"State dim {d}")
        axes[d].legend()

    axes[-1].set_xlabel("Time index t")
    fig.suptitle(title)
    fig.tight_layout()


def plot_action_correction_magnitude(
    original: ActionsCollection[NumDemos, NumPoints, TWO],
    pseudo: ActionsCollection[NumDemos, NumPoints, TWO],
    *,
    fig: Figure | None = None,
    ax: Axes | None = None,
    title: str = "Action Correction Magnitude",
) -> None:
    """Plot ||a_pseudo - a_original|| over time."""
    fig, ax = ensure_fig_ax(fig=fig, ax=ax, figsize=(8, 4))
    for i, (orig, new) in enumerate(zip(original, pseudo)):
        delta = np.linalg.norm(new.numpy() - orig.numpy(), axis=1)
        ax.plot(delta, label=f"Demo {i}")
    ax.set_xlabel("Time index")
    ax.set_ylabel("Correction magnitude")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()


def plot_ribbon_action_field(
    bins: Bins[NumBins, NumDemos, NumPoints, TWO, TWO],
    *,
    fig: Figure | None = None,
    ax: Axes | None = None,
    title: str = "Ribbon Reference Action Field",
    scale: float = 1.0,
    pad: float = 0.05,
) -> None:
    """
    Visualize reference action vectors at each bin.
    Assumes state_dim == 2 and action_dim == 2.
    """
    _xs: list[npDType] = []
    _ys: list[npDType] = []
    _us: list[npDType] = []
    _vs: list[npDType] = []

    for bin in bins:
        token = bin.ribbon_token
        state = token.state_anchor
        action = token.action_anchor

        _xs.append(state[0])
        _ys.append(state[1])
        _us.append(action[0])
        _vs.append(action[1])

    xs = np.asarray(_xs)
    ys = np.asarray(_ys)
    us = np.asarray(_us)
    vs = np.asarray(_vs)

    fig, ax = ensure_fig_ax(fig=fig, ax=ax, figsize=(6, 6))
    ax.quiver(xs, ys, us, vs, angles="xy", scale_units="xy", scale=scale)

    # Include both arrow starts and ends
    x_all = np.concatenate([xs, xs + us / scale])
    y_all = np.concatenate([ys, ys + vs / scale])

    # Padding
    x_range = x_all.max() - x_all.min()
    y_range = y_all.max() - y_all.min()

    ax.set_xlim(x_all.min() - pad * x_range, x_all.max() + pad * x_range)
    ax.set_ylim(y_all.min() - pad * y_range, y_all.max() + pad * y_range)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()


def plot_residual_distribution(
    residuals_collection: ResidualsCollection[NumDemos, NumPoints],
    *,
    fig: Figure | None = None,
    ax: Axes | None = None,
    title: str = "Residual distribution",
    bins: int = 32,
) -> None:
    """Plot histogram of correction residuals."""
    fig, ax = ensure_fig_ax(fig=fig, ax=ax, figsize=(7, 4))
    flat = [residual for residuals in residuals_collection for residual in residuals]
    ax.hist(flat, bins=bins)
    ax.set_xlabel("Residual magnitude")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    fig.tight_layout()


def plot_trust_heatmap(
    trust_values: TrustValuesCollection[NumDemos, NumPoints],
    *,
    fig: Figure | None = None,
    ax: Axes | None = None,
    title: str = "Trust value heatmap",
) -> None:
    """Visualize trust values across demonstrations and time."""
    fig, ax = ensure_fig_ax(fig=fig, ax=ax, figsize=(10, 4))
    matrix = np.asarray(
        [np.asarray(values, dtype=npDType) for values in trust_values.values()]
    )
    im = ax.imshow(matrix, aspect="auto")
    fig.colorbar(im, ax=ax, label="Trust")
    ax.set_xlabel("Time index")
    ax.set_ylabel("Demo index")
    ax.set_title(title)
    fig.tight_layout()


def plot_bin_occupancy(
    bins: Bins[NumBins, NumDemos, NumPoints, TWO, TWO],
    *,
    fig: Figure | None = None,
    ax: Axes | None = None,
    title: str = "Phase bin occupancy",
) -> None:
    """Plot number of samples assigned to each phase bin."""
    fig, ax = ensure_fig_ax(fig=fig, ax=ax, figsize=(8, 4))
    occupancies: list[int] = []
    for bin in bins:
        count = sum(len(samples) for samples in bin.samples_collection.values())
        occupancies.append(count)
    ax.plot(occupancies)
    ax.set_xlabel("Bin index")
    ax.set_ylabel("Sample count")
    ax.set_title(title)
    fig.tight_layout()


def plot_trust_vs_correction(
    trust_values: TrustValuesCollection[NumDemos, NumPoints],
    correction_analysis: CorrectionMagnitudeAnalysis[NumDemos, NumPoints, TWO, TWO],
    *,
    fig: Figure | None = None,
    ax: Axes | None = None,
    title: str = "Trust vs correction magnitude",
) -> None:
    """Scatter plot comparing trust and correction size."""
    fig, ax = ensure_fig_ax(fig=fig, ax=ax, figsize=(6, 6))
    xs: list[TrustValue] = []
    ys: list[MetricValue] = []
    for demo_idx, trust_series in trust_values.items():
        magnitudes = correction_analysis.magnitudes[demo_idx]
        for trust, magnitude in zip(trust_series, magnitudes):
            xs.append(trust)
            ys.append(magnitude)
    ax.scatter(xs, ys, alpha=0.5)
    ax.set_xlabel("Trust value")
    ax.set_ylabel("Correction magnitude")
    ax.set_title(title)
    fig.tight_layout()


def plot_ribbon_statistics(
    bins: Bins[NumBins, NumDemos, NumPoints, TWO, TWO],
    *,
    fig: Figure | None = None,
    ax: Axes | None = None,
    title: str = "Ribbon statistics",
) -> None:
    """Plot robust ribbon statistics over phase bins."""
    fig, ax = ensure_fig_ax(fig=fig, ax=ax, figsize=(10, 5))
    strengths: list[npDType] = []
    variability: list[Residual] = []
    for bin in bins:
        token = bin.ribbon_token
        strengths.append(token.action_strength)
        variability.append(token.action_residual_scale)
    ax.plot(strengths, label="(Median) action strength")
    ax.plot(variability, label="Action residual scale (MAD residual)")
    ax.set_xlabel("Bin index")
    ax.set_ylabel("Magnitude")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()


def plot_phase_velocity(
    phases: PhasesCollection[NumDemos, NumPoints],
    *,
    fig: Figure | None = None,
    ax: Axes | None = None,
    title: str = "Phase velocity",
) -> None:
    """Plot d(tau)/dt for demonstrations."""
    fig, ax = ensure_fig_ax(fig=fig, ax=ax, figsize=(8, 4))
    for i, phase in phases.items():
        velocity = np.diff(phase)
        ax.plot(velocity, label=f"Demo {i}")
    ax.set_xlabel("Time index")
    ax.set_ylabel("Phase velocity")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()


def plot_smoothness_comparison(
    original: SmoothnessAnalysis[NumDemos, NumPoints],
    pseudo: SmoothnessAnalysis[NumDemos, NumPoints],
    *,
    fig: Figure | None = None,
    ax: Axes | None = None,
    title: str = "Trajectory smoothness comparison",
) -> None:
    """Compare smoothness before and after PACER."""
    fig, ax = ensure_fig_ax(fig=fig, ax=ax, figsize=(7, 4))
    xs = np.arange(len(original.smoothness_scores))
    ax.plot(xs, original.smoothness_scores, label="Original")
    ax.plot(xs, pseudo.smoothness_scores, label="Pseudo")
    ax.set_xlabel("Demo index")
    ax.set_ylabel("Jerk energy")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()


def plot_trust_colored_trajectory(
    states: States[NumPoints, TWO],
    trust_values: MetricSeries[NumPoints] | Vector[NumPoints],
    *,
    fig: Figure | None = None,
    ax: Axes | None = None,
    title: str = "Trust-Colored Trajectory",
    cmap: str = "viridis",
    linewidth: float = 2.5,
    point_size: float = 28,
) -> None:
    """
    Plot 2D trajectories where each sample is colored by trust value.
    High trust -> bright/clean; Low trust  -> dark/suspicious
    """
    xs = states.coord(0)
    ys = states.coord(1)
    trust = np.asarray(trust_values, dtype=npDType)
    fig, ax = ensure_fig_ax(fig=fig, ax=ax, figsize=(7, 7))
    scatter = ax.scatter(xs, ys, c=trust, cmap=cmap, s=point_size, zorder=3)
    ax.plot(xs, ys, color="grey", alpha=0.35, linewidth=linewidth, zorder=1)
    fig.colorbar(scatter, label="Trust")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.axis("equal")
    ax.margins(0.05)
    fig.tight_layout()


def plot_trust_colored_action_field(
    states: States[NumPoints, TWO],
    actions: Actions[NumPoints, TWO],
    trust_values: MetricSeries[NumPoints] | Vector[NumPoints],
    *,
    fig: Figure | None = None,
    ax: Axes | None = None,
    title: str = "Trust-Colored Action Field",
    cmap: str = "coolwarm",
    action_scale: float = 1.0,
    step: int = 1,
) -> None:
    """Plot action vectors colored by trust value."""
    xs = states.coord(0)[::step]
    ys = states.coord(1)[::step]
    ax_ = actions.coord(0)[::step]
    ay_ = actions.coord(1)[::step]
    trust = np.asarray(trust_values, dtype=npDType)[::step]

    fig, ax = ensure_fig_ax(fig=fig, ax=ax, figsize=(7, 7))
    quiver = ax.quiver(
        xs,
        ys,
        ax_,
        ay_,
        trust,
        cmap=cmap,
        angles="xy",
        scale_units="xy",
        scale=1.0 / action_scale,
        width=0.004,
        alpha=0.9,
    )
    fig.colorbar(quiver, label="Trust")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.axis("equal")
    ax.margins(0.05)
    fig.tight_layout()


def plot_action_correction_vectors(
    states: States[NumPoints, TWO],
    original_actions: Actions[NumPoints, TWO],
    pseudo_actions: Actions[NumPoints, TWO],
    *,
    fig: Figure | None = None,
    ax: Axes | None = None,
    title: str = "Action Correction Vectors",
    correction_scale: float = 1.0,
    threshold: float = 1e-3,
    step: int = 1,
) -> None:
    """
    Visualize PACER corrections:
        `delta = pseudo - original`
    Only corrections above threshold are drawn.
    """
    original = original_actions.numpy()
    pseudo = pseudo_actions.numpy()
    delta = pseudo - original
    magnitude = np.linalg.norm(delta, axis=1)
    xs = states.coord(0)
    ys = states.coord(1)
    mask = magnitude > threshold
    xs_q: Vector[int] = xs[mask][::step]  # pyright: ignore[reportUnknownVariableType]
    ys_q: Vector[int] = ys[mask][::step]  # pyright: ignore[reportUnknownVariableType]
    dx = delta[:, 0][mask][::step]
    dy = delta[:, 1][mask][::step]

    fig, ax = ensure_fig_ax(fig=fig, ax=ax, figsize=(7, 7))
    ax.plot(xs, ys, alpha=0.25)
    ax.quiver(
        xs_q,
        ys_q,
        dx,
        dy,
        magnitude[mask][::step],
        cmap="inferno",
        angles="xy",
        scale_units="xy",
        scale=1.0 / correction_scale,
        width=0.004,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.axis("equal")
    fig.tight_layout()


def plot_phase_aligned_trajectories(
    states_collection: StatesCollection[NumDemos, NumPoints, TWO],
    phases: PhasesCollection[NumDemos, NumPoints],
    *,
    fig: Figure | None = None,
    ax: Axes | None = None,
    title: str = "Phase-Aligned Trajectories",
    alpha: float = 0.8,
) -> None:
    """
    Plot trajectories parameterized by phase instead of time.
    Useful for visualising PACER alignment quality.
    """
    fig, ax = ensure_fig_ax(fig=fig, ax=ax, figsize=(7, 7))
    scatter = None
    for i, (states, tau) in enumerate(zip(states_collection, phases.values())):
        xs = states.coord(0)
        ys = states.coord(1)
        scatter = ax.scatter(
            xs,
            ys,
            c=np.asarray(tau, dtype=npDType),
            cmap="plasma",
            s=14,
            alpha=alpha,
            label=f"Demo {i}",
        )
        ax.plot(xs, ys, alpha=0.2)
    if scatter is not None:
        fig.colorbar(scatter, label=r"Phase $\tau$")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.axis("equal")
    fig.tight_layout()


def plot_ribbon_corridor(
    bins: Bins[NumBins, NumDemos, NumPoints, TWO, TWO],
    *,
    fig: Figure | None = None,
    ax: Axes | None = None,
    title: str = "Ribbon Consensus Corridor",
    variability_scale: float = 1.0,
) -> None:
    """
    Plot ribbon reference trajectory with variability corridor.
    Corridor radius is derived from MAD residual.
    """
    ref_states = bins.consensus_trajectory.states()
    xs, ys = ref_states.coord(0), ref_states.coord(1)
    variability = [bin.ribbon_token.action_residual_scale for bin in bins]
    var = variability_scale * np.asarray(variability, dtype=npDType)

    fig, ax = ensure_fig_ax(fig=fig, ax=ax, figsize=(7, 7))
    ax.plot(xs, ys, linewidth=3, label="Ribbon anchor")
    ax.fill_between(xs, ys - var, ys + var, alpha=0.2, label="Residual corridor")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.axis("equal")
    ax.legend()
    fig.tight_layout()


def plot_residual_vs_phase(
    phases: PhasesCollection[NumDemos, NumPoints],
    residuals: ResidualsCollection[NumDemos, NumPoints],
    *,
    fig: Figure | None = None,
    ax: Axes | None = None,
    title: str = "Residual vs Phase",
    alpha: float = 0.5,
) -> None:
    """
    Scatter plot of residual magnitude against phase.
    Useful for identifying difficult task regions.
    """
    xs: list[npDType] = []
    ys: list[Residual] = []
    for i in phases.keys():
        tau = phases[i]
        residual_series = residuals[i]
        xs.extend(tau)
        ys.extend(residual_series)

    fig, ax = ensure_fig_ax(fig=fig, ax=ax, figsize=(8, 4))
    ax.scatter(xs, ys, alpha=alpha)
    ax.set_xlabel(r"Phase $\tau$")
    ax.set_ylabel("Residual magnitude")
    ax.set_title(title)
    fig.tight_layout()


def plot_action_angle_deviation(
    actions: ActionsCollection[NumDemos, NumPoints, TWO],
    bins: Bins[NumBins, NumDemos, NumPoints, TWO, TWO],
    phases: PhasesCollection[NumDemos, NumPoints],
    *,
    fig: Figure | None = None,
    ax: Axes | None = None,
    title: str = "Action Angle Deviation",
) -> None:
    """
    Plot angular deviation from ribbon reference action.
    Measures directional disagreement.
    """
    xs: list[npDType] = []
    ys: list[npDType] = []
    bin_list = list(bins)
    for i, demo_actions in enumerate(actions):
        tau = phases[i]
        for t in range(len(demo_actions)):
            action = demo_actions[t]
            phase = tau[t]
            bin_index = min(int(phase * len(bin_list)), len(bin_list) - 1)
            reference = bin_list[bin_index].ribbon_token.action_anchor
            numerator = np.dot(action, reference)
            denominator = np.linalg.norm(action) * np.linalg.norm(reference) + 1e-8
            cosine = np.clip(numerator / denominator, -1.0, 1.0)
            angle = np.arccos(cosine)
            xs.append(phase)
            ys.append(angle)

    fig, ax = ensure_fig_ax(fig=fig, ax=ax, figsize=(8, 4))
    ax.scatter(xs, ys, alpha=0.4)
    ax.set_xlabel(r"Phase $\tau$")
    ax.set_ylabel("Angular deviation [rad]")
    ax.set_title(title)
    fig.tight_layout()


# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class StackedTrajectoryStyle:
    mode: Literal["isometric", "3d"] = "isometric"

    # spacing between demonstrations
    spacing: float = 1.0

    # isometric offsets
    offset_x: float = 0.4
    offset_y: float = 0.4

    # rendering
    cmap: str = "viridis"
    linewidth: float = 3.0
    alpha: float = 0.95

    # trust emphasis
    trust_width_scale: float = 2.0
    trust_alpha_floor: float = 0.15

    # misc
    show_phase_markers: bool = True
    phase_marker_count: int = 5

    # ribbon overlay
    show_reference: bool = True
    reference_linewidth: float = 5.0


def _make_segments_2d(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    points = np.column_stack([xs, ys])
    return np.stack([points[:-1], points[1:]], axis=1)


def _make_segments_3d(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> np.ndarray:
    points = np.column_stack([xs, ys, zs])
    return np.stack([points[:-1], points[1:]], axis=1)


def plot_stacked_trust_colored_trajectories(
    states_collection: StatesCollection[NumDemos, NumPoints, TWO],
    trust_values_collection: Mapping[int, MetricSeries[NumPoints]]
    | Iterable[MetricSeries[NumPoints]],
    consensus_trajectory: StateActionPairs[NumBins, DimState, DimAction],
    *,
    fig: Figure | None = None,
    ax: Axes3D | None = None,
    title: str = "Stacked Trust-Colored Trajectories",
    style: StackedTrajectoryStyle | None = None,
) -> None:
    """
    PACER-style comparative trust visualization.

    Each demonstration is rendered as:
        - an offset trajectory layer
        - trust-colored segments
        - optional phase markers

    Modes:
        - "isometric" : fake depth using planar offsets
        - "3d"        : actual 3D stacked planes
    """

    # Setup
    if style is None:
        style = StackedTrajectoryStyle()
    if style.mode == "3d":
        if fig is None or ax is None:
            fig = plt.figure(figsize=(9, 8))
            ax = fig.add_subplot(111, projection="3d")
    else:
        fig, ax = ensure_fig_ax(fig=fig, ax=ax, figsize=(9, 8))
    assert ax is not None

    trust_list: list[MetricSeries[NumPoints]] = (
        list(trust_values_collection.values())
        if isinstance(trust_values_collection, Mapping)
        else list(trust_values_collection)
    )

    # Global ribbon reference
    if style.show_reference:
        ref_states = consensus_trajectory.states()
        xs_ref, ys_ref = ref_states.coord(0), ref_states.coord(1)

        if style.mode == "3d":
            ax.plot(
                np.zeros(len(xs_ref)),
                xs_ref,
                ys_ref,
                color="red",
                linewidth=style.reference_linewidth,
                alpha=0.5,
                label="Ribbon reference",
            )
        else:
            ax.plot(
                xs_ref,
                ys_ref,
                color="red",
                linewidth=style.reference_linewidth,
                alpha=0.5,
                zorder=100,
                path_effects=[
                    pe.Stroke(linewidth=7, foreground="white"),
                    pe.Normal(),
                ],
            )

    # Per-demo rendering
    for i, (states, trust_values) in enumerate(zip(states_collection, trust_list)):
        xs = np.asarray(states.coord(0), dtype=npDType)
        ys = np.asarray(states.coord(1), dtype=npDType)
        trust = np.asarray(trust_values, dtype=npDType)
        seg_trust = 0.5 * (trust[:-1] + trust[1:])

        # Isometric
        if style.mode == "isometric":
            dx = i * style.offset_x * style.spacing
            dy = i * style.offset_y * style.spacing
            xs_ = xs + dx
            ys_ = ys + dy
            segments = _make_segments_2d(xs_, ys_)
            linewidths = style.linewidth + style.trust_width_scale * (1.0 - seg_trust)
            collection = LineCollection(
                segments, cmap=style.cmap, linewidths=linewidths, alpha=style.alpha
            )
            collection.set_array(seg_trust)
            ax.add_collection(collection)

            # Faint backbone
            ax.plot(xs_, ys_, color="grey", alpha=0.15, linewidth=1.0, zorder=0)

            # Phase markers
            if style.show_phase_markers:
                idxs = np.linspace(0, len(xs_) - 1, style.phase_marker_count, dtype=int)
                ax.scatter(
                    xs_[idxs],
                    ys_[idxs],
                    s=24,
                    edgecolors="black",
                    linewidths=0.5,
                    zorder=5,
                )
            ax.text(xs_[0], ys_[0], f"D{i}", fontsize=9, alpha=0.8)  # Demo label

        # 3D
        else:
            demo_axis = np.full_like(xs, i * style.spacing)
            segments = _make_segments_3d(demo_axis, xs, ys)
            linewidths = style.linewidth + style.trust_width_scale * (1.0 - seg_trust)
            collection = Line3DCollection(
                segments, cmap=style.cmap, linewidths=linewidths, alpha=style.alpha
            )
            collection.set_array(seg_trust)
            ax.add_collection(collection)

            # Phase markers
            if style.show_phase_markers:
                idxs = np.linspace(
                    0,
                    len(xs) - 1,
                    style.phase_marker_count,
                    dtype=int,
                )
                ax.scatter(
                    demo_axis[idxs],
                    xs[idxs],
                    ys[idxs],
                    s=24,
                    edgecolors="black",
                    linewidths=0.5,
                )

    # Final styling
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if style.mode == "3d":
        ax.set_xlabel("Demonstration")
        ax.set_ylabel("x")
        ax.set_zlabel("y")
        ax.view_init(elev=20, azim=-35)
        ax.set_box_aspect((1, 2, 2))

        num_demos = len(trust_list)
        demo_positions = np.arange(num_demos) * style.spacing
        ax.set_xticks(demo_positions)
        ax.set_xticklabels([str(i) for i in range(num_demos)])
        ax.xaxis._axinfo["grid"]["linewidth"] = 0.0
    ax.margins(0.08)

    # Shared colorbar
    sm = plt.cm.ScalarMappable(cmap=style.cmap)
    sm.set_array([0.0, 1.0])
    fig.colorbar(sm, ax=ax, label="Trust")

    fig.tight_layout()


## ── PACER Visualisation ──────────────────────────────────────────────────────


@dataclass
class PACERVisualisationConfig:
    show: bool = False
    save_dir: Path | str | None = None

    trajectories: bool = False
    phases: bool = False
    trust_values: bool = False

    states_before_after: bool = False
    actions_before_after: bool = False
    action_comparison: bool = False
    state_comparison: bool = False

    ribbon_action_field: bool = False
    action_correction_magnitude: bool = False

    residual_distribution: bool = False
    trust_heatmap: bool = False
    bin_occupancy: bool = False
    trust_vs_correction: bool = False
    ribbon_statistics: bool = False
    phase_velocity: bool = False
    smoothness_comparison: bool = False

    trust_colored_trajectory: bool = False
    trust_colored_action_field: bool = False
    action_correction_vectors: bool = False
    phase_aligned_trajectories: bool = False
    ribbon_corridor: bool = False
    residual_vs_phase: bool = False
    action_angle_deviation: bool = False


@dataclass
class PACERVisualiser(RuntimeGeneric[NumBins, NumDemos, NumPoints]):
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
                    self.demonstrations.states, self.pacer_result.pseudo_labels.states
                )

        if self.config.actions_before_after:
            for i, demo in enumerate(self.demonstrations):
                plot_actions_before_after(
                    demo.states,
                    demo.actions,
                    self.pacer_result.pseudo_labels.actions[i],
                    title=f"Demo {i}: Actions Before vs After",
                    step=2,
                    show_corrections=True,
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
                self.demonstrations.actions, self.pacer_result.pseudo_labels.actions
            )

        if self.config.residual_distribution:
            residual_analysis = ResidualAnalyser(
                self.demonstrations, pacer_result=self.pacer_result
            ).compute_action_residuals()
            plot_residual_distribution(residual_analysis.residuals)

        if self.config.trust_heatmap:
            plot_trust_heatmap(self.pacer_result.action_trust_values)

        if self.config.bin_occupancy:
            plot_bin_occupancy(self.pacer_result.bins)

        if self.config.trust_vs_correction:
            correction_analysis = CorrectionMagnitudeAnalyser(
                self.demonstrations, self.pacer_result
            ).analyse_actions()
            plot_trust_vs_correction(
                self.pacer_result.action_trust_values, correction_analysis
            )

        if self.config.ribbon_statistics:
            plot_ribbon_statistics(self.pacer_result.bins)

        if self.config.phase_velocity:
            plot_phase_velocity(self.pacer_result.phases)

        if self.config.smoothness_comparison:
            original_smoothness = SmoothnessAnalyser(
                self.demonstrations.actions
            ).analyse()
            pseudo_smoothness = SmoothnessAnalyser(
                self.pacer_result.pseudo_labels.actions
            ).analyse()
            plot_smoothness_comparison(original_smoothness, pseudo_smoothness)

        if self.config.trust_colored_trajectory:
            for i, demo in enumerate(self.demonstrations):
                plot_trust_colored_trajectory(
                    demo.states,
                    self.pacer_result.action_trust_values[i],
                    title=f"Demo {i}: Trust-Colored Trajectory",
                )
            plot_stacked_trust_colored_trajectories(
                self.demonstrations.states,
                self.pacer_result.action_trust_values,
                self.pacer_result.bins.consensus_trajectory,
                style=StackedTrajectoryStyle(
                    mode="isometric", spacing=10.0, offset_x=0, offset_y=0
                ),
            )
            plot_stacked_trust_colored_trajectories(
                self.demonstrations.states,
                self.pacer_result.action_trust_values,
                self.pacer_result.bins.consensus_trajectory,
                style=StackedTrajectoryStyle(
                    mode="3d", spacing=10.0, offset_x=0, offset_y=0
                ),
            )

        if self.config.trust_colored_action_field:
            for i, demo in enumerate(self.demonstrations):
                plot_trust_colored_action_field(
                    demo.states,
                    demo.actions,
                    self.pacer_result.action_trust_values[i],
                    title=f"Demo {i}: Trust-Colored Action Field",
                )

        if self.config.action_correction_vectors:
            for i, demo in enumerate(self.demonstrations):
                plot_action_correction_vectors(
                    demo.states,
                    demo.actions,
                    self.pacer_result.pseudo_labels.actions[i],
                    title=f"Demo {i}: Action Correction Vectors",
                )

        if self.config.phase_aligned_trajectories:
            plot_phase_aligned_trajectories(
                self.demonstrations.states, self.pacer_result.phases
            )

        if self.config.ribbon_corridor:
            plot_ribbon_corridor(self.pacer_result.bins)

        if self.config.residual_vs_phase:
            residual_analysis = ResidualAnalyser(
                self.demonstrations, pacer_result=self.pacer_result
            ).compute_action_residuals()
            plot_residual_vs_phase(
                self.pacer_result.phases, residual_analysis.residuals
            )

        if self.config.action_angle_deviation:
            plot_action_angle_deviation(
                self.demonstrations.actions,
                self.pacer_result.bins,
                self.pacer_result.phases,
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

        plt.close("all")


## ─────────────────────────────────────────────────────────────────────────────

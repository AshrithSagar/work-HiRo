"""
Plotting utils
=======
"""
# src/pacer/plotting.py

# pyright: reportUnknownMemberType = false

## ── Imports ──────────────────────────────────────────────────────────────────

from collections.abc import Iterable
from dataclasses import KW_ONLY, dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typingkit.core import RuntimeGeneric
from typingkit.numpy._typed.helpers import TWO

from pacer import console
from pacer.base import (
    Actions,
    ActionsCollection,
    Demonstrations,
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
    for i, (orig, new) in enumerate(zip(original, pseudo)):
        ax.plot(orig.coord(0), orig.coord(1), "--", alpha=0.5, label=f"Original {i}")
        ax.plot(new.coord(0), new.coord(1), label=f"Refined {i}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.legend()
    ax.axis("equal")
    ax.margins(0.05)
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

    fig, ax = ensure_fig_ax(fig=fig, ax=ax, figsize=(6, 6))
    ax.quiver(xs, ys, us, vs, angles="xy", scale_units="xy", scale=scale)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.axis("equal")
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
        strengths.append(token.median_action_strength)
        variability.append(token.MAD_action_residual)
    ax.plot(strengths, label="Median action strength")
    ax.plot(variability, label="MAD residual")
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
    Plot ribbon median trajectory with variability corridor.
    Corridor radius is derived from MAD residual.
    """
    median_xs: list[npDType] = []
    median_ys: list[npDType] = []
    variability: list[Residual] = []
    for bin in bins:
        token = bin.ribbon_token
        median_state = token.median_state
        median_xs.append(median_state[0])
        median_ys.append(median_state[1])
        variability.append(token.MAD_action_residual)
    xs = np.asarray(median_xs, dtype=npDType)
    ys = np.asarray(median_ys, dtype=npDType)
    var = variability_scale * np.asarray(variability, dtype=npDType)

    fig, ax = ensure_fig_ax(fig=fig, ax=ax, figsize=(7, 7))
    ax.plot(xs, ys, linewidth=3, label="Ribbon median")
    ax.fill_between(xs, ys - var, ys + var, alpha=0.2, label="MAD corridor")
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
    Plot angular deviation from ribbon median action.
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
            reference = bin_list[bin_index].ribbon_token.median_action
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

    residual_distribution: bool = True
    trust_heatmap: bool = True
    bin_occupancy: bool = True
    trust_vs_correction: bool = True
    ribbon_statistics: bool = True
    phase_velocity: bool = True
    smoothness_comparison: bool = True

    trust_colored_trajectory: bool = True
    trust_colored_action_field: bool = True
    action_correction_vectors: bool = True
    phase_aligned_trajectories: bool = True
    ribbon_corridor: bool = True
    residual_vs_phase: bool = True
    action_angle_deviation: bool = True


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

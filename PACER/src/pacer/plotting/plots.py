"""
Plots
=====
Composable semantic plot factories.
"""
# src/pacer/plotting/plots.py

# pyright: reportUnknownMemberType = false

## ── Imports ──────────────────────────────────────────────────────────────────

from typingkit.numpy._typed.helpers import TWO

from pacer.base import Actions, Demonstrations, States, StatesCollection
from pacer.pacer.base import MetricSeries
from pacer.plotting.artists import (
    ColoredTrajectoryArtist,
    CorrectionVectorArtist,
    SeriesArtist,
    StartPointArtist,
    TrajectoryArtist,
    VectorFieldArtist,
)
from pacer.plotting.core import Plot2D
from pacer.plotting.geometry import ColoredTrajectory2D, trajectory2d, vectorfield2d
from pacer.plotting.styles import LineStyle, PlotStyle, QuiverStyle
from pacer.typings import NumDemos, NumPoints, Vector

## ── Plots ────────────────────────────────────────────────────────────────────


def trajectories_plot(
    demonstrations: Demonstrations[NumDemos, NumPoints, TWO, TWO],
    *,
    title: str = "Demonstration trajectories",
) -> Plot2D:

    plot = Plot2D(
        style=PlotStyle(
            title=title,
            xlabel="x",
            ylabel="y",
        )
    )

    for i, demo in enumerate(demonstrations):
        trajectory = trajectory2d(demo.states)

        plot.add(
            TrajectoryArtist(
                trajectory,
                label=f"Demo {i}",
            ),
            StartPointArtist(
                trajectory,
            ),
        )

    return plot


def states_plot(
    states_collection: StatesCollection[NumDemos, NumPoints, TWO],
    *,
    title: str = "States",
) -> Plot2D:

    plot = Plot2D(
        style=PlotStyle(
            title=title,
        )
    )

    for i, states in enumerate(states_collection):
        plot.add(
            TrajectoryArtist(
                trajectory2d(states),
                label=f"Demo {i}",
            )
        )

    return plot


def states_and_actions_plot(
    states: States[NumPoints, TWO],
    actions: Actions[NumPoints, TWO],
    *,
    title: str = "States and actions",
    action_scale: float = 1.0,
) -> Plot2D:

    trajectory = trajectory2d(states)
    field = vectorfield2d(states, actions)

    plot = Plot2D(
        style=PlotStyle(
            title=title,
        )
    )

    plot.add(
        TrajectoryArtist(
            trajectory,
        ),
        VectorFieldArtist(
            field,
            style=QuiverStyle(
                scale=action_scale,
                alpha=0.5,
            ),
        ),
        StartPointArtist(
            trajectory,
        ),
    )

    return plot


def actions_before_after_plot(
    states: States[NumPoints, TWO],
    original_actions: Actions[NumPoints, TWO],
    pseudo_actions: Actions[NumPoints, TWO],
    *,
    title: str = "Actions Before vs After",
    action_scale: float = 1.0,
    show_corrections: bool = False,
) -> Plot2D:

    trajectory = trajectory2d(states)

    original_field = vectorfield2d(states, original_actions)

    pseudo_field = vectorfield2d(states, pseudo_actions)

    plot = Plot2D(
        style=PlotStyle(
            title=title,
        )
    )

    plot.add(
        TrajectoryArtist(
            trajectory,
            style=LineStyle(
                alpha=0.3,
                color="grey",
            ),
        )
    )

    plot.add(
        VectorFieldArtist(
            original_field,
            label="Original",
            style=QuiverStyle(
                scale=action_scale,
                color="tab:blue",
            ),
        )
    )

    plot.add(
        VectorFieldArtist(
            pseudo_field,
            label="Pseudo",
            style=QuiverStyle(
                scale=action_scale,
                color="tab:orange",
            ),
        )
    )

    if show_corrections:
        plot.add(
            CorrectionVectorArtist(
                states=trajectory,
                original=original_field,
                pseudo=pseudo_field,
                scale=action_scale,
            )
        )

    return plot


def trust_colored_trajectory_plot(
    states: States[NumPoints, TWO],
    trust_values: MetricSeries[NumPoints],
    *,
    title: str = "Trust-Colored Trajectory",
    cmap: str = "viridis",
) -> Plot2D:

    trajectory = trajectory2d(states)

    geometry = ColoredTrajectory2D(
        trajectory=trajectory,
        values=Vector[NumPoints](trust_values),
        cmap=cmap,
    )

    plot = Plot2D(
        style=PlotStyle(
            title=title,
        )
    )

    plot.add(
        ColoredTrajectoryArtist(
            geometry,
        ),
        StartPointArtist(
            trajectory,
        ),
    )

    return plot


def series_plot(
    values: MetricSeries[NumPoints],
    *,
    title: str = "Series",
    ylabel: str = "Value",
) -> Plot2D:
    plot = Plot2D(
        style=PlotStyle(
            title=title,
            xlabel="Time index",
            ylabel=ylabel,
            equal=False,
        )
    )
    plot.add(SeriesArtist(Vector[NumPoints](values)))
    return plot


## ─────────────────────────────────────────────────────────────────────────────

"""
Artists
=======
Composable rendering primitives.
"""
# src/pacer/plotting/artists.py

# pyright: reportUnknownMemberType = false

## ── Imports ──────────────────────────────────────────────────────────────────

import dataclasses
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from typingkit.core import RuntimeGeneric

from pacer.plotting.geometry import (
    ColoredTrajectory2D,
    Trajectory2D,
    VectorField2D,
    segments2d,
)
from pacer.plotting.styles import LineStyle, QuiverStyle, ScatterStyle
from pacer.typings import NumPoints, Vector

## ── Artists ──────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class TextArtist:
    x: float
    y: float
    text: str

    alpha: float = 1.0
    fontsize: float = 10.0
    color: str | None = None

    def draw(self, ax: Axes) -> None:
        ax.text(
            self.x,
            self.y,
            self.text,
            alpha=self.alpha,
            fontsize=self.fontsize,
            color=self.color,
        )


@dataclass(slots=True)
class TrajectoryArtist(RuntimeGeneric[NumPoints]):
    trajectory: Trajectory2D[NumPoints]

    label: str | None = None

    style: LineStyle = field(default_factory=LineStyle)

    def draw(self, ax: Axes) -> None:
        ax.plot(
            self.trajectory.x,
            self.trajectory.y,
            label=self.label,
            linewidth=self.style.linewidth,
            alpha=self.style.alpha,
            linestyle=self.style.linestyle,
            color=self.style.color,
        )


@dataclass(slots=True)
class ScatterArtist(RuntimeGeneric[NumPoints]):
    trajectory: Trajectory2D

    label: str | None = None

    style: ScatterStyle = field(default_factory=ScatterStyle)

    def draw(self, ax: Axes) -> None:
        ax.scatter(
            self.trajectory.x,
            self.trajectory.y,
            s=self.style.size,
            alpha=self.style.alpha,
            color=self.style.color,
            label=self.label,
        )


@dataclass(slots=True)
class VectorFieldArtist(RuntimeGeneric[NumPoints]):
    field: VectorField2D[NumPoints]

    label: str | None = None

    style: QuiverStyle = dataclasses.field(default_factory=QuiverStyle)

    def draw(self, ax: Axes) -> None:
        ax.quiver(
            self.field.x,
            self.field.y,
            self.field.u,
            self.field.v,
            angles="xy",
            scale_units="xy",
            scale=1.0 / self.style.scale,
            alpha=self.style.alpha,
            width=self.style.width,
            color=self.style.color,
        )


@dataclass(slots=True)
class ColoredTrajectoryArtist(RuntimeGeneric[NumPoints]):
    geometry: ColoredTrajectory2D[NumPoints]

    linewidth: float = 3.0

    alpha: float = 1.0

    colorbar: bool = True

    def draw(self, ax: Axes) -> None:
        segments = segments2d(self.geometry.trajectory)

        values = np.asarray(
            self.geometry.values,
            dtype=np.float32,
        )

        segment_values = 0.5 * (values[:-1] + values[1:])

        collection = LineCollection(
            segments,
            cmap=self.geometry.cmap,
            linewidths=self.linewidth,
            alpha=self.alpha,
        )

        collection.set_array(segment_values)

        ax.add_collection(collection)

        if self.colorbar:
            fig = ax.figure
            fig.colorbar(collection, ax=ax)


@dataclass(slots=True)
class StartPointArtist(RuntimeGeneric[NumPoints]):
    trajectory: Trajectory2D[NumPoints]

    size: float = 32.0

    color: str | None = None

    def draw(self, ax: Axes) -> None:
        ax.scatter(
            [self.trajectory.x[0]],
            [self.trajectory.y[0]],
            s=self.size,
            color=self.color,
            zorder=10,
        )


@dataclass(slots=True)
class CorrectionVectorArtist(RuntimeGeneric[NumPoints]):
    states: Trajectory2D[NumPoints]

    original: VectorField2D[Any]
    pseudo: VectorField2D[Any]

    scale: float = 1.0

    alpha: float = 0.7

    def draw(self, ax: Axes) -> None:
        dx = self.pseudo.u - self.original.u
        dy = self.pseudo.v - self.original.v

        ax.quiver(
            self.states.x,
            self.states.y,
            dx,
            dy,
            angles="xy",
            scale_units="xy",
            scale=1.0 / self.scale,
            alpha=self.alpha,
            color="red",
        )


# ── Analytical Artists ────────────────────────────────────────────────────────


@dataclass(slots=True)
class SeriesArtist(RuntimeGeneric[NumPoints]):
    values: Vector[NumPoints]

    x: Vector[NumPoints] | None = None

    label: str | None = None

    linewidth: float = 2.0

    alpha: float = 1.0

    color: str | None = None

    linestyle: str = "-"

    def draw(self, ax: Axes) -> None:
        xs = self.x if self.x is not None else np.arange(len(self.values))

        ax.plot(
            xs,
            self.values,
            label=self.label,
            linewidth=self.linewidth,
            alpha=self.alpha,
            color=self.color,
            linestyle=self.linestyle,
        )


@dataclass(slots=True)
class HistogramArtist(RuntimeGeneric[NumPoints]):
    values: Vector[NumPoints]

    bins: int = 32

    alpha: float = 1.0

    label: str | None = None

    def draw(self, ax: Axes) -> None:
        ax.hist(
            self.values,
            bins=self.bins,
            alpha=self.alpha,
            label=self.label,
        )


@dataclass(slots=True)
class HeatmapArtist:
    matrix: np.ndarray

    cmap: str = "viridis"

    colorbar: bool = True

    def draw(self, ax: Axes) -> None:
        image = ax.imshow(
            self.matrix,
            aspect="auto",
            cmap=self.cmap,
        )

        if self.colorbar:
            ax.figure.colorbar(image, ax=ax)


@dataclass(slots=True)
class FillBetweenArtist:
    x: Vector[Any]

    lower: Vector[Any]

    upper: Vector[Any]

    alpha: float = 0.2

    label: str | None = None

    color: str | None = None

    def draw(self, ax: Axes) -> None:
        ax.fill_between(
            self.x,
            self.lower,
            self.upper,
            alpha=self.alpha,
            color=self.color,
            label=self.label,
        )


## ─────────────────────────────────────────────────────────────────────────────

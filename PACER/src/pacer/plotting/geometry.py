"""
Geometry
========
Geometry extraction layer.
"""
# src/pacer/plotting/geometry.py

## ── Imports ──────────────────────────────────────────────────────────────────

from dataclasses import dataclass

import numpy as np
from typingkit.core import RuntimeGeneric
from typingkit.numpy._typed.helpers import TWO

from pacer.base import Actions, States
from pacer.typings import NumPoints, Vector, npDType

## ── Geometry ─────────────────────────────────────────────────────────────────


@dataclass(slots=True, frozen=True)
class Trajectory2D(RuntimeGeneric[NumPoints]):
    x: Vector[NumPoints]
    y: Vector[NumPoints]

    @property
    def points(self) -> np.ndarray:
        return np.column_stack([self.x, self.y])


@dataclass(slots=True, frozen=True)
class VectorField2D(RuntimeGeneric[NumPoints]):
    x: Vector[NumPoints]
    y: Vector[NumPoints]

    u: Vector[NumPoints]
    v: Vector[NumPoints]


@dataclass(slots=True, frozen=True)
class ColoredTrajectory2D(RuntimeGeneric[NumPoints]):
    trajectory: Trajectory2D[NumPoints]
    values: Vector[NumPoints]

    cmap: str = "viridis"


def trajectory2d(states: States[NumPoints, TWO]) -> Trajectory2D[NumPoints]:
    return Trajectory2D(
        x=states.coord(0),
        y=states.coord(1),
    )


def vectorfield2d(
    states: States[NumPoints, TWO],
    actions: Actions[NumPoints, TWO],
) -> VectorField2D[NumPoints]:
    return VectorField2D(
        x=states.coord(0),
        y=states.coord(1),
        u=actions.coord(0),
        v=actions.coord(1),
    )


def segments2d(trajectory: Trajectory2D[NumPoints]) -> np.ndarray:
    points = trajectory.points.astype(npDType)
    return np.stack([points[:-1], points[1:]], axis=1)


## ─────────────────────────────────────────────────────────────────────────────

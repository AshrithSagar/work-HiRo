"""
Curve utils
=======
src/tp_gpt/curve.py
"""

from typing import Self

import numpy as np
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D  # type: ignore[import-untyped]
from numpy.typing import ArrayLike
from typed_numpy.helpers import Array1D

from tp_gpt.typings import THREE, TWO, DimT, Point, PointsArray, Space


class Curve(Space[DimT]):
    """Represents a general curve defined in a general space."""

    def __init__(self, points: ArrayLike) -> None:
        self.points = self.PointsArray(points)
        self.n_points: int = len(self.points)

    def __getitem__(self, idx: int) -> Point[DimT]:
        return self.points[idx]

    def __len__(self) -> int:
        return self.n_points

    def __repr__(self) -> str:
        return f"Curve(points={self.points}, shape={self.points.shape})"

    @property
    def start_pt(self) -> Point[DimT]:
        return self.points[0]

    @property
    def end_pt(self) -> Point[DimT]:
        return self.points[-1]

    @classmethod
    def from_points(cls, points: ArrayLike) -> Self:
        """Create a Curve instance from an array of points."""
        return cls(points=points)

    def plot(self, ax: Axes, *args, **kwargs) -> None:
        """Plot the curve on the given `Axes`."""
        raise NotImplementedError


class Curve2D(Curve[TWO]):
    """Represents a 2D curve in cartesian coordinates."""

    def __init__(self, xs: ArrayLike, ys: ArrayLike) -> None:
        self.xs = Array1D(xs)
        self.ys = Array1D(ys)
        super().__init__(points=np.column_stack((self.xs, self.ys)))

    @classmethod
    def from_points(cls, points: ArrayLike) -> Self:
        points_arr = PointsArray(points)
        return cls(xs=points_arr[:, 0], ys=points_arr[:, 1])

    def plot(self, ax: Axes, *args, **kwargs) -> None:
        ax.plot(self.xs, self.ys, *args, **kwargs)


class Curve3D(Curve[THREE]):
    """Represents a 3D curve in cartesian coordinates."""

    def __init__(self, xs: ArrayLike, ys: ArrayLike, zs: ArrayLike) -> None:
        self.xs = Array1D(xs)
        self.ys = Array1D(ys)
        self.zs = Array1D(zs)
        super().__init__(points=np.column_stack((self.xs, self.ys, self.zs)))

    @classmethod
    def from_points(cls, points: ArrayLike) -> Self:
        """Create a Curve3D instance from an array of points."""
        points_arr = PointsArray(points)
        return cls(xs=points_arr[:, 0], ys=points_arr[:, 1], zs=points_arr[:, 2])

    def plot(self, ax: Axes3D, *args, **kwargs) -> None:
        ax.plot(self.xs, self.ys, self.zs, *args, **kwargs)

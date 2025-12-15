"""
Curve utils
=======
src/tp_gpt/curve.py
"""

from typing import Generic, Self

import numpy as np
from matplotlib.axes import Axes
from numpy.typing import ArrayLike
from typed_numpy.helpers import Array2, Array3, ArrayN, ArrayNx2, ArrayNx3

from tp_gpt.typings import PointsT, PointT


class Curve(Generic[PointT, PointsT]):
    """Represents a general curve defined in a general space."""

    PointClass: type[PointT]
    PointsClass: type[PointsT]

    def __init__(self, points: ArrayLike) -> None:
        # Create a Curve instance from an array of points.
        self.points = self.PointsClass(points)
        self.n_points = len(self.points)

    def __getitem__(self, idx: int) -> PointT:
        return self.points[idx]

    def __len__(self) -> int:
        return self.n_points

    def __repr__(self) -> str:
        return f"Curve(points={self.points}, shape={self.points.shape})"

    @property
    def start_pt(self) -> PointT:
        return self.points[0]

    @property
    def end_pt(self) -> PointT:
        return self.points[-1]

    def plot(self, ax: Axes, *args, **kwargs) -> None:
        """Plot the curve on the given `Axes`."""
        raise NotImplementedError


class Curve2D(Curve[Array2, ArrayNx2]):
    """Represents a 2D curve defined by x and y coordinates."""

    PointClass = Array2
    PointsClass = ArrayNx2

    def __init__(self, xs: ArrayLike, ys: ArrayLike) -> None:
        self.xs = ArrayN(xs)
        self.ys = ArrayN(ys)
        self.points = self.PointsClass(np.column_stack((self.xs, self.ys)))
        self.n_points = len(self.points)

    @classmethod
    def from_points(cls, points: ArrayLike) -> Self:
        """Create a Curve2D instance from an array of points."""
        points_arr = cls.PointsClass(points)
        return cls(xs=points_arr[:, 0], ys=points_arr[:, 1])

    def plot(self, ax: Axes, *args, **kwargs) -> None:
        ax.plot(self.xs, self.ys, *args, **kwargs)


class Curve3D(Curve[Array3, ArrayNx3]):
    """Represents a 3D curve defined by x, y and z coordinates."""

    PointClass = Array3
    PointsClass = ArrayNx3

    def __init__(self, xs: ArrayLike, ys: ArrayLike, zs: ArrayLike) -> None:
        self.xs = ArrayN(xs)
        self.ys = ArrayN(ys)
        self.zs = ArrayN(zs)
        self.points = self.PointsClass(np.column_stack((self.xs, self.ys, self.zs)))
        self.n_points = len(self.points)

    @classmethod
    def from_points(cls, points: ArrayLike) -> Self:
        """Create a Curve3D instance from an array of points."""
        points_arr = cls.PointsClass(points)
        return cls(xs=points_arr[:, 0], ys=points_arr[:, 1], zs=points_arr[:, 2])

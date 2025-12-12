"""
Curve utils
=======
src/tp_gpt/curve.py
"""

from typing import Self

import numpy as np
from matplotlib.axes import Axes
from numpy.typing import ArrayLike
from typed_numpy import TypedNDArray
from typed_numpy.helpers import Array2, Array3, ArrayN, ArrayNx2, ArrayNx3


class Curve:
    """Represents a general curve defined in a general space."""

    def __init__(self, points: ArrayLike) -> None:
        # Create a Curve instance from an array of points.
        self.points = TypedNDArray(points)
        self.n_points = len(self.points)

    def __getitem__(self, idx: int) -> TypedNDArray:
        return self.points[idx]

    def __len__(self) -> int:
        return self.n_points

    def __repr__(self) -> str:
        return f"Curve(points={self.points}, shape={self.points.shape})"

    @property
    def start_pt(self) -> TypedNDArray:
        return self.points[0]

    @property
    def end_pt(self) -> TypedNDArray:
        return self.points[-1]

    def plot(self, ax: Axes, *args, **kwargs) -> None:
        """Plot the curve on the given `Axes`."""
        ax.plot(*self.points, *args, **kwargs)


class Curve2D:
    """Represents a 2D curve defined by x and y coordinates."""

    def __init__(self, xs: ArrayLike, ys: ArrayLike) -> None:
        self.xs = ArrayN(xs)
        self.ys = ArrayN(ys)
        self.points = ArrayNx2(np.column_stack((self.xs, self.ys)))
        self.n_points = len(self.points)

    def __getitem__(self, idx: int) -> Array2:
        return self.points[idx]

    def __len__(self) -> int:
        return self.n_points

    def __repr__(self) -> str:
        return f"Curve2D(points={self.points}, shape={self.points.shape})"

    @property
    def start_pt(self) -> Array2:
        return self.points[0]

    @property
    def end_pt(self) -> Array2:
        return self.points[-1]

    @classmethod
    def from_points(cls, points: ArrayLike) -> Self:
        """Create a Curve2D instance from an array of points."""
        points_arr = ArrayNx2(points)
        return cls(xs=points_arr[:, 0], ys=points_arr[:, 1])

    def plot(self, ax: Axes, *args, **kwargs) -> None:
        """Plot the curve on the given `Axes`."""
        ax.plot(self.xs, self.ys, *args, **kwargs)


class Curve3D:
    """Represents a 3D curve defined by x, y and z coordinates."""

    def __init__(self, xs: ArrayLike, ys: ArrayLike, zs: ArrayLike) -> None:
        self.xs = ArrayN(xs)
        self.ys = ArrayN(ys)
        self.zs = ArrayN(zs)
        self.points = ArrayNx3(np.column_stack((self.xs, self.ys, self.zs)))
        self.n_points = len(self.points)

    def __getitem__(self, idx: int) -> Array3:
        return self.points[idx]

    def __len__(self) -> int:
        return self.n_points

    def __repr__(self) -> str:
        return f"Curve2D(points={self.points}, shape={self.points.shape})"

    @property
    def start_pt(self) -> Array3:
        return self.points[0]

    @property
    def end_pt(self) -> Array3:
        return self.points[-1]

    @classmethod
    def from_points(cls, points: ArrayLike) -> Self:
        """Create a Curve3D instance from an array of points."""
        points_arr = ArrayNx3(points)
        return cls(xs=points_arr[:, 0], ys=points_arr[:, 1], zs=points_arr[:, 2])

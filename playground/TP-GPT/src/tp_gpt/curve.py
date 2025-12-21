"""
Curve utils
=======
src/tp_gpt/curve.py
"""

from typing import Self

import numpy as np
from matplotlib.axes import Axes
from numpy.typing import ArrayLike

from tp_gpt.core.spaces import Point, ScalarArray, SpaceCollection
from tp_gpt.core.typings import DimSpace, NumPoints, ThreeD, TwoD


class Curve(SpaceCollection[NumPoints, DimSpace]):
    """Represents a general curve defined in a general space."""

    def __init__(self, points: ArrayLike) -> None:
        self.points = self._PointSet(points)
        self.n_points, self.dim = self.points.shape

    def __getitem__(self, idx: int) -> Point[DimSpace]:
        return self._Point(self.points[idx])

    def __len__(self) -> int:
        return self.n_points

    def __repr__(self) -> str:
        return f"Curve(points={self.points}, shape={self.points.shape})"

    @property
    def start_pt(self) -> Point[DimSpace]:
        return self[0]

    @property
    def end_pt(self) -> Point[DimSpace]:
        return self[-1]

    def coord(self, axis: int) -> ScalarArray[NumPoints]:
        """Return one coordinate component by axis index."""
        if not 0 <= axis < self.dim:
            raise IndexError(f"Axis {axis} out of bounds for dim={self.dim}")
        return self._ScalarArray(self.points[:, axis])

    @property
    def components(self) -> tuple[ScalarArray[NumPoints], ...]:
        """Return coordinate components as a tuple"""
        return tuple(self.coord(i) for i in range(self.dim))

    @classmethod
    def from_components(cls, *components: ArrayLike) -> Self:
        return cls(np.column_stack(components))

    def plot(self, ax: Axes, *args, **kwargs) -> None:
        """Plot the curve on the given `Axes`."""
        assert 2 <= self.dim <= 3, "Base implementation only supports 2D / 3D plots."
        ax.plot(*self.components, *args, **kwargs)


class Curve2D(Curve[NumPoints, TwoD]):
    """Represents a 2D curve in cartesian coordinates."""

    @classmethod
    def from_xy(cls, xs: ArrayLike, ys: ArrayLike) -> Self:
        return cls.from_components(xs, ys)


class Curve3D(Curve[NumPoints, ThreeD]):
    """Represents a 3D curve in cartesian coordinates."""

    @classmethod
    def from_xyz(cls, xs: ArrayLike, ys: ArrayLike, zs: ArrayLike) -> Self:
        return cls.from_components(xs, ys, zs)

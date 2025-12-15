"""
Obstacles
=======
src/tp_gpt/obstacle.py
"""

from abc import ABC, abstractmethod
from typing import Generic

import numpy as np
from matplotlib.axes import Axes
from numpy.typing import ArrayLike
from typed_numpy.helpers import Array2, Array3, ArrayNx2, ArrayNx3, def_dtype

from tp_gpt.typings import PointsT, PointT


class Obstacle(Generic[PointT, PointsT], ABC):
    """Abstract base class for obstacles."""

    PointClass: type[PointT]
    PointsClass: type[PointsT]

    @abstractmethod
    def boundary_points(self) -> PointsT:
        """Returns an array of points describing the obstacle boundary."""
        raise NotImplementedError

    @property
    def center(self) -> PointT:
        """Returns the center point of the obstacle."""
        # Defaults to the centroid of the boundary points
        pts = self.boundary_points()
        return self.PointClass(np.mean(pts, axis=0))

    def plot(self, ax: Axes, *args, **kwargs) -> None:
        """Plot the obstacle on the given `Axes`."""
        pts = self.boundary_points()
        ax.plot(pts[:, 0], pts[:, 1], *args, **kwargs)


class BallObstacle(Obstacle[PointT, PointsT]):
    """A general n-Ball Obstacle"""

    def __init__(self, center: ArrayLike, radius: float, n_points: int = 20):
        assert radius > 0, "Radius must be positive."
        assert n_points >= 3, "Number of points must be at least 3."

        self._center = self.PointClass(center)
        self.radius = float(radius)
        self.n_points = int(n_points)

    @property
    def center(self) -> PointT:
        return self._center

    def boundary_points(self) -> PointsT:
        theta = np.linspace(0, 2 * np.pi, self.n_points)
        cx, cy = self._center
        X = cx + self.radius * np.cos(theta, dtype=def_dtype)
        Y = cy + self.radius * np.sin(theta, dtype=def_dtype)
        return self.PointsClass(np.column_stack((X, Y)))

    @property
    def center_tile(self) -> PointsT:
        return self.PointsClass(np.tile(self.center, (self.n_points, 1)))


class CircularObstacle(BallObstacle[Array2, ArrayNx2]):
    """A 2D Circular Obstacle"""

    PointClass = Array2
    PointsClass = ArrayNx2


class SphericalObstacle(BallObstacle[Array3, ArrayNx3]):
    """A 3D Spherical Obstacle"""

    PointClass = Array3
    PointsClass = ArrayNx3

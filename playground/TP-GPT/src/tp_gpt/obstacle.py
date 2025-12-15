"""
Obstacles
=======
src/tp_gpt/obstacle.py
"""

from abc import ABC, abstractmethod
from functools import cached_property
from typing import Generic

import numpy as np
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D  # type: ignore[import-untyped]
from numpy.typing import ArrayLike

from tp_gpt.typings import DimT, Point, PointsArray, ThreeD, TwoD


class Obstacle(Generic[DimT], ABC):
    """Abstract base class for obstacles."""

    @cached_property
    @abstractmethod
    def boundary_points(self) -> PointsArray[DimT]:
        """Returns an array of points describing the obstacle boundary."""
        raise NotImplementedError

    @property
    def center(self) -> Point[DimT]:
        """Returns the center point of the obstacle."""
        # Defaults to the centroid of the boundary points
        return Point[DimT](np.mean(self.boundary_points, axis=0))

    @property
    def n_points(self) -> int:
        """Number of boundary points."""
        return self.boundary_points.shape[0]

    @n_points.setter
    def n_points(self, value: int) -> None: ...

    @property
    def center_tile(self) -> PointsArray[DimT]:
        return PointsArray[DimT](np.tile(self.center, (self.n_points, 1)))

    def plot(self, ax: Axes, *args, **kwargs) -> None:
        """Plot the obstacle on the given `Axes`."""
        pts = self.boundary_points
        if pts.shape[1] == 2:
            ax.plot(pts[:, 0], pts[:, 1], *args, **kwargs)
        elif pts.shape[1] == 3:
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], *args, **kwargs)
        else:
            raise ValueError("Unsupported obstacle dimensionality")


class BallObstacle(Obstacle[DimT], ABC):
    """Abstract base class for ball-shaped obstacles."""

    _center: Point[DimT]
    radius: float

    @cached_property
    @abstractmethod
    def boundary_points(self) -> PointsArray[DimT]:
        raise NotImplementedError

    @property
    def center(self) -> Point[DimT]:
        return self._center


class CircularObstacle(BallObstacle[TwoD]):
    """A 2D Circular Obstacle"""

    def __init__(self, center: ArrayLike, radius: float, n_theta: int = 20) -> None:
        self._center = Point[TwoD](center)
        self.radius = float(np.abs(radius))
        self.n_theta = int(n_theta)

        assert self.n_theta >= 3
        self.n_points = self.n_theta

    @cached_property
    def boundary_points(self) -> PointsArray[TwoD]:
        theta = np.linspace(0, 2 * np.pi, self.n_theta)

        cx, cy = self._center
        xs = cx + self.radius * np.cos(theta)
        ys = cy + self.radius * np.sin(theta)

        return PointsArray[TwoD](np.column_stack((xs, ys)))

    def plot(self, ax: Axes, *args, **kwargs) -> None:
        pts = self.boundary_points
        ax.plot(pts[:, 0], pts[:, 1], *args, **kwargs)


class SphericalObstacle(BallObstacle[ThreeD]):
    """A 3D Spherical Obstacle"""

    def __init__(
        self,
        center: ArrayLike,
        radius: float,
        n_theta: int = 20,
        n_phi: int = 20,
    ) -> None:
        self._center = Point[ThreeD](center)
        self.radius = float(np.abs(radius))
        self.n_theta = int(n_theta)
        self.n_phi = int(n_phi)

        assert self.n_theta >= 3
        assert self.n_phi >= 2
        self.n_points = self.n_theta * self.n_phi

    @cached_property
    def boundary_points(self) -> PointsArray[ThreeD]:
        theta = np.linspace(0, 2 * np.pi, self.n_theta)
        phi = np.linspace(0, np.pi, self.n_phi)
        theta, phi = np.meshgrid(theta, phi)

        cx, cy, cz = self._center
        xs = cx + self.radius * np.cos(theta) * np.sin(phi)
        ys = cy + self.radius * np.sin(theta) * np.sin(phi)
        zs = cz + self.radius * np.cos(phi)

        return PointsArray[ThreeD](
            np.column_stack((xs.ravel(), ys.ravel(), zs.ravel()))
        )

    def plot(self, ax: Axes3D, *args, **kwargs) -> None:
        pts = self.boundary_points
        ax.plot_surface(pts[:, 0], pts[:, 1], pts[:, 2], *args, **kwargs)

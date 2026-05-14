"""
Obstacles
=======
src/gpto/obstacle.py
"""

from abc import ABC, abstractmethod
from typing import Generic

import numpy as np
from numpy.typing import ArrayLike
from typed_numpy._typed.helpers import Array1D, Array2D

from gpto.core.typings import DimSpace, NumPoints, Point, PointSet, ThreeD, TwoD


class Obstacle(Generic[NumPoints, DimSpace], ABC):
    """Abstract base class for obstacles."""

    @property
    @abstractmethod
    def boundary_points(self) -> PointSet[NumPoints, DimSpace]:
        """Returns an array of points describing the obstacle boundary."""
        raise NotImplementedError

    @property
    def center(self) -> Point[DimSpace]:
        """Returns the center point of the obstacle."""
        # Defaults to the centroid of the boundary points
        return Point[DimSpace](np.mean(self.boundary_points, axis=0))

    @property
    def n_points(self) -> int:
        """Number of boundary points."""
        return self.boundary_points.shape[0]

    @n_points.setter
    def n_points(self, value: int) -> None: ...

    @property
    def center_tile(self) -> PointSet[NumPoints, DimSpace]:
        return PointSet[NumPoints, DimSpace](np.tile(self.center, (self.n_points, 1)))


class BallObstacle(Obstacle[NumPoints, DimSpace], ABC):
    """Abstract base class for ball-shaped obstacles."""

    _center: Point[DimSpace]
    radius: float

    @property
    @abstractmethod
    def boundary_points(self) -> PointSet[NumPoints, DimSpace]:
        raise NotImplementedError

    @property
    def center(self) -> Point[DimSpace]:
        return self._center


class CircularObstacle(BallObstacle[NumPoints, TwoD]):
    """A 2D Circular Obstacle"""

    def __init__(self, center: ArrayLike, radius: float, n_theta: int = 20) -> None:
        self._center = Point[TwoD](center)
        self.radius = float(np.abs(radius))
        self.n_theta = int(n_theta)

        assert self.n_theta >= 3
        self.n_points = self.n_theta

    @property
    def boundary_points(self) -> PointSet[NumPoints, TwoD]:
        theta = np.linspace(0, 2 * np.pi, self.n_theta)

        cx, cy = self._center
        xs = cx + self.radius * np.cos(theta)
        ys = cy + self.radius * np.sin(theta)

        return PointSet[NumPoints, TwoD](np.column_stack((xs, ys)))


class SphericalObstacle(BallObstacle[NumPoints, ThreeD]):
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

    def _spherical_mesh(self):
        _theta = Array1D(np.linspace(0, 2 * np.pi, self.n_theta))
        _phi = Array1D(np.linspace(0, np.pi, self.n_phi))
        theta, phi = np.meshgrid(_theta, _phi)

        cx, cy, cz = self._center
        X = Array2D(cx + self.radius * np.outer(np.cos(theta), np.sin(phi)))
        Y = Array2D(cy + self.radius * np.outer(np.sin(theta), np.sin(phi)))
        _ones = np.ones((self.n_phi, self.n_theta))
        Z = Array2D(cz + self.radius * np.outer(_ones, np.cos(phi)))
        return X, Y, Z

    @property
    def boundary_points(self) -> PointSet[NumPoints, ThreeD]:
        X, Y, Z = self._spherical_mesh()
        return PointSet[NumPoints, ThreeD](
            np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
        )

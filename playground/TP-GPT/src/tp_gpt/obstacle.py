"""
Obstacles
=======
src/tp_gpt/obstacle.py
"""

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D  # type: ignore[import-untyped]
from numpy.typing import ArrayLike
from typed_numpy._typed.helpers import Array1D, Array2D

from tp_gpt.core.spaces import Point, PointSet, SpaceCollection
from tp_gpt.core.typings import DimSpace, NumPoints, ThreeD, TwoD


class Obstacle(SpaceCollection[NumPoints, DimSpace], ABC):
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
        return self._Point(np.mean(self.boundary_points, axis=0))

    @property
    def n_points(self) -> int:
        """Number of boundary points."""
        return self.boundary_points.shape[0]

    @n_points.setter
    def n_points(self, value: int) -> None: ...

    @property
    def center_tile(self) -> PointSet[NumPoints, DimSpace]:
        return self._PointSet(np.tile(self.center, (self.n_points, 1)))

    def plot(self, ax: Axes, *args, **kwargs) -> None:
        """Plot the obstacle on the given `Axes`."""
        pts = self.boundary_points
        if pts.shape[1] == 2:
            ax.plot(pts[:, 0], pts[:, 1], *args, **kwargs)
        elif pts.shape[1] == 3:
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], *args, **kwargs)
        else:
            raise ValueError("Unsupported obstacle dimensionality")


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
        self._center = self._Point(center)
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

        return self._PointSet(np.column_stack((xs, ys)))

    def plot(self, ax: Axes, *args, **kwargs) -> None:
        pts = self.boundary_points
        ax.plot(pts[:, 0], pts[:, 1], *args, **kwargs)


class SphericalObstacle(BallObstacle[NumPoints, ThreeD]):
    """A 3D Spherical Obstacle"""

    def __init__(
        self,
        center: ArrayLike,
        radius: float,
        n_theta: int = 20,
        n_phi: int = 20,
    ) -> None:
        self._center = self._Point(center)
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
        return self._PointSet(np.column_stack((X.ravel(), Y.ravel(), Z.ravel())))

    def plot(
        self,
        ax: Axes3D,
        mode: Literal["scatter", "surface", "wireframe"],
        *args,
        **kwargs,
    ) -> None:
        match mode:
            case "scatter":
                pts = self.boundary_points
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], *args, **kwargs)  # type: ignore
            case "surface":
                X, Y, Z = self._spherical_mesh()
                ax.plot_surface(X, Y, Z, *args, **kwargs)
            case "wireframe":
                X, Y, Z = self._spherical_mesh()
                ax.plot_wireframe(X, Y, Z, *args, **kwargs)

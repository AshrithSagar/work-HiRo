"""
Obstacles
=======
src/tp_gpt/obstacle.py
"""

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from typed_numpy.helpers import Array1D, Array2D, def_dtype


class Obstacle(ABC):
    """Abstract base class for obstacles."""

    @abstractmethod
    def boundary_points(self) -> Array2D:
        """Returns an array of points describing the obstacle boundary."""
        raise NotImplementedError

    @property
    def center(self) -> Array1D:
        """Returns the center point of the obstacle."""
        # Defaults to the centroid of the boundary points
        pts = self.boundary_points()
        return Array1D(np.mean(pts, axis=0))

    def plot(self, ax: Axes, *args, **kwargs):
        """Plots the obstacle on the given `Axes`."""
        pts = self.boundary_points()
        ax.plot(pts[:, 0], pts[:, 1], *args, **kwargs)


class CircularObstacle(Obstacle):
    def __init__(self, center: npt.ArrayLike, radius: float, n_points: int = 20):
        assert radius > 0, "Radius must be positive."
        assert n_points >= 3, "Number of points must be at least 3."

        self._center = Array1D(center)
        self.radius = float(radius)
        self.n_points = int(n_points)

    @property
    def center(self) -> Array1D:
        return self._center

    def boundary_points(self) -> Array2D:
        theta = Array1D(np.linspace(0, 2 * np.pi, self.n_points))
        cx, cy = self._center
        X = cx + self.radius * np.cos(theta, dtype=def_dtype)
        Y = cy + self.radius * np.sin(theta, dtype=def_dtype)
        return Array2D(np.column_stack((X, Y)))

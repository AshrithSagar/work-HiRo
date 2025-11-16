"""
Obstacles
=======
src/tp_gpt/obstacle.py
"""

from abc import ABC, abstractmethod

import numpy as np

from tp_gpt.typings import Array1D, Array2D, dtype


class Obstacle(ABC):
    """Abstract base class for obstacles."""

    @abstractmethod
    def boundary_points(self) -> Array2D:
        """Returns an array of points describing the obstacle boundary."""
        raise NotImplementedError

    def center(self) -> Array1D:
        """Returns the center point of the obstacle."""
        # Defaults to the centroid of the boundary points
        pts: Array2D = self.boundary_points()
        return np.mean(pts, axis=0)


class CircleObstacle(Obstacle):
    def __init__(self, center: Array1D, radius: float, n_points: int = 20):
        self._center: Array1D = np.asarray(center, dtype=dtype)
        self.radius = float(radius)
        self.n_points = int(n_points)

    def center(self) -> Array1D:
        return self._center

    def boundary_points(self) -> Array2D:
        theta: Array1D = np.linspace(0, 2 * np.pi, self.n_points, dtype=dtype)
        cx, cy = self._center
        X = cx + self.radius * np.cos(theta, dtype=dtype)
        Y = cy + self.radius * np.sin(theta, dtype=dtype)
        return np.column_stack((X, Y)).astype(dtype=dtype)

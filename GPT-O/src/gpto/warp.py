"""
Warp utils
=======
src/gpto/warp.py
"""

from typing import Generic, Sequence

import numpy as np

from gpto.core.transportation import NonLinearTransform, PolicyTransportation
from gpto.core.typings import DimSpace, Point, PointSet, ThreeD, TwoD
from gpto.curve import Curve
from gpto.obstacle import Obstacle


class ObstacleAvoidanceWarp(Generic[NonLinearTransform, DimSpace]):
    def __init__(
        self,
        transportation: PolicyTransportation[NonLinearTransform, DimSpace],
        obstacles: Sequence[Obstacle[int, DimSpace]],
        curve: Curve[int, DimSpace],
    ) -> None:
        self.transportation = transportation
        self.obstacles = obstacles
        self.curve = curve

    def _make_keypoints(
        self, target_end_point: Point[DimSpace]
    ) -> tuple[PointSet[int, DimSpace], PointSet[int, DimSpace]]:
        obs_pts = PointSet[int, DimSpace](
            np.vstack([o.boundary_points for o in self.obstacles])
        )
        obs_centers = PointSet[int, DimSpace](
            np.vstack([o.center_tile for o in self.obstacles])
        )
        target_points = PointSet[int, DimSpace](
            np.vstack((self.curve.start_pt, obs_pts, target_end_point))
        )
        source_points = PointSet[int, DimSpace](
            np.vstack((self.curve.start_pt, obs_centers, self.curve.end_pt))
        )
        return source_points, target_points

    def warp_curve(self) -> Curve[int, DimSpace]:
        warped = self.transportation.transport_positions(self.curve.points)
        return type(self.curve)(points=warped)

    def fit(self, target_end_point: Point[DimSpace]) -> None:
        source_points, target_points = self._make_keypoints(target_end_point)
        self.transportation.fit(source_points, target_points)


class ObstacleAvoidanceWarp2D(ObstacleAvoidanceWarp[NonLinearTransform, TwoD]): ...


class ObstacleAvoidanceWarp3D(ObstacleAvoidanceWarp[NonLinearTransform, ThreeD]): ...

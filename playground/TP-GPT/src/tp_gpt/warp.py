"""
Warp utils
=======
src/tp_gpt/warp.py
"""

from typing import Generic, Sequence

import numpy as np

from tp_gpt.core.spaces import (
    DimSpace,
    NumPoints,
    Point,
    PointSet,
    SpaceCollection,
    ThreeD,
    TwoD,
)
from tp_gpt.core.transportation import NonLinearTransform, PolicyTransportation
from tp_gpt.curve import Curve, Curve2D, Curve3D
from tp_gpt.obstacle import Obstacle


class ObstacleAvoidanceWarp(
    Generic[NumPoints, DimSpace, NonLinearTransform],
    SpaceCollection[NumPoints, DimSpace],
):
    def __init__(
        self,
        transportation: PolicyTransportation[NumPoints, DimSpace, NonLinearTransform],
        obstacles: Sequence[Obstacle[NumPoints, DimSpace]],
        curve: Curve[NumPoints, DimSpace],
    ) -> None:
        self.transportation = transportation
        self.obstacles = obstacles
        self.curve = curve

    def _make_keypoints(
        self, target_end_point: Point[DimSpace]
    ) -> tuple[PointSet[NumPoints, DimSpace], PointSet[NumPoints, DimSpace]]:
        obs_pts = self._PointSet(np.vstack([o.boundary_points for o in self.obstacles]))
        obs_centers = self._PointSet(np.vstack([o.center_tile for o in self.obstacles]))
        target_points = self._PointSet(
            np.vstack((self.curve.start_pt, obs_pts, target_end_point))
        )
        source_points = self._PointSet(
            np.vstack((self.curve.start_pt, obs_centers, self.curve.end_pt))
        )
        return source_points, target_points

    def warp_curve(self) -> Curve[NumPoints, DimSpace]:
        warped = self.transportation.transport_positions(self.curve.points)
        return type(self.curve).from_points(warped)

    def fit(self, target_end_point: Point[DimSpace]) -> None:
        source_points, target_points = self._make_keypoints(target_end_point)
        self.transportation.fit(source_points, target_points)


class ObstacleAvoidanceWarp2D(
    ObstacleAvoidanceWarp[NumPoints, TwoD, NonLinearTransform]
):
    def warp_curve(self) -> Curve2D[NumPoints]:
        warped = self.transportation.transport_positions(self.curve.points)
        return Curve2D[NumPoints].from_points(warped)


class ObstacleAvoidanceWarp3D(
    ObstacleAvoidanceWarp[NumPoints, ThreeD, NonLinearTransform]
):
    def warp_curve(self) -> Curve3D[NumPoints]:
        warped = self.transportation.transport_positions(self.curve.points)
        return Curve3D[NumPoints].from_points(warped)

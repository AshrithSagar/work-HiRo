"""
Policy transportation
=======
src/tp_gpt/transportation.py
"""

from typing import Generic, Optional, TypeVar

import numpy as np

from tp_gpt.transforms import AffineTransform, Transform
from tp_gpt.typings import THREE, TWO, DimT, JacobianArray, PointsArray

NonLinearTransformT = TypeVar("NonLinearTransformT", bound=Transform)


class PolicyTransportation(Generic[NonLinearTransformT, DimT], Transform[DimT]):
    def __init__(
        self,
        nonlinear_transform: Optional[NonLinearTransformT] = None,
        affine_transform: AffineTransform = AffineTransform(scale=False, rotate=True),
        *,
        use_residuals: bool = True,
    ):
        self.affine_transform = affine_transform
        self.nonlinear_transform = nonlinear_transform
        self.use_residuals = use_residuals

    def fit(
        self, source_points: PointsArray[DimT], target_points: PointsArray[DimT], /
    ) -> None:
        self.affine_transform.fit(source_points, target_points)
        source_points_transformed = self.affine_transform.predict(source_points)

        if self.nonlinear_transform is not None:
            if self.use_residuals:
                residuals = PointsArray[DimT](target_points - source_points_transformed)
                self.nonlinear_transform.fit(source_points_transformed, residuals)
            else:
                self.nonlinear_transform.fit(source_points_transformed, target_points)

    def predict(self, points: PointsArray[DimT], /) -> PointsArray[DimT]:
        points_transformed = PointsArray[DimT](self.affine_transform.predict(points))
        if self.nonlinear_transform is not None:
            residuals = PointsArray[DimT](
                self.nonlinear_transform.predict(points_transformed)
            )
            if self.use_residuals:
                return PointsArray[DimT](points_transformed + residuals)
            else:
                return residuals
        else:
            return points_transformed

    def jacobian(self, points: PointsArray[DimT], /) -> JacobianArray[DimT]:
        points_transformed = self.affine_transform.predict(points)
        J_gamma = self.affine_transform.jacobian(points)
        if self.nonlinear_transform is not None:
            J_psi = JacobianArray[DimT](
                self.nonlinear_transform.jacobian(points_transformed)
            )
            if self.use_residuals:
                J_phi = J_gamma + J_psi @ J_gamma
            else:
                J_phi = J_psi @ J_gamma
            return JacobianArray[DimT](J_phi)
        else:
            return JacobianArray[DimT](J_gamma)

    def transport_positions(self, positions: PointsArray[DimT], /) -> PointsArray[DimT]:
        return self.predict(positions)

    def transport_velocities(
        self, positions: PointsArray[DimT], velocities: PointsArray[DimT]
    ) -> PointsArray[DimT]:
        J_phi = self.jacobian(positions)
        velocities_transported = PointsArray[DimT](
            np.einsum("nij,nj->ni", J_phi, velocities)
        )
        return velocities_transported


class PolicyTransportation2D(
    Generic[NonLinearTransformT], PolicyTransportation[NonLinearTransformT, TWO]
): ...


class PolicyTransportation3D(
    Generic[NonLinearTransformT],
    PolicyTransportation[NonLinearTransformT, THREE],
): ...

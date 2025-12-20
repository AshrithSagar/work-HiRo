"""
Policy transportation
=======
src/tp_gpt/transportation.py
"""

from typing import Generic, Optional, TypeVar

import numpy as np

from tp_gpt.transforms import AffineTransform, Transform
from tp_gpt.typings import THREE, TWO, DimDT, DimNT, JacobianArray, PointsArray

NonLinearTransformT = TypeVar("NonLinearTransformT", bound=Transform)


class PolicyTransportation(
    Generic[NonLinearTransformT, DimNT, DimDT], Transform[DimNT, DimDT]
):
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
        self,
        source_points: PointsArray[DimNT, DimDT],
        target_points: PointsArray[DimNT, DimDT],
        /,
    ) -> None:
        self.affine_transform.fit(source_points, target_points)
        source_points_transformed = self.affine_transform.predict(source_points)

        if self.nonlinear_transform is not None:
            if self.use_residuals:
                residuals = PointsArray[DimNT, DimDT](
                    target_points - source_points_transformed
                )
                self.nonlinear_transform.fit(source_points_transformed, residuals)
            else:
                self.nonlinear_transform.fit(source_points_transformed, target_points)

    def predict(
        self, points: PointsArray[DimNT, DimDT], /
    ) -> PointsArray[DimNT, DimDT]:
        points_transformed = PointsArray[DimNT, DimDT](
            self.affine_transform.predict(points)
        )
        if self.nonlinear_transform is not None:
            residuals = PointsArray[DimNT, DimDT](
                self.nonlinear_transform.predict(points_transformed)
            )
            if self.use_residuals:
                return PointsArray[DimNT, DimDT](points_transformed + residuals)
            else:
                return residuals
        else:
            return points_transformed

    def jacobian(
        self, points: PointsArray[DimNT, DimDT], /
    ) -> JacobianArray[DimNT, DimDT]:
        points_transformed = self.affine_transform.predict(points)
        J_gamma = self.affine_transform.jacobian(points)
        if self.nonlinear_transform is not None:
            J_psi = JacobianArray[DimNT, DimDT](
                self.nonlinear_transform.jacobian(points_transformed)
            )
            if self.use_residuals:
                J_phi = J_gamma + J_psi @ J_gamma
            else:
                J_phi = J_psi @ J_gamma
            return JacobianArray[DimNT, DimDT](J_phi)
        else:
            return JacobianArray[DimNT, DimDT](J_gamma)

    def transport_positions(
        self, positions: PointsArray[DimNT, DimDT], /
    ) -> PointsArray[DimNT, DimDT]:
        return self.predict(positions)

    def transport_velocities(
        self,
        positions: PointsArray[DimNT, DimDT],
        velocities: PointsArray[DimNT, DimDT],
    ) -> PointsArray[DimNT, DimDT]:
        J_phi = self.jacobian(positions)
        velocities_transported = PointsArray[DimNT, DimDT](
            np.einsum("nij,nj->ni", J_phi, velocities)
        )
        return velocities_transported


class PolicyTransportation2D(
    Generic[NonLinearTransformT, DimNT],
    PolicyTransportation[NonLinearTransformT, DimNT, TWO],
): ...


class PolicyTransportation3D(
    Generic[NonLinearTransformT, DimNT],
    PolicyTransportation[NonLinearTransformT, DimNT, THREE],
): ...

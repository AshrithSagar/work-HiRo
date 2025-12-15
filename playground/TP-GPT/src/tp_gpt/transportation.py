"""
Policy transportation
=======
src/tp_gpt/transportation.py
"""

from typing import Generic, Optional, TypeVar

import numpy as np
from typed_numpy.helpers import (
    Array2x2,
    Array3x3,
    ArrayN,
    ArrayNx2,
    ArrayNx2x2,
    ArrayNx3,
    ArrayNx3x3,
)

from tp_gpt.transforms import AffineTransform, Transform
from tp_gpt.typings import JacobianT, PointsT, PointT, RotationT

NonLinearTransformT = TypeVar("NonLinearTransformT", bound=Transform)


class PolicyTransportation(
    Generic[NonLinearTransformT, PointsT, PointT, RotationT, JacobianT],
    Transform[PointsT, JacobianT],
):
    PointsClass: type[PointsT]
    PointClass: type[PointT]
    RotationClass: type[RotationT]
    JacobianClass: type[JacobianT]

    def __init__(
        self,
        nonlinear_transform: Optional[NonLinearTransformT] = None,
        affine_transform: AffineTransform[
            PointT, PointsT, RotationT, JacobianT
        ] = AffineTransform(scale=False, rotate=True),
        *,
        use_residuals: bool = True,
    ):
        self.affine_transform = affine_transform
        self.nonlinear_transform = nonlinear_transform
        self.use_residuals = use_residuals

    def fit(self, source_points: PointsT, target_points: PointsT, /) -> None:
        self.affine_transform.fit(source_points, target_points)
        source_points_transformed = self.affine_transform.predict(source_points)

        if self.nonlinear_transform is not None:
            if self.use_residuals:
                residuals = self.PointsClass(target_points - source_points_transformed)
                self.nonlinear_transform.fit(source_points_transformed, residuals)
            else:
                self.nonlinear_transform.fit(source_points_transformed, target_points)

    def predict(self, points: PointsT, /) -> PointsT:
        points_transformed = self.affine_transform.predict(points)
        if self.nonlinear_transform is not None:
            residuals = self.PointsClass(
                self.nonlinear_transform.predict(points_transformed)
            )
            if self.use_residuals:
                return self.PointsClass(points_transformed + residuals)
            else:
                return residuals
        else:
            return points_transformed

    def jacobian(self, points: PointsT, /) -> JacobianT:
        points_transformed = self.affine_transform.predict(points)
        J_gamma = self.affine_transform.jacobian(points)
        if self.nonlinear_transform is not None:
            J_psi = self.JacobianClass(
                self.nonlinear_transform.jacobian(points_transformed)
            )
            if self.use_residuals:
                J_phi = J_gamma + J_psi @ J_gamma
            else:
                J_phi = J_psi @ J_gamma
            return self.JacobianClass(J_phi)
        else:
            return J_gamma

    def transport_positions(self, positions: PointsT, /) -> PointsT:
        return self.predict(positions)

    def transport_velocities(self, positions: PointsT, velocities: PointsT) -> PointsT:
        J_phi = self.jacobian(positions)
        velocities_transported = self.PointsClass(
            np.einsum("nij,nj->ni", J_phi, velocities)
        )
        return velocities_transported


class PolicyTransportation2D(
    Generic[NonLinearTransformT],
    PolicyTransportation[NonLinearTransformT, ArrayNx2, ArrayN, Array2x2, ArrayNx2x2],
):
    PointsClass = ArrayNx2
    PointClass = ArrayN
    RotationClass = Array2x2
    JacobianClass = ArrayNx2x2


class PolicyTransportation3D(
    Generic[NonLinearTransformT],
    PolicyTransportation[NonLinearTransformT, ArrayNx3, ArrayN, Array3x3, ArrayNx3x3],
):
    PointsClass = ArrayNx3
    PointClass = ArrayN
    RotationClass = Array3x3
    JacobianClass = ArrayNx3x3

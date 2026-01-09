"""
Policy transportation
=======
src/gpto/core/transportation.py
"""

from typing import Any, Generic, Optional, TypeVar

import numpy as np

from gpto.core.typings import (
    DimSpace,
    JacobianSet,
    LearnableEndomorphicMappingProtocol,
    NumPoints,
    PointSet,
    ThreeD,
    TwoD,
)
from gpto.transforms import AffineTransform

NonLinearTransform = TypeVar(
    "NonLinearTransform", bound=LearnableEndomorphicMappingProtocol, default=Any
)


class PolicyTransportation(
    Generic[NonLinearTransform, DimSpace], LearnableEndomorphicMappingProtocol[DimSpace]
):
    def __init__(
        self,
        nonlinear_transform: Optional[NonLinearTransform] = None,
        affine_transform: AffineTransform[DimSpace] = AffineTransform[DimSpace](
            scale=False, rotate=True
        ),
        *,
        use_residuals: bool = True,
    ):
        self.affine_transform = affine_transform
        self.nonlinear_transform = nonlinear_transform
        self.use_residuals = use_residuals

    def fit(
        self,
        source_points: PointSet[NumPoints, DimSpace],
        target_points: PointSet[NumPoints, DimSpace],
        /,
    ) -> None:
        self.affine_transform.fit(source_points, target_points)
        source_points_transformed = self.affine_transform.predict(source_points)

        if self.nonlinear_transform is not None:
            if self.use_residuals:
                residuals = PointSet[NumPoints, DimSpace](
                    target_points - source_points_transformed
                )
                self.nonlinear_transform.fit(source_points_transformed, residuals)
            else:
                self.nonlinear_transform.fit(source_points_transformed, target_points)

    def predict(
        self, points: PointSet[NumPoints, DimSpace], /
    ) -> PointSet[NumPoints, DimSpace]:
        points_transformed = PointSet[NumPoints, DimSpace](
            self.affine_transform.predict(points)
        )
        if self.nonlinear_transform is not None:
            residuals = PointSet[NumPoints, DimSpace](
                self.nonlinear_transform.predict(points_transformed)
            )
            if self.use_residuals:
                points_transported = PointSet[NumPoints, DimSpace](
                    points_transformed + residuals
                )
                return points_transported
            else:
                return residuals
        else:
            return points_transformed

    def jacobian(
        self, points: PointSet[NumPoints, DimSpace], /
    ) -> JacobianSet[NumPoints, DimSpace]:
        points_transformed = self.affine_transform.predict(points)
        J_gamma = self.affine_transform.jacobian(points)
        if self.nonlinear_transform is not None:
            J_psi = JacobianSet[NumPoints, DimSpace](
                self.nonlinear_transform.jacobian(points_transformed)
            )
            if self.use_residuals:
                J_phi = J_gamma + J_psi @ J_gamma
            else:
                J_phi = J_psi @ J_gamma
            return JacobianSet[NumPoints, DimSpace](J_phi)
        else:
            return JacobianSet[NumPoints, DimSpace](J_gamma)

    def transport_positions(
        self, positions: PointSet[NumPoints, DimSpace], /
    ) -> PointSet[NumPoints, DimSpace]:
        return self.predict(positions)

    def transport_velocities(
        self,
        positions: PointSet[NumPoints, DimSpace],
        velocities: PointSet[NumPoints, DimSpace],
    ) -> PointSet[NumPoints, DimSpace]:
        J_phi = self.jacobian(positions)
        velocities_transported = PointSet[NumPoints, DimSpace](
            np.einsum("nij,nj->ni", J_phi, velocities)
        )
        return velocities_transported


class PolicyTransportation2D(
    Generic[NonLinearTransform], PolicyTransportation[NonLinearTransform, TwoD]
): ...


class PolicyTransportation3D(
    Generic[NonLinearTransform], PolicyTransportation[NonLinearTransform, ThreeD]
): ...

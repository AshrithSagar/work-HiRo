"""
Policy transportation
=======
src/tp_gpt/core/transportation.py
"""

from typing import Generic, Optional, TypeVar

import numpy as np

from tp_gpt.core.mappings import (
    EndomorphicMappingCollection,
    JacobianSet,
    LearnableEndomorphicMappingProtocol,
)
from tp_gpt.core.spaces import DimSpace, NumPoints, PointSet, ThreeD, TwoD
from tp_gpt.transforms import AffineTransform

NonLinearTransform = TypeVar(
    "NonLinearTransform", bound=LearnableEndomorphicMappingProtocol
)


class PolicyTransportation(
    Generic[NonLinearTransform, NumPoints, DimSpace],
    EndomorphicMappingCollection[NumPoints, DimSpace],
    LearnableEndomorphicMappingProtocol[NumPoints, DimSpace],
):
    def __init__(
        self,
        nonlinear_transform: Optional[NonLinearTransform] = None,
        affine_transform: AffineTransform = AffineTransform(scale=False, rotate=True),
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
                residuals = self._PointSet(target_points - source_points_transformed)
                self.nonlinear_transform.fit(source_points_transformed, residuals)
            else:
                self.nonlinear_transform.fit(source_points_transformed, target_points)

    def predict(
        self, points: PointSet[NumPoints, DimSpace], /
    ) -> PointSet[NumPoints, DimSpace]:
        points_transformed = self._PointSet(self.affine_transform.predict(points))
        if self.nonlinear_transform is not None:
            residuals = self._PointSet(
                self.nonlinear_transform.predict(points_transformed)
            )
            if self.use_residuals:
                points_transported = self._PointSet(points_transformed + residuals)
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
            J_psi = self._JacobianSet(
                self.nonlinear_transform.jacobian(points_transformed)
            )
            if self.use_residuals:
                J_phi = J_gamma + J_psi @ J_gamma
            else:
                J_phi = J_psi @ J_gamma
            return self._JacobianSet(J_phi)
        else:
            return self._JacobianSet(J_gamma)

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
        velocities_transported = self._PointSet(
            np.einsum("nij,nj->ni", J_phi, velocities)
        )
        return velocities_transported


class PolicyTransportation2D(
    Generic[NonLinearTransform, NumPoints],
    PolicyTransportation[NonLinearTransform, NumPoints, TwoD],
): ...


class PolicyTransportation3D(
    Generic[NonLinearTransform, NumPoints],
    PolicyTransportation[NonLinearTransform, NumPoints, ThreeD],
): ...

"""
Policy transportation
=======
src/tp_gpt/transportation.py
"""

from typing import Generic, Optional, TypeVar, overload

from numpy.typing import NDArray
from typed_numpy.helpers import ArrayNx2, ArrayNx2x2, ArrayNx3, ArrayNx3x3

from tp_gpt.base import Transform
from tp_gpt.models import AffineTransform

NonLinearTransform = TypeVar("NonLinearTransform", bound=Transform)


class PolicyTransportation(Generic[NonLinearTransform], Transform):
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

    @overload
    def fit(self, source_points: ArrayNx2, target_points: ArrayNx2, /) -> None: ...
    @overload
    def fit(self, source_points: ArrayNx3, target_points: ArrayNx3, /) -> None: ...
    @overload
    def fit(self, source_points: NDArray, target_points: NDArray, /) -> None: ...

    def fit(self, source_points: NDArray, target_points: NDArray, /) -> None:
        self.affine_transform.fit(source_points, target_points)
        source_points_transformed = self.affine_transform.predict(source_points)

        if self.nonlinear_transform is not None:
            if self.use_residuals:
                residuals = target_points - source_points_transformed
                self.nonlinear_transform.fit(source_points_transformed, residuals)
            else:
                self.nonlinear_transform.fit(source_points_transformed, target_points)

    @overload
    def predict(self, points: ArrayNx2, /) -> ArrayNx2: ...
    @overload
    def predict(self, points: ArrayNx3, /) -> ArrayNx3: ...
    @overload
    def predict(self, points: NDArray, /) -> NDArray: ...

    def predict(self, points: NDArray, /) -> NDArray:
        points_transformed = self.affine_transform.predict(points)
        if self.nonlinear_transform is not None:
            residuals = self.nonlinear_transform.predict(points_transformed)
            if self.use_residuals:
                return points_transformed + residuals
            else:
                return residuals
        else:
            return points_transformed

    def transport_positions(self, positions: NDArray, /) -> NDArray:
        return self.predict(positions)

    @overload
    def jacobian(self, points: ArrayNx2, /) -> ArrayNx2x2: ...
    @overload
    def jacobian(self, points: ArrayNx3, /) -> ArrayNx3x3: ...
    @overload
    def jacobian(self, points: NDArray, /) -> NDArray: ...

    def jacobian(self, points: NDArray, /) -> NDArray:
        points_transformed = self.affine_transform.predict(points)
        J_gamma = self.affine_transform.jacobian(points)
        if self.nonlinear_transform is not None:
            J_psi = self.nonlinear_transform.jacobian(points_transformed)
            if self.use_residuals:
                J_phi = J_gamma + J_psi @ J_gamma
            else:
                J_phi = J_psi @ J_gamma
            return J_phi
        else:
            return J_gamma

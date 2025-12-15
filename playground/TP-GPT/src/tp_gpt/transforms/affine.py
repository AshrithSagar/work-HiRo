"""
Affine Transform
=======
src/tp_gpt/transforms/affine.py
"""

from typing import Generic

import numpy as np

from tp_gpt.transforms.base import Transform
from tp_gpt.typings import (
    DimT,
    JacobianArray,
    Point,
    PointsArray,
    RotationMatrix,
    ThreeD,
    TwoD,
)


class AffineTransform(Generic[DimT], Transform[DimT]):
    """
    Performs an affine transformation (rotation + scale + translation).
    Kabsch algorithm.
    """

    source_centroid: Point[DimT]
    target_centroid: Point[DimT]
    source_centered: PointsArray[DimT]
    target_centered: PointsArray[DimT]
    rotation_matrix: RotationMatrix[DimT]
    translation: PointsArray[DimT]

    def __init__(self, scale: bool = False, rotate: bool = True) -> None:
        self.do_scale: bool = scale
        self.do_rotation: bool = rotate

        self.scale: float = 1.0

    def fit(
        self, source_points: PointsArray[DimT], target_points: PointsArray[DimT], /
    ) -> None:
        assert len(source_points) == len(target_points)
        dim: int = source_points.shape[1]

        self.source_centroid = Point[DimT](np.mean(source_points, axis=0))
        self.target_centroid = Point[DimT](np.mean(target_points, axis=0))

        self.source_centered = PointsArray[DimT](source_points - self.source_centroid)
        self.target_centered = PointsArray[DimT](target_points - self.target_centroid)

        H = RotationMatrix[DimT](np.dot(self.source_centered.T, self.target_centered))
        rank_H: int = np.linalg.matrix_rank(H)

        # Rotation
        if not self.do_rotation or rank_H < dim:
            self.rotation_matrix = RotationMatrix[DimT](np.eye(dim))
        else:
            U, _S, Vt = np.linalg.svd(H)
            U, Vt = RotationMatrix[DimT](U), RotationMatrix[DimT](Vt)
            V = Vt.T

            self.rotation_matrix = RotationMatrix[DimT](V @ U.T)
            if np.linalg.det(self.rotation_matrix) < 0:
                V[:, -1] *= -1
                self.rotation_matrix = RotationMatrix[DimT](V @ U.T)

        # Scale
        if self.do_scale:
            source_rotated = PointsArray[DimT](
                self.source_centered @ self.rotation_matrix.T
            )
            self.scale = float(
                np.sum(source_rotated * self.target_centered)
                / np.sum(source_rotated**2)
            )

        # Translation
        self.translation = PointsArray[DimT](
            self.target_centroid - self.source_centroid
        )

    def predict(self, points: PointsArray[DimT], /) -> PointsArray[DimT]:
        points_transported = PointsArray[DimT](
            self.scale * (points - self.source_centroid) @ self.rotation_matrix.T
            + self.target_centroid
        )
        return points_transported

    def inverse(self, points: PointsArray[DimT], /) -> PointsArray[DimT]:
        points_inverse = PointsArray[DimT](
            (1 / self.scale) * (points - self.target_centroid) @ self.rotation_matrix
            + self.source_centroid
        )
        return points_inverse

    def jacobian(self, points: PointsArray[DimT], /) -> JacobianArray[DimT]:
        n_points = points.shape[0]
        jacobian = JacobianArray[DimT](
            np.tile(self.scale * self.rotation_matrix, (n_points, 1, 1))
        )
        return jacobian


class AffineTransform2D(AffineTransform[TwoD]): ...


class AffineTransform3D(AffineTransform[ThreeD]): ...

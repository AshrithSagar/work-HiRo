"""
Affine Transform
=======
src/tp_gpt/transforms/affine.py
"""

from typing import Generic

import numpy as np
from typed_numpy.helpers import (
    Array2,
    Array2x2,
    Array3,
    Array3x3,
    ArrayNx2,
    ArrayNx2x2,
    ArrayNx3,
    ArrayNx3x3,
)

from tp_gpt.transforms.base import Transform
from tp_gpt.typings import JacobianT, PointsT, PointT, RotationT


class AffineTransform(
    Generic[PointT, PointsT, RotationT, JacobianT],
    Transform[PointsT, JacobianT],
):
    """
    Performs an affine transformation (rotation + scale + translation).
    Kabsch algorithm.
    """

    PointClass: type[PointT]
    PointsClass: type[PointsT]
    RotationClass: type[RotationT]
    JacobianClass: type[JacobianT]

    def __init__(self, scale: bool = False, rotate: bool = True) -> None:
        self.do_scale: bool = scale
        self.do_rotation: bool = rotate

        self.scale: float = 1.0

    def fit(self, source_points: PointsT, target_points: PointsT, /) -> None:
        assert len(source_points) == len(target_points)
        dim: int = source_points.shape[1]

        self.S_centroid = self.PointClass(np.mean(source_points, axis=0))
        self.T_centroid = self.PointClass(np.mean(target_points, axis=0))

        self.source_points_centered = self.PointsClass(source_points - self.S_centroid)
        self.target_points_centered = self.PointsClass(target_points - self.T_centroid)

        H = self.RotationClass(
            np.dot(self.source_points_centered.T, self.target_points_centered)
        )
        rank_H: int = np.linalg.matrix_rank(H)

        # Rotation
        if not self.do_rotation or rank_H < dim:
            self.rotation_matrix = self.RotationClass(np.eye(dim))
        else:
            U, _S, Vt = np.linalg.svd(H)
            U, Vt = self.RotationClass(U), self.RotationClass(Vt)
            V = Vt.T

            self.rotation_matrix = self.RotationClass(V @ U.T)
            if np.linalg.det(self.rotation_matrix) < 0:
                V[:, -1] *= -1
                self.rotation_matrix = self.RotationClass(V @ U.T)

        # Scale
        if self.do_scale:
            source_rotated = self.PointsClass(
                self.source_points_centered @ self.rotation_matrix.T
            )
            self.scale = float(
                np.sum(source_rotated * self.target_points_centered)
                / np.sum(source_rotated**2)
            )

        # Translation
        self.translation = self.PointClass(self.T_centroid - self.S_centroid)

    def predict(self, points: PointsT, /) -> PointsT:
        points_transported = self.PointsClass(
            self.scale * (points - self.S_centroid) @ self.rotation_matrix.T
            + self.T_centroid
        )
        return points_transported

    def inverse(self, points: PointsT, /) -> PointsT:
        points_inverse = self.PointsClass(
            (1 / self.scale) * (points - self.T_centroid) @ self.rotation_matrix
            + self.S_centroid
        )
        return points_inverse

    def jacobian(self, points: PointsT, /) -> JacobianT:
        n_points = points.shape[0]
        jacobian = self.JacobianClass(
            np.tile(self.scale * self.rotation_matrix, (n_points, 1, 1))
        )
        return jacobian


class AffineTransform2D(AffineTransform[Array2, ArrayNx2, Array2x2, ArrayNx2x2]):
    PointClass = Array2
    PointsClass = ArrayNx2
    RotationClass = Array2x2
    JacobianClass = ArrayNx2x2


class AffineTransform3D(AffineTransform[Array3, ArrayNx3, Array3x3, ArrayNx3x3]):
    PointClass = Array3
    PointsClass = ArrayNx3
    RotationClass = Array3x3
    JacobianClass = ArrayNx3x3

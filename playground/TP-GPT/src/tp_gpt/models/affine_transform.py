"""
Affine Transform
=======
src/tp_gpt/models/affine_transform.py
"""

from typing import overload

import numpy as np
from numpy.typing import NDArray
from typed_numpy.helpers import (
    Array2x2,
    ArrayN,
    ArrayNx2,
    ArrayNx2x2,
    ArrayNx3,
    ArrayNx3x3,
)

from tp_gpt.base import Transform


class AffineTransform(Transform):
    """
    Performs an affine transformation (rotation + scale + translation).
    Kabsch algorithm.
    """

    def __init__(self, scale: bool = False, rotate: bool = True) -> None:
        self.do_scale: bool = scale
        self.do_rotation: bool = rotate

        self.scale: float = 1.0

    @overload
    def fit(self, source_points: ArrayNx2, target_points: ArrayNx2, /) -> None: ...
    @overload
    def fit(self, source_points: NDArray, target_points: NDArray, /) -> None: ...

    def fit(self, source_points: NDArray, target_points: NDArray, /) -> None:
        # [TODO] Support 3D
        source_points = ArrayNx2(source_points)
        target_points = ArrayNx2(target_points)
        assert len(source_points) == len(target_points)
        dim: int = source_points.shape[1]

        self.S_centroid = ArrayN(np.mean(source_points, axis=0))
        self.T_centroid = ArrayN(np.mean(target_points, axis=0))

        self.source_points_centered = ArrayNx2(source_points - self.S_centroid)
        self.target_points_centered = ArrayNx2(target_points - self.T_centroid)

        H: Array2x2 = np.dot(
            np.transpose(self.source_points_centered), self.target_points_centered
        )
        rank_H: int = np.linalg.matrix_rank(H)

        # Rotation
        if not self.do_rotation or rank_H < dim:
            self.rotation_matrix = Array2x2(np.eye(dim))
        else:
            U, _S, Vt = np.linalg.svd(H)
            U, Vt = Array2x2(U), Array2x2(Vt)
            V = Vt.T

            self.rotation_matrix = Array2x2(V @ U.T)
            if np.linalg.det(self.rotation_matrix) < 0:
                V[:, -1] *= -1
                self.rotation_matrix = Array2x2(V @ U.T)

        # Scale
        if self.do_scale:
            source_rotated = ArrayNx2(
                np.transpose(
                    self.rotation_matrix @ np.transpose((self.source_points_centered))
                )
            )
            self.scale = float(
                np.sum(source_rotated * self.target_points_centered)
                / np.sum(source_rotated**2)
            )

        # Translation
        self.translation = ArrayN(self.T_centroid - self.S_centroid)

    @overload
    def predict(self, points: ArrayNx2, /) -> ArrayNx2: ...
    @overload
    def predict(self, points: NDArray, /) -> NDArray: ...

    def predict(self, points: NDArray, /) -> NDArray:
        # [TODO] Support 3D
        points = ArrayNx2(points)
        RT = Array2x2(np.transpose(self.rotation_matrix))
        points_transported = ArrayNx2(
            self.scale * (points - self.S_centroid) @ RT + self.T_centroid
        )
        return points_transported

    def inverse(self, points: NDArray, /) -> NDArray:
        # [TODO] Support 3D
        points = ArrayNx2(points)
        R = self.rotation_matrix
        points_inverse = ArrayNx2(
            (1 / self.scale) * (points - self.T_centroid) @ R + self.S_centroid
        )
        return points_inverse

    @overload
    def jacobian(self, points: ArrayNx2, /) -> ArrayNx2x2: ...
    @overload
    def jacobian(self, points: ArrayNx3, /) -> ArrayNx3x3: ...
    @overload
    def jacobian(self, points: NDArray, /) -> NDArray: ...

    def jacobian(self, points: NDArray, /) -> NDArray:
        # [TODO] Support 3D
        points = ArrayNx2(points)
        n_points = points.shape[0]
        J = np.tile(self.scale * self.rotation_matrix, (n_points, 1, 1))
        return J  # Shape: (n_points, 2, 2)

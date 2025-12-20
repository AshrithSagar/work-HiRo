"""
Affine Transform
=======
src/tp_gpt/transforms/affine.py
"""

import numpy as np

from tp_gpt.core.mappings import (
    EndomorphicMappingCollection,
    JacobianSet,
    LearnableEndomorphicMappingProtocol,
)
from tp_gpt.core.spaces import (
    DimSpace,
    NumPoints,
    Point,
    PointSet,
    RotationMatrix,
    ThreeD,
    TwoD,
)


class AffineTransform(
    EndomorphicMappingCollection[NumPoints, DimSpace],
    LearnableEndomorphicMappingProtocol[NumPoints, DimSpace],
):
    """
    Performs an affine transformation (rotation + scale + translation).
    Kabsch algorithm.
    """

    source_centroid: Point[DimSpace]
    target_centroid: Point[DimSpace]
    source_centered: PointSet[NumPoints, DimSpace]
    target_centered: PointSet[NumPoints, DimSpace]
    rotation_matrix: RotationMatrix[DimSpace]
    translation: Point[DimSpace]

    def __init__(self, scale: bool = False, rotate: bool = True) -> None:
        self.do_scale: bool = scale
        self.do_rotation: bool = rotate

        self.scale: float = 1.0

    def fit(
        self,
        source_points: PointSet[NumPoints, DimSpace],
        target_points: PointSet[NumPoints, DimSpace],
        /,
    ) -> None:
        assert len(source_points) == len(target_points)
        dim: int = source_points.shape[1]

        self.source_centroid = self._Point(np.mean(source_points, axis=0))
        self.target_centroid = self._Point(np.mean(target_points, axis=0))

        self.source_centered = self._PointSet(source_points - self.source_centroid)
        self.target_centered = self._PointSet(target_points - self.target_centroid)

        H = self._RotationMatrix(np.dot(self.source_centered.T, self.target_centered))
        rank_H: int = np.linalg.matrix_rank(H)

        # Rotation
        if not self.do_rotation or rank_H < dim:
            self.rotation_matrix = self._RotationMatrix(np.eye(dim))
        else:
            U, _S, Vt = np.linalg.svd(H)
            U, Vt = self._RotationMatrix(U), self._RotationMatrix(Vt)
            V = Vt.T

            self.rotation_matrix = self._RotationMatrix(V @ U.T)
            if np.linalg.det(self.rotation_matrix) < 0:
                V[:, -1] *= -1
                self.rotation_matrix = self._RotationMatrix(V @ U.T)

        # Scale
        if self.do_scale:
            source_rotated = self._PointSet(
                self.source_centered @ self.rotation_matrix.T
            )
            self.scale = float(
                np.sum(source_rotated * self.target_centered)
                / np.sum(source_rotated**2)
            )

        # Translation
        self.translation = self._Point(self.target_centroid - self.source_centroid)

    def predict(
        self, points: PointSet[NumPoints, DimSpace], /
    ) -> PointSet[NumPoints, DimSpace]:
        points_transported = self._PointSet(
            self.scale * (points - self.source_centroid) @ self.rotation_matrix.T
            + self.target_centroid
        )
        return points_transported

    def inverse(
        self, points: PointSet[NumPoints, DimSpace], /
    ) -> PointSet[NumPoints, DimSpace]:
        points_inverse = self._PointSet(
            (1 / self.scale) * (points - self.target_centroid) @ self.rotation_matrix
            + self.source_centroid
        )
        return points_inverse

    def jacobian(
        self, points: PointSet[NumPoints, DimSpace], /
    ) -> JacobianSet[NumPoints, DimSpace]:
        n_points = points.shape[0]
        jacobian = self._JacobianSet(
            np.tile(self.scale * self.rotation_matrix, (n_points, 1, 1))
        )
        return jacobian


class AffineTransform2D(AffineTransform[NumPoints, TwoD]): ...


class AffineTransform3D(AffineTransform[NumPoints, ThreeD]): ...

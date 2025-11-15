"""
Base
=======
src/tp_gpt/base.py
"""

from typing import Any, Callable, Literal

import numpy as np
import numpy.typing as npt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel

FloatNDArray = npt.NDArray[np.double]
"""A type alias for a numpy NDArray of floats."""


class AffineTransform:
    """Performs an affine transformation (rotation + scale + translation)."""

    def __init__(self, do_scale: bool = False, do_rotation: bool = True):
        self.do_scale: bool = do_scale
        self.do_rotation: bool = do_rotation

        self.scale: float = 1.0

    def fit(self, source_points: FloatNDArray, target_points: FloatNDArray):
        assert len(source_points) == len(target_points)
        dim: int = source_points.shape[1]

        self.S_centroid: FloatNDArray = np.mean(source_points, axis=0)
        self.T_centroid: FloatNDArray = np.mean(target_points, axis=0)

        self.source_points_centered = source_points - self.S_centroid
        self.target_points_centered = target_points - self.T_centroid

        H: FloatNDArray = np.dot(
            np.transpose(self.source_points_centered), self.target_points_centered
        )
        rank_H: int = np.linalg.matrix_rank(H)

        # Rotation
        if not self.do_rotation or rank_H < dim:
            self.rotation_matrix = np.eye(dim)
        else:
            U: FloatNDArray
            V: FloatNDArray
            U, _S, Vt = np.linalg.svd(H)
            V = Vt.T

            self.rotation_matrix = V @ U.T
            if np.linalg.det(self.rotation_matrix) < 0:
                V[:, -1] *= -1
                self.rotation_matrix = V @ U.T

        # Scale
        if self.do_scale:
            source_rotated = np.transpose(
                self.rotation_matrix @ np.transpose((self.source_points_centered))
            )
            self.scale = np.sum(source_rotated * self.target_points_centered) / np.sum(
                source_rotated**2
            )

        # Translation
        self.translation = self.T_centroid - self.S_centroid

    def predict(self, x: FloatNDArray) -> FloatNDArray:
        transported_x = (
            self.scale * (x - self.S_centroid) @ np.transpose(self.rotation_matrix)
            + self.T_centroid
        )
        return transported_x


class GaussianProcess:
    """Wrapper for `scikit-learn`'s `GaussianProcessRegressor`."""

    def __init__(
        self,
        kernel: Kernel,
        alpha: float = 1e-10,
        optimizer: Callable[..., Any]
        | Literal["fmin_l_bfgs_b"]
        | None = "fmin_l_bfgs_b",
        n_restarts_optimizer: int = 5,
        n_targets: int | None = None,
    ):
        self.kernel = kernel
        self.alpha = alpha
        self.is_residual: bool = True

        if optimizer is not None:
            self.gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=alpha,
                optimizer=optimizer,
                n_restarts_optimizer=n_restarts_optimizer,
            )
        else:
            self.gp = GaussianProcessRegressor(
                kernel=kernel, alpha=alpha, optimizer=optimizer
            )

    def fit(self, X: FloatNDArray, Y: FloatNDArray):
        self.X = X
        self.Y = Y

        self.n_features = np.shape(self.X)[1]
        self.n_samples = np.shape(self.X)[0]
        self.n_outputs = np.shape(self.Y)[1]

        self.gp.fit(self.X, self.Y)
        self.kernel = self.gp.kernel_

    def predict(
        self, x: FloatNDArray, return_std: bool = False, return_cov: bool = False
    ):
        return self.gp.predict(x, return_std=return_std, return_cov=return_cov)

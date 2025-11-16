"""
Base
=======
src/tp_gpt/base.py
"""

from typing import Any, Callable, Literal, NoReturn, overload

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel

from tp_gpt.typings import Array1D, Array2D, Array3D, Array4D


class AffineTransform:
    """Performs an affine transformation (rotation + scale + translation)."""

    def __init__(self, scale: bool = False, rotate: bool = True):
        self.do_scale: bool = scale
        self.do_rotation: bool = rotate

        self.scale: float = 1.0

    def fit(self, source_points: Array2D, target_points: Array2D):
        assert len(source_points) == len(target_points)
        dim: int = source_points.shape[1]

        self.S_centroid: Array1D = np.mean(source_points, axis=0)
        self.T_centroid: Array1D = np.mean(target_points, axis=0)

        self.source_points_centered: Array2D = np.asarray(
            source_points - self.S_centroid, dtype=np.double
        )
        self.target_points_centered: Array2D = np.asarray(
            target_points - self.T_centroid, dtype=np.double
        )

        H: Array2D = np.dot(
            np.transpose(self.source_points_centered), self.target_points_centered
        )
        rank_H: int = np.linalg.matrix_rank(H)

        # Rotation
        if not self.do_rotation or rank_H < dim:
            self.rotation_matrix: Array2D = np.eye(dim, dtype=np.double)
        else:
            U: Array2D
            Vt: Array2D
            U, _S, Vt = np.linalg.svd(H)
            V = Vt.T

            self.rotation_matrix: Array2D = np.asarray(V @ U.T, dtype=np.double)
            if np.linalg.det(self.rotation_matrix) < 0:
                V[:, -1] *= -1
                self.rotation_matrix = np.asarray(V @ U.T, dtype=np.double)

        # Scale
        if self.do_scale:
            source_rotated: Array2D = np.asarray(
                np.transpose(
                    self.rotation_matrix @ np.transpose((self.source_points_centered))
                ),
                dtype=np.double,
            )
            self.scale: float = np.sum(
                source_rotated * self.target_points_centered
            ) / np.sum(source_rotated**2)

        # Translation
        self.translation: Array1D = np.asarray(
            self.T_centroid - self.S_centroid, dtype=np.double
        )

    def predict(self, x: Array2D) -> Array2D:
        transported_x: Array2D = np.asarray(
            self.scale * (x - self.S_centroid) @ np.transpose(self.rotation_matrix)
            + self.T_centroid,
            dtype=np.double,
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
        self.kernel: Kernel = kernel
        self.alpha: float = alpha
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

    @overload
    def fit(self, X: Array2D, Y: Array2D): ...
    @overload
    def fit(self, X: Array3D, Y: Array3D): ...
    def fit(self, X: Array2D | Array3D, Y: Array2D | Array3D):
        self.X = X
        self.Y = Y

        self.n_samples = np.shape(self.X)[0]
        self.n_features = np.shape(self.X)[1]
        self.n_outputs = np.shape(self.Y)[1]

        self.gp.fit(self.X, self.Y)
        self.kernel = self.gp.kernel_

    @overload
    def predict(
        self,
        x: Array2D,
        *,
        return_std: Literal[False] = False,
        return_cov: Literal[False] = False,
    ) -> Array2D: ...
    @overload
    def predict(
        self,
        x: Array3D,
        *,
        return_std: Literal[False] = False,
        return_cov: Literal[False] = False,
    ) -> Array3D: ...
    @overload
    def predict(
        self,
        x: Array2D,
        *,
        return_std: Literal[True],
        return_cov: Literal[False] = False,
    ) -> tuple[Array2D, Array2D]: ...
    @overload
    def predict(
        self,
        x: Array3D,
        *,
        return_std: Literal[True],
        return_cov: Literal[False] = False,
    ) -> tuple[Array3D, Array3D]: ...
    @overload
    def predict(
        self,
        x: Array2D,
        *,
        return_std: Literal[False] = False,
        return_cov: Literal[True],
    ) -> tuple[Array2D, Array3D]: ...
    @overload
    def predict(
        self,
        x: Array3D,
        *,
        return_std: Literal[False] = False,
        return_cov: Literal[True],
    ) -> tuple[Array3D, Array4D]: ...
    @overload
    def predict(
        self, x, *, return_std: Literal[True], return_cov: Literal[True]
    ) -> NoReturn: ...
    def predict(
        self,
        x: Array2D | Array3D,
        *,
        return_std: bool = False,
        return_cov: bool = False,
    ):
        return self.gp.predict(x, return_std=return_std, return_cov=return_cov)

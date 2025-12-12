"""
Gaussian Process Transform
=======
src/tp_gpt/models/gaussian_process.py
"""

from typing import Any, Callable, Literal, NoReturn, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel
from typed_numpy.helpers import Array2D, Array3D, Array4D

from tp_gpt.base import Transform


class GaussianProcess(Transform):
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
    def fit(self, X: Array2D, Y: Array2D, /) -> None: ...
    @overload
    def fit(self, X: Array3D, Y: Array3D, /) -> None: ...
    @overload
    def fit(self, X: NDArray, Y: NDArray, /) -> None: ...

    def fit(self, X: NDArray, Y: NDArray, /) -> None:
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
        /,
        *,
        return_std: Literal[False] = False,
        return_cov: Literal[False] = False,
    ) -> Array2D: ...
    @overload
    def predict(
        self,
        x: Array3D,
        /,
        *,
        return_std: Literal[False] = False,
        return_cov: Literal[False] = False,
    ) -> Array3D: ...
    @overload
    def predict(
        self,
        x: Array2D,
        /,
        *,
        return_std: Literal[True],
        return_cov: Literal[False] = False,
    ) -> tuple[Array2D, Array2D]: ...
    @overload
    def predict(
        self,
        x: Array3D,
        /,
        *,
        return_std: Literal[True],
        return_cov: Literal[False] = False,
    ) -> tuple[Array3D, Array3D]: ...
    @overload
    def predict(
        self,
        x: Array2D,
        /,
        *,
        return_std: Literal[False] = False,
        return_cov: Literal[True],
    ) -> tuple[Array2D, Array3D]: ...
    @overload
    def predict(
        self,
        x: Array3D,
        /,
        *,
        return_std: Literal[False] = False,
        return_cov: Literal[True],
    ) -> tuple[Array3D, Array4D]: ...
    @overload
    def predict(
        self, x, /, *, return_std: Literal[True], return_cov: Literal[True]
    ) -> NoReturn: ...
    @overload
    def predict(
        self, x: ArrayLike, /, *, return_std: bool = False, return_cov: bool = False
    ) -> NDArray | tuple[NDArray, NDArray] | tuple[NDArray, NDArray, NDArray]: ...

    def predict(
        self, x: ArrayLike, /, *, return_std: bool = False, return_cov: bool = False
    ) -> NDArray | tuple[NDArray, NDArray] | tuple[NDArray, NDArray, NDArray]:
        return self.gp.predict(x, return_std=return_std, return_cov=return_cov)

    def jacobian(self, points: NDArray, /) -> NDArray:
        raise NotImplementedError

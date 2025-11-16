"""
Plots
=======
src/tp_gpt/plots.py
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

from tp_gpt.base import AffineTransform, GaussianProcess
from tp_gpt.obstacle import CircleObstacle
from tp_gpt.typings import Array1D, Array2D, NDArray, Shape1D, dtype


def plot_curve():
    y: Array1D = np.linspace(0, 1, 200, dtype=dtype)
    x: Array1D = np.asarray(2 * y**3 + 1 * y**2, dtype=dtype)
    curve: Array2D = np.column_stack((x, y))

    mid_center: Array1D = np.array([2.0, 0.6], dtype=dtype)

    # Original keypoints
    _S: Array2D = np.array([[x[0], y[0]], mid_center, [x[-1], y[-1]]])

    # Circle boundary around `mid_center`
    obs = CircleObstacle(center=mid_center, radius=0.15, n_points=20)
    circle_pts = obs.boundary_points()

    # Sweep last keypoint
    last_targets: Array1D = np.linspace(0.9, -5.0, 100, dtype=dtype)

    kernel = ConstantKernel(1.0) * RBF(length_scale=0.6) + WhiteKernel(
        noise_level=1e-10
    )

    plt.figure(figsize=(8, 8))
    colors: NDArray[Shape1D] = plt.get_cmap("plasma")(
        np.linspace(0, 1, len(last_targets))
    )

    plt.plot(
        curve[:, 0],
        curve[:, 1],
        color="gray",
        linestyle="--",
        zorder=3,
        label="Source curve",
    )
    plt.plot(
        circle_pts[:, 0],
        circle_pts[:, 1],
        "k--",
        zorder=2,
        label="Obstacle keypoints",
    )

    for idx, y_last in enumerate(last_targets):
        # Targets: replace middle keypoint with circle points
        T: Array2D = np.vstack(
            (
                np.atleast_2d(np.array([x[0], y[0]], dtype=dtype)),
                circle_pts,
                np.atleast_2d(np.array([3.0, y_last], dtype=dtype)),
            ),
            dtype=dtype,
        )
        S_ext: Array2D = np.vstack(
            (
                np.atleast_2d(np.array([x[0], y[0]], dtype=dtype)),
                np.tile(mid_center, (len(circle_pts), 1)),
                np.atleast_2d(np.array([x[-1], y[-1]], dtype=dtype)),
            ),
            dtype=dtype,
        )
        aff = AffineTransform(scale=False, rotate=True)
        aff.fit(S_ext, T)
        resid: Array2D = T - aff.predict(S_ext)

        gp = GaussianProcess(kernel=kernel, alpha=1e-10, optimizer=None)
        gp.fit(S_ext, resid)

        def phi_pts(P: Array2D) -> Array2D:
            return aff.predict(P) + gp.predict(P)

        warped = phi_pts(curve)
        plt.plot(
            warped[:, 0],
            warped[:, 1],
            color=colors[idx],
            linewidth=1.4,
            zorder=1,
            label=rf"Last $y={y_last:.2f}$"
            if idx in [0, len(last_targets) - 1]
            else None,
        )

    plt.axis("equal")
    plt.title("Obstacle avoidance")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_curve()

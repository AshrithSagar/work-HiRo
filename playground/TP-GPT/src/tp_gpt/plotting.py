"""
Plots
=======
src/tp_gpt/plotting.py
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

from tp_gpt.base import AffineTransform, GaussianProcess
from tp_gpt.obstacle import CircularObstacle
from tp_gpt.typings import Array1D, Array2D, NDArray, Shape1D, dtype


def plot_single_obstacle():
    y: Array1D = np.linspace(0, 1, 200, dtype=dtype)
    x: Array1D = np.asarray(2 * y**3 + 1 * y**2, dtype=dtype)
    curve: Array2D = np.column_stack((x, y))

    # Circle boundary
    circle_obs = CircularObstacle(center=(2.0, 0.6), radius=0.15, n_points=20)
    circle_pts = circle_obs.boundary_points()

    # Original keypoints
    _S: Array2D = np.array([[x[0], y[0]], circle_obs.center, [x[-1], y[-1]]])

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
    circle_obs.plot(plt.gca(), "k--", zorder=2, label="Obstacle keypoints")

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
                np.tile(circle_obs.center, (len(circle_pts), 1)),
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


def plot_multiple_obstacles():
    y: Array1D = np.linspace(0, 1, 200, dtype=dtype)
    x: Array1D = np.asarray(2 * y**3 + 1 * y**2, dtype=dtype)
    curve: Array2D = np.column_stack((x, y))

    obstacles: list[CircularObstacle] = [
        CircularObstacle(center=(1.5, 0.4), radius=0.1, n_points=20),
        CircularObstacle(center=(2.2, 0.7), radius=0.15, n_points=20),
        CircularObstacle(center=(2.8, 0.3), radius=0.1, n_points=20),
    ]
    n_sweeps: int = 100

    start_pt = np.array([x[0], y[0]], dtype=dtype)
    end_x = x[-1]

    obs_pts = np.vstack([obs.boundary_points() for obs in obstacles])
    obs_centers = np.vstack(
        [np.tile(obs.center, (obs.n_points, 1)) for obs in obstacles]
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    colors: NDArray[Shape1D] = plt.get_cmap("plasma")(np.linspace(0, 1, n_sweeps))

    ax.plot(curve[:, 0], curve[:, 1], "k--", label="Source curve", zorder=3)
    for obs in obstacles:
        obs.plot(ax, "k--", zorder=2)

    kernel = ConstantKernel(1.0) * RBF(length_scale=0.6) + WhiteKernel(
        noise_level=1e-10
    )
    last_targets = np.linspace(1.0, -5.0, n_sweeps, dtype=dtype)
    for i, y_last in enumerate(last_targets):
        # Target array: start -> obstacle boundary -> final point
        T: Array2D = np.vstack(
            (
                np.atleast_2d(start_pt),
                obs_pts,
                np.atleast_2d(np.array([end_x, y_last], dtype=dtype)),
            )
        )

        # Source array: start -> obstacle centers -> original final
        S_ext: Array2D = np.vstack(
            (
                np.atleast_2d(start_pt),
                obs_centers,
                np.atleast_2d(np.array([end_x, y[-1]], dtype=dtype)),
            )
        )

        aff = AffineTransform(scale=False, rotate=True)
        aff.fit(S_ext, T)
        resid = T - aff.predict(S_ext)

        gp = GaussianProcess(kernel=kernel, alpha=1e-10, optimizer=None)
        gp.fit(S_ext, resid)

        def warp(P: Array2D) -> Array2D:
            return aff.predict(P) + gp.predict(P)

        warped = warp(curve)

        ax.plot(
            warped[:, 0],
            warped[:, 1],
            color=colors[i],
            linewidth=1.4,
            zorder=1,
            label=f"y_last={y_last:.2f}" if i in {0, n_sweeps - 1} else None,
        )

    ax.set_aspect("equal")
    ax.set_title("Obstacle avoidance — Multiple obstacles")
    ax.legend()
    fig.tight_layout()


if __name__ == "__main__":
    plot_single_obstacle()
    plot_multiple_obstacles()

    plt.show()

"""
Demo
=========
src/tp_gpt/demo.py
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from typed_numpy.helpers import Array2D, ArrayNx2

from tp_gpt.base import AffineTransform, GaussianProcess
from tp_gpt.curve import Curve
from tp_gpt.obstacle import CircularObstacle


def make_demo_curve(n_points: int = 200) -> Curve:
    ys = np.linspace(0, 1, n_points)
    xs = 2 * ys**3 + ys**2
    return Curve(xs, ys)


def make_demo_end_targets(n_points: int = 100) -> Curve:
    ys = np.linspace(0.9, -5.0, n_points)
    xs = 3.0 * np.ones_like(ys)
    return Curve(xs, ys)


def plot_single_obstacle():
    curve = make_demo_curve()
    end_targets = make_demo_end_targets()

    # Circle boundary
    circle_obs = CircularObstacle(center=(2.0, 0.6), radius=0.15, n_points=20)
    circle_pts = circle_obs.boundary_points()

    # Original keypoints
    _S = ArrayNx2([curve.start_pt, circle_obs.center, curve.end_pt])

    kernel = ConstantKernel(1.0) * RBF(length_scale=0.6) + WhiteKernel(
        noise_level=1e-10
    )

    plt.figure(figsize=(8, 8))
    colors = Array2D(plt.get_cmap("plasma")(np.linspace(0, 1, end_targets.n_points)))

    plt.plot(
        curve.xs,
        curve.ys,
        color="gray",
        linestyle="--",
        zorder=3,
        label="Source curve",
    )
    circle_obs.plot(plt.gca(), "k--", zorder=2, label="Obstacle keypoints")

    for idx, end_pt in enumerate(end_targets.points):
        # Targets: replace middle keypoint with circle points
        target_points = ArrayNx2(np.vstack((curve.start_pt, circle_pts, end_pt)))
        source_points = ArrayNx2(
            np.vstack(
                (
                    curve.start_pt,
                    np.tile(circle_obs.center, (circle_obs.n_points, 1)),
                    curve.end_pt,
                )
            )
        )
        aff = AffineTransform(scale=False, rotate=True)
        aff.fit(source_points, target_points)
        resid = ArrayNx2(target_points - aff.predict(source_points))

        gp = GaussianProcess(kernel=kernel, alpha=1e-10, optimizer=None)
        gp.fit(source_points, resid)

        def warp(P: ArrayNx2) -> Curve:
            return Curve.from_points(aff.predict(P) + gp.predict(P))

        warped = warp(curve.points)
        plt.plot(
            warped.xs,
            warped.ys,
            color=colors[idx],
            linewidth=1.4,
            zorder=1,
            label=rf"EndPt={end_pt}" if idx in {0, end_targets.n_points - 1} else None,
        )

    plt.axis("equal")
    plt.title("Obstacle avoidance")
    plt.legend()
    plt.tight_layout()


def plot_multiple_obstacles():
    curve = make_demo_curve()
    end_targets = make_demo_end_targets()

    obstacles = [
        CircularObstacle(center=(1.5, 0.4), radius=0.15, n_points=20),
        CircularObstacle(center=(2, 0.6), radius=0.15, n_points=20),
        CircularObstacle(center=(2.8, 0.3), radius=0.15, n_points=20),
    ]

    obs_pts = ArrayNx2(np.vstack([obs.boundary_points() for obs in obstacles]))
    obs_centers = ArrayNx2(
        np.vstack([np.tile(obs.center, (obs.n_points, 1)) for obs in obstacles])
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    colors = Array2D(plt.get_cmap("plasma")(np.linspace(0, 1, end_targets.n_points)))

    ax.plot(curve.xs, curve.ys, "k--", label="Source curve", zorder=3)
    for obs in obstacles:
        obs.plot(ax, "k--", zorder=2)

    kernel = ConstantKernel(1.0) * RBF(length_scale=0.6) + WhiteKernel(
        noise_level=1e-10
    )
    for idx, end_pt in enumerate(end_targets.points):
        # Target array: start -> obstacle boundary -> final point
        target_points = ArrayNx2(np.vstack((curve.start_pt, obs_pts, end_pt)))

        # Source array: start -> obstacle centers -> original final
        source_points = ArrayNx2(np.vstack((curve.start_pt, obs_centers, curve.end_pt)))

        aff = AffineTransform(scale=False, rotate=True)
        aff.fit(source_points, target_points)
        resid = ArrayNx2(target_points - aff.predict(source_points))

        gp = GaussianProcess(kernel=kernel, alpha=1e-10, optimizer=None)
        gp.fit(source_points, resid)

        def warp(P: ArrayNx2) -> Curve:
            return Curve.from_points(aff.predict(P) + gp.predict(P))

        warped = warp(curve.points)
        ax.plot(
            warped.xs,
            warped.ys,
            color=colors[idx],
            linewidth=1.4,
            zorder=1,
            label=rf"EndPt={end_pt}" if idx in {0, end_targets.n_points - 1} else None,
        )

    ax.set_aspect("equal")
    ax.set_title("Obstacle avoidance — Multiple obstacles")
    ax.legend()
    fig.tight_layout()


if __name__ == "__main__":
    plot_single_obstacle()
    plot_multiple_obstacles()

    plt.show()

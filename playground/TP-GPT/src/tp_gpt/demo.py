"""
Demo
=======
src/tp_gpt/demo.py
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from typed_numpy._typed.helpers import Array2D, ArrayNx2

from tp_gpt.curve import Curve2D
from tp_gpt.helpers import warp_2D
from tp_gpt.obstacle import CircularObstacle
from tp_gpt.plotting import InteractionManager, InteractiveCircularObstacle


def make_demo_curve_2D(n_points: int = 200) -> Curve2D:
    ys = np.linspace(0.0, 1.0, n_points)
    xs = 2.0 * ys**3 + ys**2
    return Curve2D(xs, ys)


def make_demo_end_targets_2D(n_points: int = 100) -> Curve2D:
    ys = np.linspace(0.9, -5.0, n_points)
    xs = 3.0 * np.ones_like(ys)
    return Curve2D(xs, ys)


def plot_single_obstacle_2D():
    curve = make_demo_curve_2D(n_points=200)
    end_targets = make_demo_end_targets_2D(n_points=100)
    circle_obs = CircularObstacle(center=(2.0, 0.6), radius=0.15, n_theta=20)

    # Original keypoints
    _S = ArrayNx2([curve.start_pt, circle_obs.center, curve.end_pt])

    plt.figure(figsize=(8, 8))
    colors = Array2D(plt.get_cmap("plasma")(np.linspace(0, 1, end_targets.n_points)))

    curve.plot(plt.gca(), color="gray", linestyle="--", zorder=3, label="Source curve")
    circle_obs.plot(plt.gca(), "k--", zorder=2, label="Obstacle keypoints")

    kernel = ConstantKernel(1.0) * RBF(length_scale=0.6) + WhiteKernel(
        noise_level=1e-10
    )
    warped_curves = warp_2D(
        curve,
        end_targets,
        kernel,
        obs_pts=ArrayNx2(circle_obs.boundary_points),
        obs_centers=ArrayNx2(circle_obs.center_tile),
    )
    for idx, warped_curve in enumerate(warped_curves):
        warped_curve.plot(plt.gca(), color=colors[idx], linewidth=1.4, zorder=1)

    plt.axis("equal")
    plt.title("Obstacle avoidance")
    plt.legend()
    plt.tight_layout()


def plot_single_obstacle_interactive_2D():
    curve = make_demo_curve_2D(n_points=200)
    end_targets = make_demo_end_targets_2D(n_points=100)
    circle_obs = InteractiveCircularObstacle(center=(2.0, 0.6), radius=0.15, n_theta=20)

    # Original keypoints
    _S = ArrayNx2([curve.start_pt, circle_obs.center, curve.end_pt])

    kernel = ConstantKernel(1.0) * RBF(length_scale=0.6) + WhiteKernel(
        noise_level=1e-10
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    colors = Array2D(plt.get_cmap("plasma")(np.linspace(0, 1, end_targets.n_points)))

    ax.axis("equal")
    ax.set_title("Obstacle avoidance")
    curve.plot(ax, color="gray", linestyle="--", zorder=3, label="Source curve")
    circle_obs.plot(
        ax, fill=False, zorder=2, ec="k", ls="--", label="Obstacle keypoints"
    )
    warp_lines = [
        ax.plot([], [], color=colors[idx], linewidth=1.4, zorder=1)[0]
        for idx in range(end_targets.n_points)
    ]

    def update_warp(autoscale: bool = True) -> None:
        warped_curves = warp_2D(
            curve,
            end_targets,
            kernel,
            obs_pts=ArrayNx2(circle_obs.boundary_points),
            obs_centers=ArrayNx2(circle_obs.center_tile),
        )
        for line, warped_curve in zip(warp_lines, warped_curves):
            line.set_data(warped_curve.xs, warped_curve.ys)
        if autoscale:
            ax.relim()
            ax.autoscale_view()
        fig.canvas.draw_idle()

    update_warp(autoscale=True)
    _ = InteractionManager(
        fig=fig,
        ax=ax,
        draggables=[circle_obs],
        on_release_callback=update_warp,
        render_during_drag=True,
    )
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_multiple_obstacles_2D():
    curve = make_demo_curve_2D(n_points=200)
    end_targets = make_demo_end_targets_2D(n_points=100)

    obstacles = [
        CircularObstacle(center=(1.5, 0.4), radius=0.15, n_theta=20),
        CircularObstacle(center=(2, 0.6), radius=0.15, n_theta=20),
        CircularObstacle(center=(2.8, 0.3), radius=0.15, n_theta=20),
    ]

    obs_pts = ArrayNx2(np.vstack([obs.boundary_points for obs in obstacles]))
    obs_centers = ArrayNx2(np.vstack([obs.center_tile for obs in obstacles]))

    fig, ax = plt.subplots(figsize=(8, 8))
    colors = Array2D(plt.get_cmap("plasma")(np.linspace(0, 1, end_targets.n_points)))

    curve.plot(ax, "k--", label="Source curve", zorder=3)
    for obs in obstacles:
        obs.plot(ax, "k--", zorder=2)

    kernel = ConstantKernel(1.0) * RBF(length_scale=0.6) + WhiteKernel(
        noise_level=1e-10
    )
    warped_curves = warp_2D(curve, end_targets, kernel, obs_pts, obs_centers)
    for idx, warped_curve in enumerate(warped_curves):
        warped_curve.plot(ax, color=colors[idx], linewidth=1.4, zorder=1)

    ax.set_aspect("equal")
    ax.set_title("Obstacle avoidance — Multiple obstacles")
    ax.legend()
    fig.tight_layout()


def plot_multiple_obstacles_interactive_2D():
    curve = make_demo_curve_2D(n_points=200)
    end_targets = make_demo_end_targets_2D(n_points=100)

    obstacles = [
        InteractiveCircularObstacle(center=(1.5, 0.4), radius=0.15, n_theta=20),
        InteractiveCircularObstacle(center=(2.0, 0.6), radius=0.15, n_theta=20),
        InteractiveCircularObstacle(center=(2.8, 0.3), radius=0.15, n_theta=20),
    ]

    kernel = ConstantKernel(1.0) * RBF(length_scale=0.6) + WhiteKernel(
        noise_level=1e-10
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    colors = Array2D(plt.get_cmap("plasma")(np.linspace(0, 1, end_targets.n_points)))

    ax.axis("equal")
    ax.set_title("Obstacle avoidance — Multiple interactive obstacles")

    curve.plot(ax, "k--", zorder=3, label="Source curve")
    for obs in obstacles:
        obs.plot(ax, fill=False, ec="k", ls="--", zorder=2)

    warp_lines = [
        ax.plot([], [], color=colors[idx], linewidth=1.4, zorder=1)[0]
        for idx in range(end_targets.n_points)
    ]

    def update_warp(autoscale: bool = True) -> None:
        obs_pts = ArrayNx2(np.vstack([obs.boundary_points for obs in obstacles]))
        obs_centers = ArrayNx2(np.vstack([obs.center_tile for obs in obstacles]))
        warped_curves = warp_2D(
            curve,
            end_targets,
            kernel,
            obs_pts=obs_pts,
            obs_centers=obs_centers,
        )
        for line, warped_curve in zip(warp_lines, warped_curves):
            line.set_data(warped_curve.xs, warped_curve.ys)
        if autoscale:
            ax.relim()
            ax.autoscale_view()
        fig.canvas.draw_idle()

    update_warp(autoscale=True)
    _ = InteractionManager(
        fig=fig,
        ax=ax,
        draggables=obstacles,
        on_release_callback=update_warp,
        render_during_drag=True,
    )
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_single_obstacle_2D()
    plot_multiple_obstacles_2D()
    plt.show()

    plot_single_obstacle_interactive_2D()
    plot_multiple_obstacles_interactive_2D()

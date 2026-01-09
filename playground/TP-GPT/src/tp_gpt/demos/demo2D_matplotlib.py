"""
Demo
=======
src/tp_gpt/demo.py
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from typed_numpy._typed.helpers import Array2D

from tp_gpt.core.transportation import PolicyTransportation2D
from tp_gpt.core.typings import Point, TwoD
from tp_gpt.curve import Curve2D
from tp_gpt.obstacle import CircularObstacle
from tp_gpt.plotting.matplotlib import (
    InteractiveCircularObstacle,
    Plot2D,
    PlotInteractive2D,
    PlotSession,
)
from tp_gpt.transforms import AffineTransform2D, GaussianProcessTransform2D
from tp_gpt.warp import ObstacleAvoidanceWarp2D


def make_demo_curve_2D(n_points: int = 200) -> Curve2D:
    ys = np.linspace(0.0, 1.0, n_points)
    xs = 2.0 * ys**3 + ys**2
    return Curve2D.from_xy(xs, ys)


def make_demo_end_targets_2D(n_points: int = 100) -> Curve2D:
    ys = np.linspace(0.9, -5.0, n_points)
    xs = 3.0 * np.ones_like(ys)
    return Curve2D.from_xy(xs, ys)


def make_policy_transportation_2D() -> PolicyTransportation2D:
    aff = AffineTransform2D(scale=False, rotate=True)
    kernel = ConstantKernel(1.0) * RBF(length_scale=0.6) + WhiteKernel(
        noise_level=1e-10
    )
    gp = GaussianProcessTransform2D(kernel=kernel, alpha=1e-10, optimizer=None)
    transport = PolicyTransportation2D(gp, aff)  # type: ignore
    return transport


def plot_single_obstacle_2D() -> None:
    curve = make_demo_curve_2D(n_points=200)
    end_targets = make_demo_end_targets_2D(n_points=100)
    transport = make_policy_transportation_2D()
    circle_obs = CircularObstacle(center=(2.0, 0.6), radius=0.15, n_theta=20)

    ps = PlotSession(title="Obstacle avoidance")
    plot = Plot2D(ax=ps.ax)
    colors = Array2D(plt.get_cmap("plasma")(np.linspace(0, 1, end_targets.n_points)))
    plot.curve(curve, color="gray", linestyle="--", zorder=3, label="Source curve")
    plot.obstacle(circle_obs, "k--", zorder=2, label="Obstacle keypoints")

    for idx, end_pt in enumerate(end_targets.points):
        warper = ObstacleAvoidanceWarp2D(transport, [circle_obs], curve)
        warper.fit(Point[TwoD](end_pt))
        warped_curve = warper.warp_curve()
        plot.curve(warped_curve, color=colors[idx], linewidth=1.4, zorder=1)
    ps.show()


def plot_single_obstacle_interactive_2D() -> None:
    curve = make_demo_curve_2D(n_points=200)
    end_targets = make_demo_end_targets_2D(n_points=100)
    transport = make_policy_transportation_2D()
    circle_obs = InteractiveCircularObstacle(center=(2.0, 0.6), radius=0.15, n_theta=20)

    ps = PlotSession(title="Obstacle avoidance", render_during_drag=True)
    plot = PlotInteractive2D(ax=ps.ax)
    plot.curve(curve, color="gray", linestyle="--", zorder=3, label="Source curve")
    plot.obstacle(
        circle_obs, fill=False, zorder=2, ec="k", ls="--", label="Obstacle keypoints"
    )
    warp_lines = ps.make_lines(
        end_targets.n_points, colormap="plasma", linewidth=1.4, zorder=1
    )

    def update_warp() -> None:
        for line, end_pt in zip(warp_lines, end_targets.points):
            warper = ObstacleAvoidanceWarp2D(transport, [circle_obs], curve)
            warper.fit(Point[TwoD](end_pt))
            warped_curve = warper.warp_curve()
            line.set_data(*warped_curve.components)

    ps.enable_interaction(draggables=[circle_obs], on_update=update_warp)
    ps.show()


def plot_multiple_obstacles_2D() -> None:
    curve = make_demo_curve_2D(n_points=200)
    end_targets = make_demo_end_targets_2D(n_points=100)
    transport = make_policy_transportation_2D()
    obstacles = [
        CircularObstacle(center=(1.5, 0.4), radius=0.15, n_theta=20),
        CircularObstacle(center=(2, 0.6), radius=0.15, n_theta=20),
        CircularObstacle(center=(2.8, 0.3), radius=0.15, n_theta=20),
    ]

    ps = PlotSession(title="Obstacle avoidance — Multiple obstacles")
    plot = Plot2D(ax=ps.ax)
    colors = Array2D(plt.get_cmap("plasma")(np.linspace(0, 1, end_targets.n_points)))
    plot.curve(curve, "k--", label="Source curve", zorder=3)
    for obs in obstacles:
        plot.obstacle(obs, "k--", zorder=2)
    for idx, end_pt in enumerate(end_targets.points):
        warper = ObstacleAvoidanceWarp2D(transport, obstacles, curve)
        warper.fit(Point[TwoD](end_pt))
        warped_curve = warper.warp_curve()
        plot.curve(warped_curve, color=colors[idx], linewidth=1.4, zorder=1)
    ps.show()


def plot_multiple_obstacles_interactive_2D() -> None:
    curve = make_demo_curve_2D(n_points=200)
    end_targets = make_demo_end_targets_2D(n_points=100)
    transport = make_policy_transportation_2D()
    obstacles = [
        InteractiveCircularObstacle(center=(1.5, 0.4), radius=0.15, n_theta=20),
        InteractiveCircularObstacle(center=(2.0, 0.6), radius=0.15, n_theta=20),
        InteractiveCircularObstacle(center=(2.8, 0.3), radius=0.15, n_theta=20),
    ]

    ps = PlotSession(
        title="Obstacle avoidance — Multiple interactive obstacles",
        render_during_drag=True,
    )
    plot = PlotInteractive2D(ax=ps.ax)
    plot.curve(curve, "k--", zorder=3, label="Source curve")
    for obs in obstacles:
        plot.obstacle(obs, fill=False, ec="k", ls="--", zorder=2)
    warp_lines = ps.make_lines(
        end_targets.n_points, colormap="plasma", linewidth=1.4, zorder=1
    )

    def update_warp() -> None:
        for line, end_pt in zip(warp_lines, end_targets.points):
            warper = ObstacleAvoidanceWarp2D(transport, obstacles, curve)
            warper.fit(Point[TwoD](end_pt))
            warped_curve = warper.warp_curve()
            line.set_data(*warped_curve.components)

    ps.enable_interaction(draggables=obstacles, on_update=update_warp)
    ps.show()


if __name__ == "__main__":
    plot_single_obstacle_2D()
    plot_multiple_obstacles_2D()

    plot_single_obstacle_interactive_2D()
    plot_multiple_obstacles_interactive_2D()

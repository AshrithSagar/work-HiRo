"""
Demo3D
=======
src/tp_gpt/demo3D.py
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

from tp_gpt.core.transportation import PolicyTransportation3D
from tp_gpt.core.typings import Point, ThreeD
from tp_gpt.curve import Curve3D
from tp_gpt.obstacle import SphericalObstacle
from tp_gpt.plotting import set_axes_equal
from tp_gpt.transforms import AffineTransform3D, GaussianProcessTransform3D
from tp_gpt.warp import ObstacleAvoidanceWarp3D


def make_demo_curve_3D(n_points: int = 200) -> Curve3D:
    ts = np.linspace(0.0, 1.0, n_points)
    xs = 2.0 * ts**3 + ts**2
    ys = ts
    zs = 0.5 * np.sin(2 * np.pi * ts)
    return Curve3D.from_xyz(xs, ys, zs)


def make_policy_transportation_3D() -> PolicyTransportation3D:
    aff = AffineTransform3D(scale=False, rotate=True)
    kernel = ConstantKernel(1.0) * RBF(length_scale=0.6) + WhiteKernel(
        noise_level=1e-10
    )
    gp = GaussianProcessTransform3D(kernel=kernel, alpha=1e-10, optimizer=None)
    transport = PolicyTransportation3D(gp, aff)  # type: ignore
    return transport


def plot_single_obstacle_3D() -> None:
    curve = make_demo_curve_3D()
    transport = make_policy_transportation_3D()
    obstacle = SphericalObstacle(center=(2.0, 0.6, 0.0), radius=0.1, n_theta=8, n_phi=8)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("Obstacle avoidance 3D")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    curve.plot(ax, color="black", label="Source curve")
    obstacle.plot(
        ax, mode="scatter", color="gray", linewidth=0.1, label="Obstacle keypoints"
    )

    end_pt = Point[ThreeD](curve.end_pt + (-1.0, -1.0, -1.0))
    warper = ObstacleAvoidanceWarp3D(transport, [obstacle], curve)
    warper.fit(end_pt)
    warped = warper.warp_curve()
    warped.plot(ax, color="crimson", label="Warped curve")

    set_axes_equal(ax)
    ax.legend()
    plt.show()


if __name__ == "__main__":
    plot_single_obstacle_3D()

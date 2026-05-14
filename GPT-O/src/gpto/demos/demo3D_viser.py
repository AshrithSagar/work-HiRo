"""
Demo3D Viser
=======
src/gpto/demo3D_viser.py
"""

import time
from typing import cast

import numpy as np
import viser
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

from gpto.core.transportation import PolicyTransportation3D
from gpto.core.typings import Point, ThreeD
from gpto.curve import Curve3D
from gpto.obstacle import SphericalObstacle
from gpto.transforms import AffineTransform3D, GaussianProcessTransform3D
from gpto.warp import ObstacleAvoidanceWarp3D


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
    return PolicyTransportation3D(gp, aff)  # type: ignore


def main() -> None:
    curve = make_demo_curve_3D(n_points=200)
    transport = make_policy_transportation_3D()

    server = viser.ViserServer()
    server.scene.add_spline_catmull_rom(
        "/source_curve", points=curve.points, line_width=5.0
    )

    obstacles = list[SphericalObstacle]()
    obstacle_ctrls = list[viser.TransformControlsHandle]()
    endpt_ctrls = list[viser.TransformControlsHandle]()
    warped_curves = list[Curve3D]()

    ## GUI
    status = server.gui.add_text("Status", initial_value="Ready", disabled=True)
    with server.gui.add_folder("Scene"):
        add_obstacle_btn = server.gui.add_button("Add obstacle")
        add_endpt_btn = server.gui.add_button("Add endpoint")
        compute_btn = server.gui.add_button("Compute warps")

    def sync_obstacles_from_scene() -> None:
        for obs, ctrl in zip(obstacles, obstacle_ctrls):
            obs._center = Point[ThreeD](ctrl.position)

    def get_end_points() -> list[Point[ThreeD]]:
        return [Point[ThreeD](ctrl.position) for ctrl in endpt_ctrls]

    def clear_warped_curves() -> None:
        for i in range(100):
            try:
                server.scene.remove_by_name(f"/warped_curve_{i}")
            except Exception:
                pass

    def update_visualization() -> None:
        clear_warped_curves()
        n = len(warped_curves)
        for i, warped in enumerate(warped_curves):
            t = i / max(n - 1, 1)
            color = (int(50 + 200 * t), int(50), int(255 * (1 - t)))
            server.scene.add_spline_catmull_rom(
                f"/warped_curve_{i}", points=warped.points, color=color
            )

    def compute_warped_curves() -> None:
        try:
            sync_obstacles_from_scene()
            end_points = get_end_points()
            warped_curves.clear()
            for i, end_pt in enumerate(end_points):
                status.value = f"Computing... {i + 1}/{len(end_points)}"
                warper = ObstacleAvoidanceWarp3D(
                    transportation=transport, obstacles=obstacles, curve=curve
                )
                warper.fit(end_pt)
                warped = warper.warp_curve()
                warped = cast(Curve3D, warped)
                warped_curves.append(warped)
            update_visualization()
            status.value = f"Computed {len(warped_curves)} curves"
        except Exception as e:
            status.value = f"Error: {str(e)[:40]}"

    ## Event handlers
    @add_obstacle_btn.on_click
    def _(_):
        idx = len(obstacles)
        center = curve.points[len(curve.points) // 2] + np.array([0.3, 0.0, 0.0])
        obstacle = SphericalObstacle(center=center, radius=0.15, n_theta=8, n_phi=8)

        ctrl = server.scene.add_transform_controls(
            f"/obstacle_{idx}", position=center, scale=0.5, opacity=0.4
        )
        server.scene.add_icosphere(f"/obstacle_{idx}/sphere", radius=obstacle.radius)

        obstacles.append(obstacle)
        obstacle_ctrls.append(ctrl)

    @add_endpt_btn.on_click
    def _(_):
        idx = len(endpt_ctrls)
        pos = curve.end_pt + np.array([0.4, 0.0, 0.0])
        ctrl = server.scene.add_transform_controls(
            f"/endpoint_{idx}", position=pos, scale=0.25
        )
        endpt_ctrls.append(ctrl)

    @compute_btn.on_click
    def _(_):
        compute_warped_curves()

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

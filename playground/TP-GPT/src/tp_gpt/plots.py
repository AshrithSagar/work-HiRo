"""
Plots
=======
src/tp_gpt/plots.py
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

from tp_gpt.base import AffineTransform, FloatNDArray, GaussianProcess


def plot_curve():
    y: FloatNDArray = np.linspace(0, 1, 200)
    x: FloatNDArray = 2 * y**3 + 1 * y**2
    curve: FloatNDArray = np.c_[x, y]

    mid_center: FloatNDArray = np.array([2.0, 0.6])

    # Original keypoints
    _S: FloatNDArray = np.array([[x[0], y[0]], mid_center, [x[-1], y[-1]]])

    # Circle boundary around `mid_center`
    r: float = 0.15
    theta: FloatNDArray = np.linspace(0, 2 * np.pi, 20)
    circle_pts: FloatNDArray = np.c_[
        mid_center[0] + r * np.cos(theta),
        mid_center[1] + r * np.sin(theta),
    ]

    # Sweep last keypoint
    last_targets: FloatNDArray = np.linspace(0.9, -5.0, 100)

    kernel = ConstantKernel(1.0) * RBF(length_scale=0.6) + WhiteKernel(
        noise_level=1e-10
    )

    plt.figure(figsize=(8, 8))
    colors = plt.get_cmap("plasma")(np.linspace(0, 1, len(last_targets)))

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
        T: FloatNDArray = np.vstack([[x[0], y[0]], circle_pts, [3.0, y_last]])
        S_ext: FloatNDArray = np.vstack(
            [[x[0], y[0]], np.tile(mid_center, (len(circle_pts), 1)), [x[-1], y[-1]]]
        )
        aff = AffineTransform(do_scale=False, do_rotation=True)
        aff.fit(S_ext, T)
        resid = T - aff.predict(S_ext)

        gp = GaussianProcess(kernel=kernel, alpha=1e-10, optimizer=None)
        gp.fit(S_ext, resid)

        def phi_pts(P: FloatNDArray) -> FloatNDArray:
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

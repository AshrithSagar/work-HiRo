"""
Helper utils
=======
src/tp_gpt/helpers.py
"""

import numpy as np
from sklearn.gaussian_process.kernels import Kernel
from typed_numpy._typed.helpers import ArrayNx2

from tp_gpt.curve import Curve2D
from tp_gpt.transforms import AffineTransform2D, GaussianProcessTransform2D


def warp_2D(
    curve: Curve2D,
    end_targets: Curve2D,
    gp_kernel: Kernel,
    obs_pts: ArrayNx2,
    obs_centers: ArrayNx2,
):
    for end_pt in end_targets.points:
        target_points = ArrayNx2(np.vstack((curve.start_pt, obs_pts, end_pt)))
        source_points = ArrayNx2(np.vstack((curve.start_pt, obs_centers, curve.end_pt)))
        aff = AffineTransform2D(scale=False, rotate=True)
        aff.fit(source_points, target_points)
        residuals = ArrayNx2(target_points - aff.predict(source_points))

        gp = GaussianProcessTransform2D(kernel=gp_kernel, alpha=1e-10, optimizer=None)
        gp.fit(source_points, residuals)

        def _warp(points: ArrayNx2) -> Curve2D:
            return Curve2D.from_points(aff.predict(points) + gp.predict(points))

        warped = _warp(curve.points)
        yield warped

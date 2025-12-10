"""
Helper utils
=========
src/tp_gpt/helpers.py
"""

import numpy as np
from sklearn.gaussian_process.kernels import Kernel
from typed_numpy.helpers import ArrayNx2

from tp_gpt.base import AffineTransform, GaussianProcess
from tp_gpt.curve import Curve


def warp(
    curve: Curve,
    end_targets: Curve,
    gp_kernel: Kernel,
    obs_pts: ArrayNx2,
    obs_centers: ArrayNx2,
):
    for end_pt in end_targets.points:
        target_points = ArrayNx2(np.vstack((curve.start_pt, obs_pts, end_pt)))
        source_points = ArrayNx2(np.vstack((curve.start_pt, obs_centers, curve.end_pt)))
        aff = AffineTransform(scale=False, rotate=True)
        aff.fit(source_points, target_points)
        residuals = ArrayNx2(target_points - aff.predict(source_points))

        gp = GaussianProcess(kernel=gp_kernel, alpha=1e-10, optimizer=None)
        gp.fit(source_points, residuals)

        def _warp(points: ArrayNx2) -> Curve:
            return Curve.from_points(aff.predict(points) + gp.predict(points))

        warped = _warp(curve.points)
        yield warped

"""
Test utils
=======
"""
# src/pacer/testutils.py

## ── Imports ──────────────────────────────────────────────────────────────────

from typing import Any, Literal

import matplotlib.pyplot as plt
from pyLASAHandwritingDataset import SinglePatternMotion
from torch._prims_common import DeviceLikeType
from typingkit.numpy._typed.helpers import TWO

from pacer import console
from pacer.base import Demonstrations
from pacer.corruptions import DemonstrationCorrupter
from pacer.interactive import InteractiveDataSet
from pacer.lasa import LASADataSet
from pacer.phase import (
    MLPPhaseEstimator,
    NormalisedTimeIndexPhaseEstimator,
    PhaseEstimatorProtocol,
    VelocityPhaseEstimator,
)
from pacer.typings import NumDemos, NumPoints
from pacer.utils import SEED, TORCH_DEVICE

## ── Typings ──────────────────────────────────────────────────────────────────

type DemonstrationsChoice = Literal[
    "FROM_LASA", "CUSTOM_FROM_LOAD", "CUSTOM_FROM_LASA", "CUSTOM_DRAW"
]
type PhaseEstimatorChoice = Literal["MLP", "NORMALISED_TIME_INDEX", "VELOCITY"]

## ── Test Utils ───────────────────────────────────────────────────────────────


def get_demonstrations(
    choice: DemonstrationsChoice = "FROM_LASA",
    *,
    pattern: SinglePatternMotion | None = None,
    filepath: str | None = None,
    use_corruptions: bool = False,
) -> Demonstrations[Any, Any, TWO, TWO]:
    demonstrations: Demonstrations[Any, Any, TWO, TWO]
    match choice:
        case "FROM_LASA":
            assert pattern is not None
            demonstrations = LASADataSet(pattern).to_demonstrations()
        case "CUSTOM_FROM_LOAD":
            assert filepath is not None
            drawer = InteractiveDataSet.load(filepath)
            demonstrations = drawer.to_demonstrations()
        case "CUSTOM_FROM_LASA":
            assert pattern is not None
            drawer = InteractiveDataSet.from_LASA(pattern)
            plt.show(block=True)  # pyright: ignore[reportUnknownMemberType]
            demonstrations = drawer.to_demonstrations()
        case "CUSTOM_DRAW":
            drawer = InteractiveDataSet()
            plt.show(block=True)  # pyright: ignore[reportUnknownMemberType]
            if filepath is not None:
                drawer.save(filepath)
            demonstrations = drawer.to_demonstrations()
        case _:
            raise ValueError

    if use_corruptions:
        corrupter = DemonstrationCorrupter[Any, Any, TWO, TWO](
            demonstrations=demonstrations,
            noise_std=0.2,
            outlier_fraction=0.2,
            outlier_scale=5.0,
            bias_strength=0.2,
        )
        demonstrations = corrupter.inject_corruptions()

    return demonstrations


def get_phase_estimator(
    demonstrations: Demonstrations[NumDemos, NumPoints, TWO, TWO],
    choice: PhaseEstimatorChoice = "MLP",
    *,
    device: DeviceLikeType = TORCH_DEVICE,
    seed: int = SEED,
    phase_hidden_dim: int = 128,
    phase_margin: float = 1.0,  # m
    phase_lr: float = 1e-3,
    phase_epochs: int = 240,
) -> PhaseEstimatorProtocol[NumDemos, NumPoints, TWO, TWO]:
    match choice:
        case "MLP":
            phase_estimator = MLPPhaseEstimator(
                demonstrations, device=device, seed=seed
            )
            phase_loss = phase_estimator.train(
                hidden_dim=phase_hidden_dim,
                margin=phase_margin,  # m
                lr=phase_lr,
                epochs=phase_epochs,
            )
            console.print(f"Phase scorer loss: {phase_loss}")
            return phase_estimator
        case "NORMALISED_TIME_INDEX":
            return NormalisedTimeIndexPhaseEstimator(demonstrations)
        case "VELOCITY":
            return VelocityPhaseEstimator(demonstrations)
    raise ValueError


## ─────────────────────────────────────────────────────────────────────────────

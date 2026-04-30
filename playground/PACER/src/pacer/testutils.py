"""
Test utils
=======
"""
# src/pacer/testutils.py

## ── Imports ──────────────────────────────────────────────────────────────────

from dataclasses import KW_ONLY, dataclass
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
from pyLASAHandwritingDataset import SinglePatternMotion
from rich.pretty import Pretty
from torch._prims_common import DeviceLikeType
from typingkit.numpy._typed.helpers import TWO

from pacer import console
from pacer.base import Demonstrations
from pacer.corruptions import NoisyDemonstrationCorrupter
from pacer.datasets import InteractiveDataSet, LASADataSet, LegacyInteractiveDataSet
from pacer.datasets.interactive.base import make_figure
from pacer.datasets.interactive.plugins import (
    LASALoadPlugin,
    LoadPlugin,
    SavePlugin,
    default_plugins,
)
from pacer.phase.base import PhasesCollection
from pacer.phase.estimation import (
    MLPPhaseEstimator,
    NormalisedTimeIndexPhaseEstimator,
    PathLengthPhaseEstimator,
    PhaseEstimator,
)
from pacer.phase.evaluation import PhaseEvaluationReport
from pacer.typings import DimAction, DimState, NumDemos, NumPoints
from pacer.utils import SEED, TORCH_DEVICE

## ── Typings ──────────────────────────────────────────────────────────────────

type DemonstrationsChoice = Literal[
    "FROM_LASA",
    "CUSTOM_FROM_LOAD",
    "CUSTOM_FROM_LASA",
    "CUSTOM_DRAW",
    "LEGACY_CUSTOM_FROM_LOAD",
    "LEGACY_CUSTOM_FROM_LASA",
    "LEGACY_CUSTOM_DRAW",
]
type PhaseEstimatorChoice = Literal["MLP", "NORMALISED_TIME_INDEX", "PATH_LENGTH"]

## ── Test Utils ───────────────────────────────────────────────────────────────


@dataclass
class DemonstrationLoader:
    choice: DemonstrationsChoice = "FROM_LASA"
    _: KW_ONLY
    pattern: SinglePatternMotion | None = None
    filepath: str | None = None
    use_corruptions: bool = False

    def load(self) -> Demonstrations[Any, Any, TWO, TWO]:
        demonstrations: Demonstrations[Any, Any, TWO, TWO]
        match self.choice:
            case "FROM_LASA":
                assert self.pattern is not None
                demonstrations = LASADataSet(self.pattern).to_demonstrations()
            case "CUSTOM_FROM_LOAD":
                assert self.filepath is not None
                fig, (toolbar_ax, ax) = make_figure()
                plugins = default_plugins(fig, ax, toolbar_ax)
                plugins.append(LoadPlugin(Path(self.filepath)))
                drawer = InteractiveDataSet(
                    plugins=plugins, fig=fig, ax=ax, toolbar_ax=toolbar_ax
                )
                drawer.show()
                demonstrations = drawer.to_demonstrations()
            case "CUSTOM_FROM_LASA":
                assert self.pattern is not None
                fig, (toolbar_ax, ax) = make_figure()
                plugins = default_plugins(fig, ax, toolbar_ax)
                plugins.append(LASALoadPlugin(self.pattern))
                drawer = InteractiveDataSet(
                    plugins=plugins, fig=fig, ax=ax, toolbar_ax=toolbar_ax
                )
                drawer.show()
                demonstrations = drawer.to_demonstrations()
            case "CUSTOM_DRAW":
                fig, (toolbar_ax, ax) = make_figure()
                plugins = default_plugins(fig, ax, toolbar_ax)
                if self.filepath is not None:
                    plugins.append(SavePlugin(Path(self.filepath)))
                    plugins.append(LoadPlugin(Path(self.filepath)))
                drawer = InteractiveDataSet(
                    plugins=plugins, fig=fig, ax=ax, toolbar_ax=toolbar_ax
                )
                drawer.show()
                demonstrations = drawer.to_demonstrations()
            case "LEGACY_CUSTOM_FROM_LOAD":
                assert self.filepath is not None
                legacy_drawer = LegacyInteractiveDataSet.load(self.filepath)
                demonstrations = legacy_drawer.to_demonstrations()
            case "LEGACY_CUSTOM_FROM_LASA":
                assert self.pattern is not None
                legacy_drawer = LegacyInteractiveDataSet.from_LASA(self.pattern)
                plt.show(block=True)  # pyright: ignore[reportUnknownMemberType]
                demonstrations = legacy_drawer.to_demonstrations()
            case "LEGACY_CUSTOM_DRAW":
                legacy_drawer = LegacyInteractiveDataSet()
                plt.show(block=True)  # pyright: ignore[reportUnknownMemberType]
                if self.filepath is not None:
                    legacy_drawer.save(self.filepath)
                demonstrations = legacy_drawer.to_demonstrations()

        if self.use_corruptions:
            corrupter = NoisyDemonstrationCorrupter[Any, Any, TWO, TWO](
                demonstrations=demonstrations,
                noise_std=0.2,
                outlier_fraction=0.2,
                outlier_scale=5.0,
                bias_strength=0.2,
            )
            demonstrations = corrupter.inject_corruptions()

        return demonstrations


def get_phases(
    demonstrations: Demonstrations[NumDemos, NumPoints, DimState, DimAction],
    choice: PhaseEstimatorChoice = "MLP",
    *,
    device: DeviceLikeType = TORCH_DEVICE,
    seed: int = SEED,
    #
    phase_hidden_dim: int = 128,
    phase_margin: float = 1.0,  # m
    phase_lr: float = 1e-3,
    phase_epochs: int = 240,
    #
    evaluate_phases: bool = False,
) -> PhasesCollection[NumDemos, NumPoints]:
    phase_estimator: PhaseEstimator[NumDemos, NumPoints, DimState, DimAction]
    match choice:
        case "MLP":
            phase_estimator = MLPPhaseEstimator(
                demonstrations, device=device, seed=seed
            )
            scorer_loss = phase_estimator.train(
                hidden_dim=phase_hidden_dim,
                margin=phase_margin,  # m
                lr=phase_lr,
                epochs=phase_epochs,
            )
            console.print(f"Phase scorer loss: {scorer_loss}")
        case "NORMALISED_TIME_INDEX":
            phase_estimator = NormalisedTimeIndexPhaseEstimator(demonstrations)
        case "PATH_LENGTH":
            phase_estimator = PathLengthPhaseEstimator(demonstrations)

    phases = phase_estimator.estimate_phases()

    if evaluate_phases:
        report = PhaseEvaluationReport.evaluate(demonstrations, phases)
        console.print(Pretty(report, expand_all=True))

    return phases


## ─────────────────────────────────────────────────────────────────────────────

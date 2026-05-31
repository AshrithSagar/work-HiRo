"""
Phase pipeline
=======
"""
# src/pacer/phase/pipeline.py

## ── Imports ──────────────────────────────────────────────────────────────────

from dataclasses import KW_ONLY, dataclass, field
from typing import Literal

from rich.pretty import Pretty
from torch._prims_common import DeviceLikeType
from typingkit.core import RuntimeGeneric

from pacer import console
from pacer.base import Demonstrations
from pacer.phase.base import PhasesCollection
from pacer.phase.estimation import (
    DTWPhaseEstimator,
    DTWPhaseEstimatorConfig,
    MLPPhaseEstimator,
    MLPPhaseEstimatorConfig,
    NormalisedTimeIndexPhaseEstimator,
    PathLengthPhaseEstimator,
    PhaseEstimator,
)
from pacer.phase.evaluation import PhaseEvaluationReport
from pacer.typings import DimAction, DimState, NumDemos, NumPoints
from pacer.utils import SEED, TORCH_DEVICE

## ── Phase Pipeline ───────────────────────────────────────────────────────────

type PhaseEstimatorChoice = Literal[
    "MLP", "NORMALISED_TIME_INDEX", "PATH_LENGTH", "DTW"
]


@dataclass
class PhasePipelineConfig:
    _: KW_ONLY
    device: DeviceLikeType = TORCH_DEVICE
    seed: int = SEED
    phase_estimator_choice: PhaseEstimatorChoice = "MLP"
    mlp_phase_estimator_config: MLPPhaseEstimatorConfig = field(
        default_factory=MLPPhaseEstimatorConfig
    )
    dtw_phase_estimator_config: DTWPhaseEstimatorConfig = field(
        default_factory=DTWPhaseEstimatorConfig
    )
    evaluate_phases: bool = False


@dataclass
class PhasePipeline(RuntimeGeneric[NumDemos, NumPoints, DimState, DimAction]):
    demonstrations: Demonstrations[NumDemos, NumPoints, DimState, DimAction]
    _: KW_ONLY
    config: PhasePipelineConfig = field(default_factory=PhasePipelineConfig)

    def run(self) -> PhasesCollection[NumDemos, NumPoints]:
        phase_estimator: PhaseEstimator[NumDemos, NumPoints, DimState, DimAction]
        match self.config.phase_estimator_choice:
            case "MLP":
                phase_estimator = MLPPhaseEstimator(
                    self.demonstrations,
                    device=self.config.device,
                    seed=self.config.seed,
                )
                scorer_loss = phase_estimator.train(
                    self.config.mlp_phase_estimator_config
                )
                console.print(f"Phase scorer loss: {scorer_loss}")
            case "NORMALISED_TIME_INDEX":
                phase_estimator = NormalisedTimeIndexPhaseEstimator(self.demonstrations)
            case "PATH_LENGTH":
                phase_estimator = PathLengthPhaseEstimator(self.demonstrations)
            case "DTW":
                phase_estimator = DTWPhaseEstimator(
                    self.demonstrations, self.config.dtw_phase_estimator_config
                )

        phases = phase_estimator.estimate_phases()

        if self.config.evaluate_phases:
            report = PhaseEvaluationReport.evaluate(self.demonstrations, phases)
            console.print(Pretty(report, expand_all=True))

        return phases


## ─────────────────────────────────────────────────────────────────────────────

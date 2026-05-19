"""
Experiment runs
=======
"""
# src/pacer/experiments.py

## ── Imports ──────────────────────────────────────────────────────────────────

from dataclasses import KW_ONLY, dataclass, field
from typing import Any, Generic, Literal, cast
from warnings import deprecated

import matplotlib.pyplot as plt
from pyLASAHandwritingDataset import ALL_SINGLE_PATTERN_MOTIONS, SinglePatternMotion
from torch import Tensor

from pacer import console
from pacer.base import Demonstrations
from pacer.bc import BCTrainConfig, BCTrainer, WeightedBCTrainer
from pacer.datasets.loader import (
    CorruptionsChoice,
    DemonstrationLoader,
    DemonstrationLoaderConfig,
    DemonstrationsChoice,
)
from pacer.pacer import PACER, PACERConfig, PACERResult
from pacer.pacer.pseudolabel import PseudoLabelParams
from pacer.pacer.trust import TrustValueParams
from pacer.phase.estimation import MLPPhaseEstimatorConfig
from pacer.phase.pipeline import PhaseEstimatorChoice, PhasePipelineConfig
from pacer.plotting import PACERVisualisationConfig, PACERVisualiser
from pacer.typings import DimAction, DimState, NumBins, NumDemos, NumPoints

## ── Experiments ──────────────────────────────────────────────────────────────


@dataclass(kw_only=True)
class BCExperimentResult(Generic[NumDemos, NumPoints, DimState, DimAction]):
    demonstrations: Demonstrations[NumDemos, NumPoints, DimState, DimAction]
    bc_policy_loss: Tensor


@dataclass
class BCExperiment(Generic[NumDemos, NumPoints, DimState, DimAction]):
    """BC Policy."""

    demonstrations: Demonstrations[NumDemos, NumPoints, DimState, DimAction]
    _: KW_ONLY
    bc_train_config: BCTrainConfig = field(default_factory=BCTrainConfig)

    def run(self) -> BCExperimentResult[NumDemos, NumPoints, DimState, DimAction]:
        console.rule("[blue]BC policy[/blue]", style="blue")

        # Behavioral cloning
        trainer = BCTrainer(
            states=self.demonstrations.states,
            targets=self.demonstrations.actions,
            device="cpu",
        )
        policy_loss = trainer.train(self.bc_train_config)
        console.print(f"Policy loss: {policy_loss}")

        return BCExperimentResult(
            demonstrations=self.demonstrations,
            bc_policy_loss=policy_loss,
        )


# ──────────────────────────────────────────────────────────────────────────────


@dataclass(kw_only=True)
class PACERBCExperimentResult(
    Generic[NumBins, NumDemos, NumPoints, DimState, DimAction]
):
    demonstrations: Demonstrations[NumDemos, NumPoints, DimState, DimAction]
    pacer_result: PACERResult[NumBins, NumDemos, NumPoints, DimState, DimAction]
    bc_policy_loss: Tensor


@dataclass
class PACERBCExperiment(Generic[NumBins, NumDemos, NumPoints, DimState, DimAction]):
    """PACER + BC Policy."""

    demonstrations: Demonstrations[NumDemos, NumPoints, DimState, DimAction]
    _: KW_ONLY
    pacer_config: PACERConfig[NumBins] = field(default_factory=PACERConfig[NumBins])
    bc_train_config: BCTrainConfig = field(default_factory=BCTrainConfig)

    def run(
        self,
    ) -> PACERBCExperimentResult[NumBins, NumDemos, NumPoints, DimState, DimAction]:
        console.rule(
            f"[blue]PACER[{self.pacer_config.phase_pipeline_config.phase_estimator_choice}_PHASE_ESTIMATION] + BC policy[/blue]",
            style="blue",
        )

        # PACER
        pacer_result = PACER(self.demonstrations, config=self.pacer_config).run()

        # Behavioral cloning
        trainer = WeightedBCTrainer(
            states=pacer_result.pseudo_labels.states or self.demonstrations.states,
            targets=pacer_result.pseudo_labels.actions,
            weights=pacer_result.action_trust_values,
            device="cpu",
        )
        policy_loss = trainer.train(self.bc_train_config)
        console.print(f"Policy loss: {policy_loss}")

        return PACERBCExperimentResult(
            demonstrations=self.demonstrations,
            pacer_result=pacer_result,
            bc_policy_loss=policy_loss,
        )


# ──────────────────────────────────────────────────────────────────────────────


@deprecated(
    "Use BCvsPACERBCExperiment instead, till an improved sweep variation refactor is done."
)
@dataclass(kw_only=True)
class BCvsPACERBCExperimentLegacy(Generic[NumBins]):
    """BC Policy vs. PACER + BC Policy."""

    show_plots: bool = True
    demonstrations_choice: DemonstrationsChoice = "FROM_LASA"
    LASA_pattern: (
        list[SinglePatternMotion] | SinglePatternMotion | Literal["ALL"] | None
    ) = None
    filepath: str | None = None
    corruptions_choice: CorruptionsChoice | None = None
    phase_estimator_choice: (
        list[PhaseEstimatorChoice] | PhaseEstimatorChoice | Literal["ALL"]
    ) = "MLP"
    mlp_phase_estimator_config: MLPPhaseEstimatorConfig = field(
        default_factory=MLPPhaseEstimatorConfig
    )
    evaluate_phases: bool = False
    n_bins: NumBins = cast(NumBins, 96)  # B
    action_trust_value_params: TrustValueParams = field(
        default_factory=TrustValueParams
    )
    action_pseudo_label_params: PseudoLabelParams = field(
        default_factory=PseudoLabelParams
    )
    use_state_labels: bool = False
    state_trust_value_params: TrustValueParams = field(default_factory=TrustValueParams)
    state_pseudo_label_params: PseudoLabelParams = field(
        default_factory=PseudoLabelParams
    )
    bc_train_config: BCTrainConfig = field(default_factory=BCTrainConfig)
    pacer_visualisation_config: PACERVisualisationConfig = field(
        default_factory=PACERVisualisationConfig
    )

    def run(self) -> None:
        # Resolve LASA patterns
        LASA_patterns: list[SinglePatternMotion | None]
        match self.LASA_pattern:
            case list():
                LASA_patterns = list(self.LASA_pattern)
            case str():
                match self.LASA_pattern:
                    case "ALL":
                        LASA_patterns = list(ALL_SINGLE_PATTERN_MOTIONS)
                    case _:
                        LASA_patterns = [self.LASA_pattern]
            case None:
                LASA_patterns = [None]

        # Resolve phase estimator choices
        phase_estimator_choices: list[PhaseEstimatorChoice]
        match self.phase_estimator_choice:
            case list():
                phase_estimator_choices = self.phase_estimator_choice
            case str():
                match self.phase_estimator_choice:
                    case "ALL":
                        phase_estimator_choices = [
                            "MLP",
                            "NORMALISED_TIME_INDEX",
                            "PATH_LENGTH",
                        ]
                    case _:
                        phase_estimator_choices = [self.phase_estimator_choice]

        for LASA_pattern in LASA_patterns:
            match self.demonstrations_choice:
                case "FROM_LASA" | "CUSTOM_FROM_LASA" | "LEGACY_CUSTOM_FROM_LASA":
                    assert LASA_pattern is not None
                    console.rule(
                        f"[bold gold3]LASA Pattern: {LASA_pattern}[/bold gold3]",
                        characters="\u2501",
                        style="gold3",
                    )
                case "CUSTOM_FROM_LOAD" | "LEGACY_CUSTOM_FROM_LOAD":
                    assert self.filepath is not None
                    console.rule(
                        f"[bold gold3]File: {self.filepath}[/bold gold3]",
                        characters="\u2501",
                        style="bold gold3",
                    )
                case "CUSTOM_DRAW" | "LEGACY_CUSTOM_DRAW":
                    console.rule(
                        "[bold gold3]Custom demonstrations[/bold gold3]",
                        characters="\u2501",
                        style="gold3",
                    )

            demonstrations = DemonstrationLoader(
                config=DemonstrationLoaderConfig(
                    choice=self.demonstrations_choice,
                    LASA_pattern=LASA_pattern,
                    filepath=self.filepath,
                    corruptions_choice=self.corruptions_choice,
                )
            ).load()

            BCExperiment(demonstrations, bc_train_config=self.bc_train_config).run()
            for phase_estimator_choice in phase_estimator_choices:
                pacer_bc_result = PACERBCExperiment(
                    demonstrations,
                    pacer_config=PACERConfig(
                        phase_pipeline_config=PhasePipelineConfig(
                            phase_estimator_choice=phase_estimator_choice,
                            mlp_phase_estimator_config=self.mlp_phase_estimator_config,
                            evaluate_phases=self.evaluate_phases,
                        ),
                        n_bins=self.n_bins,
                        action_trust_value_params=self.action_trust_value_params,
                        action_pseudo_label_params=self.action_pseudo_label_params,
                        use_state_labels=self.use_state_labels,
                        state_trust_value_params=self.state_trust_value_params,
                        state_pseudo_label_params=self.state_pseudo_label_params,
                    ),
                    bc_train_config=self.bc_train_config,
                ).run()
                if self.show_plots:
                    PACERVisualiser(
                        demonstrations,
                        pacer_result=pacer_bc_result.pacer_result,
                        config=self.pacer_visualisation_config,
                    ).render()
                    plt.show()  # pyright: ignore[reportUnknownMemberType]

            if self.demonstrations_choice in {"CUSTOM_FROM_LOAD", "CUSTOM_DRAW"}:
                break

        console.rule(characters="\u2501", style="gold3")


# ──────────────────────────────────────────────────────────────────────────────


@dataclass(kw_only=True)
class BCvsPACERBCExperimentResult(
    Generic[NumBins, NumDemos, NumPoints, DimState, DimAction]
):
    demonstrations: Demonstrations[NumDemos, NumPoints, DimState, DimAction]
    bc_result: BCExperimentResult[NumDemos, NumPoints, DimState, DimAction]
    pacer_bc_result: PACERBCExperimentResult[
        NumBins, NumDemos, NumPoints, DimState, DimAction
    ]

    @property
    def pacer_result(
        self,
    ) -> PACERResult[NumBins, NumDemos, NumPoints, DimState, DimAction]:
        return self.pacer_bc_result.pacer_result


@dataclass
class BCvsPACERBCExperiment(Generic[NumBins, NumDemos, NumPoints, DimState, DimAction]):
    """
    BC Policy vs. PACER + BC Policy.\\
    Uses same `BCTrainConfig` for both `BCExperiment` and `PACERBCExperiment`.
    """

    demonstrations: Demonstrations[NumDemos, NumPoints, DimState, DimAction]
    _: KW_ONLY
    pacer_config: PACERConfig[NumBins] = field(default_factory=PACERConfig[NumBins])
    bc_train_config: BCTrainConfig = field(default_factory=BCTrainConfig)

    def run(
        self,
    ) -> BCvsPACERBCExperimentResult[NumBins, Any, Any, DimState, DimAction]:
        bc_result = BCExperiment(
            self.demonstrations,
            bc_train_config=self.bc_train_config,
        ).run()
        pacer_bc_result = PACERBCExperiment(
            self.demonstrations,
            pacer_config=self.pacer_config,
            bc_train_config=self.bc_train_config,
        ).run()

        return BCvsPACERBCExperimentResult(
            demonstrations=self.demonstrations,
            bc_result=bc_result,
            pacer_bc_result=pacer_bc_result,
        )


## ─────────────────────────────────────────────────────────────────────────────

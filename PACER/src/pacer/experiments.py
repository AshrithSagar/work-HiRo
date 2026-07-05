"""
Experiment runs
=======
"""
# src/pacer/experiments.py

## ── Imports ──────────────────────────────────────────────────────────────────

from dataclasses import KW_ONLY, dataclass, field
from typing import Any

from torch import Tensor
from typingkit.core import RuntimeGeneric

from pacer import console
from pacer.base import Demonstrations
from pacer.bc import BCTrainConfig, BCTrainer, WeightedBCTrainer
from pacer.gmm import GMMTrainConfig, WeightedGMMTrainer
from pacer.pacer import PACER, PACERConfig, PACERResult
from pacer.typings import DimAction, DimState, NumBins, NumDemos, NumPoints

## ── Experiments ──────────────────────────────────────────────────────────────


@dataclass(kw_only=True)
class BCExperimentResult(RuntimeGeneric[NumDemos, NumPoints, DimState, DimAction]):
    demonstrations: Demonstrations[NumDemos, NumPoints, DimState, DimAction]
    bc_policy_loss: Tensor


@dataclass
class BCExperiment(RuntimeGeneric[NumDemos, NumPoints, DimState, DimAction]):
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
    RuntimeGeneric[NumBins, NumDemos, NumPoints, DimState, DimAction]
):
    demonstrations: Demonstrations[NumDemos, NumPoints, DimState, DimAction]
    pacer_result: PACERResult[NumBins, NumDemos, NumPoints, DimState, DimAction]
    bc_policy_loss: Tensor


@dataclass
class PACERBCExperiment(
    RuntimeGeneric[NumBins, NumDemos, NumPoints, DimState, DimAction]
):
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


@dataclass(kw_only=True)
class BCvsPACERBCExperimentResult(
    RuntimeGeneric[NumBins, NumDemos, NumPoints, DimState, DimAction]
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
class BCvsPACERBCExperiment(
    RuntimeGeneric[NumBins, NumDemos, NumPoints, DimState, DimAction]
):
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


# ──────────────────────────────────────────────────────────────────────────────


@dataclass(kw_only=True)
class PACERGMMExperimentResult(
    RuntimeGeneric[NumBins, NumDemos, NumPoints, DimState, DimAction]
):
    demonstrations: Demonstrations[NumDemos, NumPoints, DimState, DimAction]
    pacer_result: PACERResult[NumBins, NumDemos, NumPoints, DimState, DimAction]
    gmm_policy_loss: Tensor


@dataclass
class PACERGMMExperiment(
    RuntimeGeneric[NumBins, NumDemos, NumPoints, DimState, DimAction]
):
    """PACER + GMM Policy."""

    demonstrations: Demonstrations[NumDemos, NumPoints, DimState, DimAction]
    _: KW_ONLY
    pacer_config: PACERConfig[NumBins] = field(default_factory=PACERConfig[NumBins])
    gmm_train_config: GMMTrainConfig = field(default_factory=GMMTrainConfig)

    def run(
        self,
    ) -> PACERGMMExperimentResult[NumBins, NumDemos, NumPoints, DimState, DimAction]:
        console.rule(
            f"[magenta]PACER[{self.pacer_config.phase_pipeline_config.phase_estimator_choice}_PHASE_ESTIMATION] + GMM policy[/magenta]",
            style="magenta",
        )

        # PACER
        pacer_result = PACER(self.demonstrations, config=self.pacer_config).run()

        # Weighted Gaussian Mixture Model optimisation
        trainer = WeightedGMMTrainer(
            states=pacer_result.pseudo_labels.states or self.demonstrations.states,
            targets=pacer_result.pseudo_labels.actions,
            weights=pacer_result.action_trust_values,
            device="cpu",
        )
        policy_loss = trainer.train(self.gmm_train_config)
        console.print(f"GMM Policy NLL loss: {policy_loss}")

        return PACERGMMExperimentResult(
            demonstrations=self.demonstrations,
            pacer_result=pacer_result,
            gmm_policy_loss=policy_loss,
        )


## ─────────────────────────────────────────────────────────────────────────────

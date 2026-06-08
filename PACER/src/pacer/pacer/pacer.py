"""
PACER
=======
"""
# src/pacer/pacer/pacer.py

## ── Imports ──────────────────────────────────────────────────────────────────

from dataclasses import KW_ONLY, dataclass, field
from typing import cast

from typingkit.core import RuntimeGeneric

from pacer.base import Demonstrations
from pacer.pacer.base import TrustValuesCollection, ZScoresCollection
from pacer.pacer.binning import Binner, Bins, RibbonTokenConsolidator
from pacer.pacer.consensus import ConsensusConfig
from pacer.pacer.mode import action_mode, state_mode
from pacer.pacer.pseudolabel import PseudoLabelComputer, PseudoLabelParams, PseudoLabels
from pacer.pacer.trust import (
    TrustValueComputationResult,
    TrustValueComputer,
    TrustValueParams,
)
from pacer.phase import PhasePipeline, PhasePipelineConfig, PhasesCollection
from pacer.typings import DimAction, DimState, NumBins, NumDemos, NumPoints

## ── PACER ────────────────────────────────────────────────────────────────────


@dataclass(kw_only=True)
class PACERConfig(RuntimeGeneric[NumBins]):
    phase_pipeline_config: PhasePipelineConfig = field(
        default_factory=PhasePipelineConfig
    )
    n_bins: NumBins = cast(NumBins, 96)  # B
    consensus_config: ConsensusConfig = field(default_factory=ConsensusConfig)
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


@dataclass(kw_only=True)
class PACERResult(RuntimeGeneric[NumBins, NumDemos, NumPoints, DimState, DimAction]):
    phases: PhasesCollection[NumDemos, NumPoints]
    bins: Bins[NumBins, NumDemos, NumPoints, DimState, DimAction]
    action_trust_result: TrustValueComputationResult[NumDemos, NumPoints]
    state_trust_result: TrustValueComputationResult[NumDemos, NumPoints] | None
    pseudo_labels: PseudoLabels[NumDemos, NumPoints, DimState, DimAction]

    @property
    def action_z_scores(self) -> ZScoresCollection[NumDemos, NumPoints]:
        return self.action_trust_result.z_scores

    @property
    def state_z_scores(self) -> ZScoresCollection[NumDemos, NumPoints] | None:
        return (
            self.state_trust_result.z_scores
            if self.state_trust_result is not None
            else None
        )

    @property
    def action_trust_values(self) -> TrustValuesCollection[NumDemos, NumPoints]:
        return self.action_trust_result.trust_values

    @property
    def state_trust_values(self) -> TrustValuesCollection[NumDemos, NumPoints] | None:
        return (
            self.state_trust_result.trust_values
            if self.state_trust_result is not None
            else None
        )


@dataclass
class PACER(RuntimeGeneric[NumBins, NumDemos, NumPoints, DimState, DimAction]):
    demonstrations: Demonstrations[NumDemos, NumPoints, DimState, DimAction]
    _: KW_ONLY
    config: PACERConfig[NumBins] = field(default_factory=PACERConfig[NumBins])

    def run(self) -> PACERResult[NumBins, NumDemos, NumPoints, DimState, DimAction]:
        phases = PhasePipeline(
            self.demonstrations, config=self.config.phase_pipeline_config
        ).run()

        bins = Binner(
            self.demonstrations, phases, n_bins=self.config.n_bins
        ).make_bins()
        bins = RibbonTokenConsolidator(
            bins, consensus_config=self.config.consensus_config
        ).consolidate_ribbon_tokens()

        action_trust_result = TrustValueComputer(
            self.demonstrations, bins, consensus_config=self.config.consensus_config
        ).compute(mode=action_mode(), params=self.config.action_trust_value_params)

        state_trust_result = None
        if self.config.use_state_labels:
            state_trust_result = TrustValueComputer(
                self.demonstrations, bins, consensus_config=self.config.consensus_config
            ).compute(mode=state_mode(), params=self.config.state_trust_value_params)

        pseudo_labels = PseudoLabelComputer(
            self.demonstrations, bins
        ).compute_pseudo_labels(
            action_trust_result.trust_values,
            state_trust_result.trust_values if state_trust_result is not None else None,
            action_params=self.config.action_pseudo_label_params,
            state_params=self.config.state_pseudo_label_params,
        )

        return PACERResult(
            phases=phases,
            bins=bins,
            action_trust_result=action_trust_result,
            state_trust_result=state_trust_result,
            pseudo_labels=pseudo_labels,
        )


## ─────────────────────────────────────────────────────────────────────────────

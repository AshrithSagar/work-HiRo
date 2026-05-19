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
from pacer.pacer.mode import action_mode, state_mode
from pacer.pacer.pseudolabel import PseudoLabelComputer, PseudoLabelParams, PseudoLabels
from pacer.pacer.trust import TrustValueComputer, TrustValueParams
from pacer.phase import PhasePipeline, PhasePipelineConfig, PhasesCollection
from pacer.typings import DimAction, DimState, NumBins, NumDemos, NumPoints

## ── PACER ────────────────────────────────────────────────────────────────────


@dataclass(kw_only=True)
class PACERConfig(RuntimeGeneric[NumBins]):
    phase_pipeline_config: PhasePipelineConfig = field(
        default_factory=PhasePipelineConfig
    )
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


@dataclass(kw_only=True)
class PACERResult(RuntimeGeneric[NumBins, NumDemos, NumPoints, DimState, DimAction]):
    phases: PhasesCollection[NumDemos, NumPoints]
    bins: Bins[NumBins, NumDemos, NumPoints, DimState, DimAction]
    action_z_scores: ZScoresCollection[NumDemos, NumPoints]
    action_trust_values: TrustValuesCollection[NumDemos, NumPoints]
    state_z_scores: ZScoresCollection[NumDemos, NumPoints] | None
    state_trust_values: TrustValuesCollection[NumDemos, NumPoints] | None
    pseudo_labels: PseudoLabels[NumDemos, NumPoints, DimState, DimAction]


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
        bins = RibbonTokenConsolidator(bins).consolidate_ribbon_tokens()

        action_z_scores, action_trust_values = TrustValueComputer(
            self.demonstrations, bins
        ).compute(mode=action_mode(), params=self.config.action_trust_value_params)

        state_z_scores, state_trust_values = None, None
        if self.config.use_state_labels:
            state_z_scores, state_trust_values = TrustValueComputer(
                self.demonstrations, bins
            ).compute(mode=state_mode(), params=self.config.state_trust_value_params)

        pseudo_labels = PseudoLabelComputer(
            self.demonstrations, bins
        ).compute_pseudo_labels(
            action_trust_values,
            state_trust_values,
            action_params=self.config.action_pseudo_label_params,
            state_params=self.config.state_pseudo_label_params,
        )

        return PACERResult(
            phases=phases,
            bins=bins,
            action_z_scores=action_z_scores,
            action_trust_values=action_trust_values,
            state_z_scores=state_z_scores,
            state_trust_values=state_trust_values,
            pseudo_labels=pseudo_labels,
        )


## ─────────────────────────────────────────────────────────────────────────────

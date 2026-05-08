"""
Experiment runs
=======
"""
# src/pacer/experiments.py

## ── Imports ──────────────────────────────────────────────────────────────────

from dataclasses import KW_ONLY, dataclass
from typing import Generic, Literal

import matplotlib.pyplot as plt
from pyLASAHandwritingDataset import ALL_SINGLE_PATTERN_MOTIONS, SinglePatternMotion
from typingkit.numpy._typed.helpers import TWO

from pacer import console
from pacer.base import Demonstrations
from pacer.pacer import (
    Binner,
    PseudoLabelComputer,
    PseudoLabelParams,
    RibbonTokenConsolidator,
    TrustValueComputer,
    TrustValueParams,
)
from pacer.plotting import (
    plot_action_comparison,
    plot_phases,
    plot_ribbon_action_field,
    plot_state_comparison,
    plot_states,
    plot_trajectories,
    plot_trust_values,
)
from pacer.testutils import (
    DemonstrationLoader,
    DemonstrationsChoice,
    PhaseEstimatorChoice,
    PhasePipeline,
)
from pacer.trainers import BCTrainer, WeightedBCTrainer
from pacer.typings import NumDemos, NumPoints

## ── Experiments ──────────────────────────────────────────────────────────────


@dataclass
class BCExperiment(Generic[NumDemos, NumPoints]):
    """BC Policy."""

    demonstrations: Demonstrations[NumDemos, NumPoints, TWO, TWO]

    def run(self) -> None:
        console.rule("[blue]BC policy[/blue]", style="blue")

        # Behavioral cloning
        trainer = BCTrainer(self.demonstrations, device="cpu")
        policy_loss = trainer.train(
            policy_hidden_dim=128,
            policy_lr=1e-3,
            policy_epochs=240,
        )
        console.print(f"Policy loss: {policy_loss}")


@dataclass
class PACERBCExperiment(Generic[NumDemos, NumPoints]):
    """PACER + BC Policy."""

    demonstrations: Demonstrations[NumDemos, NumPoints, TWO, TWO]
    _: KW_ONLY
    phase_estimator_choice: PhaseEstimatorChoice = "MLP"
    evaluate_phases: bool = False
    use_state_labels: bool = False
    show_plots: bool = True

    def run(self) -> None:
        console.rule(
            f"[blue]PACER[{self.phase_estimator_choice}_PHASE_ESTIMATION] + BC policy[/blue]",
            style="blue",
        )

        # PACER
        phases = PhasePipeline(
            self.demonstrations,
            choice=self.phase_estimator_choice,
            evaluate_phases=self.evaluate_phases,
        ).run()
        bins = Binner(
            self.demonstrations,
            phases,
            n_bins=96,  # B
        ).make_bins()
        bins = RibbonTokenConsolidator(bins).consolidate_ribbon_tokens()
        action_trust_values = TrustValueComputer(
            self.demonstrations, bins, choice="Action"
        ).compute_trust_values(
            params=TrustValueParams(
                tukey_cutoff=4.685,  # c
                min_trust=0.02,  # w_min
            ),
        )
        state_trust_values = None
        if self.use_state_labels:
            state_trust_values = TrustValueComputer(
                self.demonstrations, bins, choice="State"
            ).compute_trust_values(
                params=TrustValueParams(
                    tukey_cutoff=4.685,  # c
                    min_trust=0.02,  # w_min
                )
            )
        pseudo_labels = PseudoLabelComputer(
            self.demonstrations, bins
        ).compute_pseudo_labels(
            action_trust_values,
            state_trust_values,
            action_params=PseudoLabelParams(
                debias_weight=0.5,  # lambda_{debias}
                sideways_attenuation_shrinkage=0.5,  # rho_0
                speed_regularisation_influence=0.5,  # eta_0
                temporal_smoothing_weight=0.0,  # kappa
            ),
            state_params=PseudoLabelParams(
                debias_weight=0.1,  # lambda_{debias}
                sideways_attenuation_shrinkage=0.1,  # rho_0
                speed_regularisation_influence=0.1,  # eta_0
                temporal_smoothing_weight=0.9,  # kappa
            ),
        )

        # Behavioral cloning
        trainer = WeightedBCTrainer(
            states=pseudo_labels.states or self.demonstrations.states,
            targets=pseudo_labels.actions,
            weights=action_trust_values,
            device="cpu",
        )
        policy_loss = trainer.train(
            policy_hidden_dim=128,
            policy_lr=1e-3,
            policy_epochs=240,
        )
        console.print(f"Policy loss: {policy_loss}")

        if self.show_plots:
            plot_trajectories(self.demonstrations)
            plot_phases(phases)
            plot_trust_values(action_trust_values)
            if state_trust_values is not None:
                plot_trust_values(state_trust_values)
            plot_ribbon_action_field(bins)
            plot_action_comparison(
                self.demonstrations[0].actions,
                pseudo_labels.actions[0],
                title="Demo 0: Action refinement",
            )
            if pseudo_labels.states is not None:
                plot_states(pseudo_labels.states)
                plot_state_comparison(
                    self.demonstrations[0].states,
                    pseudo_labels.states[0],
                    title="Demo 0: State refinement",
                )


@dataclass(kw_only=True)
class BCvsPACERBCExperiment:
    """BC Policy vs. PACER + BC Policy."""

    show_plots: bool = True
    demonstrations_choice: DemonstrationsChoice = "FROM_LASA"
    LASA_pattern: (
        list[SinglePatternMotion] | SinglePatternMotion | Literal["ALL"] | None
    ) = None
    phase_estimator_choice: (
        list[PhaseEstimatorChoice] | PhaseEstimatorChoice | Literal["ALL"]
    ) = "MLP"
    evaluate_phases: bool = False
    use_corruptions: bool = False
    use_state_labels: bool = False
    filepath: str | None = None

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

        for pattern in LASA_patterns:
            match self.demonstrations_choice:
                case "FROM_LASA" | "CUSTOM_FROM_LASA" | "LEGACY_CUSTOM_FROM_LASA":
                    assert pattern is not None
                    console.rule(
                        f"[bold gold3]LASA Pattern: {pattern}[/bold gold3]",
                        characters="\u2501",
                        style="gold3",
                    )
                case "CUSTOM_FROM_LOAD" | "LEGACY_CUSTOM_FROM_LOAD":
                    assert self.filepath is not None
                    console.rule(
                        f"[bold gold3]File: {self.filepath}[/gold3]",
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
                choice=self.demonstrations_choice,
                pattern=pattern,
                filepath=self.filepath,
                use_corruptions=self.use_corruptions,
            ).load()

            BCExperiment(demonstrations).run()
            for phase_estimator_choice in phase_estimator_choices:
                PACERBCExperiment(
                    demonstrations,
                    phase_estimator_choice=phase_estimator_choice,
                    evaluate_phases=self.evaluate_phases,
                    use_state_labels=self.use_state_labels,
                    show_plots=self.show_plots,
                ).run()
                if self.show_plots:
                    plt.show()  # pyright: ignore[reportUnknownMemberType]

            if self.demonstrations_choice in {"CUSTOM_FROM_LOAD", "CUSTOM_DRAW"}:
                break

        console.rule(characters="\u2501", style="gold3")


## ─────────────────────────────────────────────────────────────────────────────

"""
Test BC Policy vs. PACER + BC Policy.
"""
# tests/test_pacer_bc.py

from typing import Literal

import matplotlib.pyplot as plt
from pyLASAHandwritingDataset import ALL_SINGLE_PATTERN_MOTIONS, SinglePatternMotion
from typingkit.core import RuntimeOptions, set_global_default_runtime_options
from typingkit.numpy._typed.helpers import TWO

from pacer import console
from pacer.base import Demonstrations
from pacer.pacer import PACER
from pacer.plotting import full_diagnostic
from pacer.testutils import (
    DemonstrationsChoice,
    PhaseEstimatorChoice,
    get_demonstrations,
    get_phases,
)
from pacer.trainers import BCTrainer, PACERBCTrainer
from pacer.typings import NumDemos, NumPoints

set_global_default_runtime_options(RuntimeOptions(validate=True))


def run_pacerbc(
    demonstrations: Demonstrations[NumDemos, NumPoints, TWO, TWO],
    phase_estimator_choice: PhaseEstimatorChoice = "MLP",
    show_plots: bool = True,
) -> None:
    console.rule(
        f"[blue]PACER[{phase_estimator_choice}_PHASE_ESTIMATION] + BC policy[/blue]",
        style="blue",
    )

    # PACER
    phases = get_phases(demonstrations, choice=phase_estimator_choice)
    pacer = PACER(
        demonstrations,
        phases,
        n_bins=96,  # B
    )
    pacer.prepare(
        tukey_cutoff=4.685,  # c
        min_trust=0.02,  # w_min
        debias_weight=0.5,  # lambda_{debias}
        sideways_attenuation_shrinkage=0.5,  # rho_0
        speed_regularisation_influence=0.5,  # eta_0
        temporal_smoothing_weight=0.0,  # kappa
    )

    # Behavioral cloning
    trainer = PACERBCTrainer(pacer, device="cpu")
    policy_loss = trainer.train(
        policy_hidden_dim=128,
        policy_lr=1e-3,
        policy_epochs=240,
    )
    console.print(f"Policy loss: {policy_loss}")

    if show_plots:
        full_diagnostic(trainer.pacer)


def run_bc(demonstrations: Demonstrations[NumDemos, NumPoints, TWO, TWO]) -> None:
    console.rule("[blue]BC policy[/blue]", style="blue")

    # Behavioral cloning
    trainer = BCTrainer(demonstrations, device="cpu")
    policy_loss = trainer.train(
        policy_hidden_dim=128,
        policy_lr=1e-3,
        policy_epochs=240,
    )
    console.print(f"Policy loss: {policy_loss}")


def test_pacer_bc(
    show_plots: bool = True,
    demonstrations_choice: DemonstrationsChoice = "FROM_LASA",
    LASA_pattern: list[SinglePatternMotion]
    | SinglePatternMotion
    | Literal["ALL"]
    | None = None,
    phase_estimator_choice: list[PhaseEstimatorChoice]
    | PhaseEstimatorChoice
    | Literal["ALL"] = "MLP",
    use_corruptions: bool = False,
    filepath: str | None = None,
) -> None:

    # Resolve LASA patterns
    LASA_patterns: list[SinglePatternMotion | None]
    match LASA_pattern:
        case list():
            LASA_patterns = list(LASA_pattern)
        case str():
            match LASA_pattern:
                case "ALL":
                    LASA_patterns = list(ALL_SINGLE_PATTERN_MOTIONS)
                case _:
                    LASA_patterns = [LASA_pattern]
        case None:
            LASA_patterns = [None]

    # Resolve phase estimator choices
    phase_estimator_choices: list[PhaseEstimatorChoice]
    match phase_estimator_choice:
        case list():
            phase_estimator_choices = phase_estimator_choice
        case str():
            match phase_estimator_choice:
                case "ALL":
                    phase_estimator_choices = [
                        "MLP",
                        "NORMALISED_TIME_INDEX",
                        "PATH_LENGTH",
                    ]
                case _:
                    phase_estimator_choices = [phase_estimator_choice]

    for pattern in LASA_patterns:
        match demonstrations_choice:
            case "FROM_LASA" | "CUSTOM_FROM_LASA":
                assert pattern is not None
                console.rule(
                    f"[bold gold3]LASA Pattern: {pattern}[/bold gold3]",
                    characters="\u2501",
                    style="gold3",
                )
            case "CUSTOM_FROM_LOAD":
                assert filepath is not None
                console.rule(
                    f"[bold gold3]File: {filepath}[/gold3]",
                    characters="\u2501",
                    style="bold gold3",
                )
            case "CUSTOM_DRAW":
                console.rule(
                    "[bold gold3]Custom demonstrations[/bold gold3]",
                    characters="\u2501",
                    style="gold3",
                )

        demonstrations = get_demonstrations(
            choice=demonstrations_choice,
            pattern=pattern,
            filepath=filepath,
            use_corruptions=use_corruptions,
        )

        run_bc(demonstrations)
        for phase_estimator_choice in phase_estimator_choices:
            run_pacerbc(demonstrations, phase_estimator_choice, show_plots=show_plots)
            if show_plots:
                plt.show()  # pyright: ignore[reportUnknownMemberType]

        if demonstrations_choice in {"CUSTOM_FROM_LOAD", "CUSTOM_DRAW"}:
            break

    console.rule(characters="\u2501", style="gold3")


if __name__ == "__main__":
    test_pacer_bc(
        show_plots=True,
        demonstrations_choice="FROM_LASA",
        LASA_pattern="GShape",
        phase_estimator_choice="MLP",
        use_corruptions=False,
        filepath=None,
    )

"""
Test BC Policy vs. PACER + BC Policy.
"""
# tests/test_pacer_bc.py

# pyright: standard

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
    get_phase_estimator,
)
from pacer.trainers import BCTrainer, PACERBCTrainer
from pacer.typings import NumDemos, NumPoints

set_global_default_runtime_options(RuntimeOptions(validate=True))


def run_pacerbc(
    demonstrations: Demonstrations[NumDemos, NumPoints, TWO, TWO],
    phase_estimator_choice: PhaseEstimatorChoice = "MLP",
    show_plots: bool = True,
) -> None:
    console.rule(f"PACER[{phase_estimator_choice}_PHASE_ESTIMATION] + BC policy")

    # PACER
    phase_estimator = get_phase_estimator(demonstrations, choice=phase_estimator_choice)
    pacer = PACER(
        demonstrations,
        phase_estimator,
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
    console.rule("BC policy")

    # Behavioral cloning
    trainer = BCTrainer(demonstrations, device="cpu")
    policy_loss = trainer.train(
        policy_hidden_dim=128,
        policy_lr=1e-3,
        policy_epochs=240,
    )
    console.print(f"Policy loss: {policy_loss}")


def test_pacer_bc(
    for_all_lasa_single_pattern_motions: bool = False,
    show_plots: bool = True,
    demonstrations_choice: DemonstrationsChoice = "FROM_LASA",
    LASA_pattern: SinglePatternMotion | None = None,
    phase_estimator_choice: PhaseEstimatorChoice = "MLP",
    use_corruptions: bool = False,
    filepath: str | None = None,
) -> None:
    def bc_and_pacerbc(pattern: SinglePatternMotion | None = None) -> None:
        match demonstrations_choice:
            case "FROM_LASA" | "CUSTOM_FROM_LASA":
                assert pattern is not None
                console.rule(f"LASA Pattern: {pattern}", style="blue")
            case "CUSTOM_FROM_LOAD":
                assert filepath is not None
                console.rule(f"File: {filepath}", style="blue")
            case "CUSTOM_DRAW":
                console.rule("Custom demonstrations", style="blue")
        demonstrations = get_demonstrations(
            choice=demonstrations_choice,
            pattern=pattern,
            filepath=filepath,
            use_corruptions=use_corruptions,
        )
        run_bc(demonstrations)
        run_pacerbc(
            demonstrations,
            phase_estimator_choice=phase_estimator_choice,
            show_plots=show_plots,
        )
        if show_plots:
            plt.show()

    if for_all_lasa_single_pattern_motions:
        for pattern in ALL_SINGLE_PATTERN_MOTIONS:
            bc_and_pacerbc(pattern)
    else:
        bc_and_pacerbc(pattern=LASA_pattern)


if __name__ == "__main__":
    test_pacer_bc(
        for_all_lasa_single_pattern_motions=False,
        show_plots=True,
        demonstrations_choice="FROM_LASA",
        LASA_pattern="GShape",
        phase_estimator_choice="MLP",
        use_corruptions=False,
        filepath=None,
    )

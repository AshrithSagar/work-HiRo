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
        full_diagnostic(trainer)


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


def test_pacer_bc() -> None:
    ## ── Config ───────────────────────────────────────────────────────────────

    SHOW_PLOTS: bool = True
    USE_CORRUPTIONS: bool = False
    FOR_ALL_LASA_SINGLE_PATTERN_MOTIONS: bool = False
    DEMONSTRATIONS_CHOICE: DemonstrationsChoice = "FROM_LASA"
    PHASE_ESTIMATOR_CHOICE: PhaseEstimatorChoice = "MLP"
    LASA_PATTERN: SinglePatternMotion = "GShape"

    # ──────────────────────────────────────────────────────────────────────────

    def bc_and_pacerbc(pattern: SinglePatternMotion) -> None:
        match DEMONSTRATIONS_CHOICE:
            case "FROM_LASA" | "CUSTOM_FROM_LASA":
                console.rule(f"LASA Pattern: {pattern}", style="blue")
        demonstrations = get_demonstrations(
            choice=DEMONSTRATIONS_CHOICE,
            pattern=pattern,
            use_corruptions=USE_CORRUPTIONS,
        )
        run_bc(demonstrations)
        run_pacerbc(
            demonstrations,
            phase_estimator_choice=PHASE_ESTIMATOR_CHOICE,
            show_plots=SHOW_PLOTS,
        )
        if SHOW_PLOTS:
            plt.show()

    if FOR_ALL_LASA_SINGLE_PATTERN_MOTIONS:
        for pattern in ALL_SINGLE_PATTERN_MOTIONS:
            bc_and_pacerbc(pattern)
    else:
        bc_and_pacerbc(pattern=LASA_PATTERN)


if __name__ == "__main__":
    test_pacer_bc()

# tests/test_pacer_bc.py

# pyright: standard

from typing import Any, Literal

import matplotlib.pyplot as plt
from typingkit.core import RuntimeOptions, set_global_default_runtime_options
from typingkit.numpy._typed.helpers import TWO

from pacer import console
from pacer.base import Demonstrations
from pacer.corruptions import DemonstrationCorrupter
from pacer.interactive import InteractiveDataSet
from pacer.lasa import LASADataSet
from pacer.pacer import PACER
from pacer.phase import (
    MLPPhaseEstimator,
    NormalisedTimeIndexPhaseEstimator,
    PhaseEstimatorProtocol,
)
from pacer.plotting import full_diagnostic
from pacer.trainers import BCTrainer, PACERBCTrainer
from pacer.typings import NumDemos, NumPoints

set_global_default_runtime_options(RuntimeOptions(validate=True))


def get_demonstrations(
    choice: Literal[
        "FROM_LASA", "CUSTOM_FROM_LOAD", "CUSTOM_FROM_LASA", "CUSTOM_DRAW"
    ] = "FROM_LASA",
) -> Demonstrations[Any, Any, TWO, TWO]:
    match choice:
        case "FROM_LASA":
            return LASADataSet("GShape").to_demonstrations()
        case "CUSTOM_FROM_LOAD":
            drawer = InteractiveDataSet.load("hand_drawn_demos.npz")
            return drawer.to_demonstrations()
        case "CUSTOM_FROM_LASA":
            drawer = InteractiveDataSet.from_LASA("GShape")
            plt.show(block=True)
            return drawer.to_demonstrations()
        case "CUSTOM_DRAW":
            drawer = InteractiveDataSet()
            plt.show(block=True)
            return drawer.to_demonstrations()
    raise ValueError


def get_phase_estimator(
    demonstrations: Demonstrations[NumDemos, NumPoints, TWO, TWO],
    choice: Literal["MLP", "NORMALISED_TIME_INDEX"] = "MLP",
) -> PhaseEstimatorProtocol[NumDemos, NumPoints, TWO, TWO]:
    match choice:
        case "MLP":
            phase_estimator = MLPPhaseEstimator(demonstrations)
            phase_loss = phase_estimator.train(
                hidden_dim=128,
                margin=1.0,  # m
                lr=1e-3,
                epochs=240,
            )
            console.print(f"Phase scorer loss: {phase_loss}")
            return phase_estimator
        case "NORMALISED_TIME_INDEX":
            return NormalisedTimeIndexPhaseEstimator(demonstrations)
    raise ValueError


def test_pacerbc(
    demonstrations: Demonstrations[NumDemos, NumPoints, TWO, TWO],
    phase_estimator_choice: Literal["MLP", "NORMALISED_TIME_INDEX"] = "MLP",
    use_corruptions: bool = False,
) -> None:
    console.rule(f"PACER[{phase_estimator_choice}_PHASE_ESTIMATION] + BC policy")

    if use_corruptions:
        corrupter = DemonstrationCorrupter(
            demonstrations=demonstrations,
            noise_std=0.2,
            outlier_fraction=0.2,
            outlier_scale=5.0,
            bias_strength=0.2,
        )
        demonstrations = corrupter.inject_corruptions()

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

    full_diagnostic(trainer)


def test_bc(
    demonstrations: Demonstrations[NumDemos, NumPoints, TWO, TWO],
    use_corruptions: bool = False,
) -> None:
    console.rule("BC policy")

    if use_corruptions:
        corrupter = DemonstrationCorrupter(
            demonstrations=demonstrations,
            noise_std=0.2,
            outlier_fraction=0.2,
            outlier_scale=5.0,
            bias_strength=0.2,
        )
        demonstrations = corrupter.inject_corruptions()

    # Behavioral cloning
    trainer = BCTrainer(demonstrations, device="cpu")
    policy_loss = trainer.train(
        policy_hidden_dim=128,
        policy_lr=1e-3,
        policy_epochs=240,
    )
    console.print(f"Policy loss: {policy_loss}")


if __name__ == "__main__":
    demonstrations = get_demonstrations(choice="FROM_LASA")
    test_bc(demonstrations)
    test_pacerbc(demonstrations, phase_estimator_choice="MLP")
    plt.show()

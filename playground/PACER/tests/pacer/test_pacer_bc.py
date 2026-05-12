"""
Test BC Policy vs. PACER + BC Policy.
"""
# tests/test_pacer_bc.py

from typingkit.core import RuntimeOptions, set_global_default_runtime_options

from pacer.experiments import BCvsPACERBCExperiment
from pacer.phase.estimation import MLPPhaseEstimatorConfig
from pacer.trainers import BCTrainConfig

set_global_default_runtime_options(RuntimeOptions(validate=True))


if __name__ == "__main__":
    BCvsPACERBCExperiment(
        show_plots=True,
        demonstrations_choice="FROM_LASA",
        LASA_pattern="GShape",
        phase_estimator_choice="MLP",
        mlp_phase_estimator_config=MLPPhaseEstimatorConfig(
            hidden_dim=128,
            margin=1.0,  # m
            lr=1e-3,
            epochs=240,
        ),
        evaluate_phases=False,
        corruptions_choice=None,
        use_state_labels=False,
        bc_train_config=BCTrainConfig(
            hidden_dim=128,
            lr=1e-3,
            epochs=240,
        ),
        filepath=None,
    ).run()

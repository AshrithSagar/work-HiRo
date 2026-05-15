"""
Test BC Policy vs. PACER + BC Policy.
"""
# tests/test_pacer_bc.py

from typingkit.core import RuntimeOptions, set_global_default_runtime_options

from pacer.experiments import BCvsPACERBCExperiment
from pacer.pacer import PseudoLabelParams, TrustValueParams
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
        n_bins=96,
        action_trust_value_params=TrustValueParams(
            tukey_cutoff=4.685,  # c
            min_trust=0.02,  # w_min
        ),
        action_pseudo_label_params=PseudoLabelParams(
            debias_weight=0.5,  # lambda_{debias}
            sideways_attenuation_shrinkage=0.5,  # rho_0
            speed_regularisation_influence=0.5,  # eta_0
            temporal_smoothing_weight=0.0,  # kappa
        ),
        use_state_labels=False,
        state_trust_value_params=TrustValueParams(
            tukey_cutoff=4.685,  # c
            min_trust=0.02,  # w_min
        ),
        state_pseudo_label_params=PseudoLabelParams(
            debias_weight=0.1,  # lambda_{debias}
            sideways_attenuation_shrinkage=0.1,  # rho_0
            speed_regularisation_influence=0.1,  # eta_0
            temporal_smoothing_weight=0.9,  # kappa
        ),
        bc_train_config=BCTrainConfig(
            hidden_dim=128,
            lr=1e-3,
            epochs=240,
        ),
        filepath=None,
    ).run()

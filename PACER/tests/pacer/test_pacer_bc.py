"""
Test BC Policy vs. PACER + BC Policy.
"""
# tests/test_pacer_bc.py

from typingkit.core import RuntimeOptions, set_global_default_runtime_options

from pacer.bc import BCTrainConfig
from pacer.experiments import BCvsPACERBCExperiment
from pacer.pacer import PACERConfig, PseudoLabelParams, TrustValueParams
from pacer.phase.estimation import MLPPhaseEstimatorConfig
from pacer.testutils import (
    DemonstrationLoader,
    DemonstrationLoaderConfig,
    PhasePipelineConfig,
)
from pacer.visualisation import PACERVisualisationConfig, PACERVisualiser

set_global_default_runtime_options(RuntimeOptions(validate=True))


if __name__ == "__main__":
    demonstrations = DemonstrationLoader(
        config=DemonstrationLoaderConfig(
            choice="FROM_LASA",
            LASA_pattern="GShape",
            filepath=None,
            corruptions_choice=None,
        ),
    ).load()

    result = BCvsPACERBCExperiment(
        demonstrations,
        pacer_config=PACERConfig(
            phase_pipeline_config=PhasePipelineConfig(
                phase_estimator_choice="MLP",
                mlp_phase_estimator_config=MLPPhaseEstimatorConfig(
                    hidden_dim=128,
                    margin=1.0,  # m
                    lr=1e-3,
                    epochs=240,
                ),
                evaluate_phases=False,
            ),
            n_bins=96,  # B
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
        ),
        bc_train_config=BCTrainConfig(
            hidden_dim=128,
            lr=1e-3,
            epochs=240,
        ),
    ).run()

    PACERVisualiser(
        demonstrations,
        pacer_result=result.pacer_result,
        config=PACERVisualisationConfig(
            show=True,
            save_dir=None,
            trajectories=True,
            phases=True,
            trust_values=True,
            action_comparison=True,
            state_comparison=True,
            ribbon_action_field=True,
        ),
    ).render()

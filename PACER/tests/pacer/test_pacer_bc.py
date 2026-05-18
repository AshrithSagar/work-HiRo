"""
Test BC Policy vs. PACER + BC Policy.
"""
# tests/test_pacer_bc.py

from typingkit.core import RuntimeOptions, set_global_default_runtime_options
from typingkit.numpy._typed.helpers import TWO

from pacer.base import Action, State
from pacer.bc import BCTrainConfig
from pacer.experiments import BCvsPACERBCExperiment
from pacer.pacer import (
    DebiasTowardsAnchorStep,
    PACERConfig,
    PseudoLabelParams,
    RefinementPipeline,
    SidewaysAttenuationStep,
    SpeedRegularisationStep,
    TemporalSmoother,
    TrustValueParams,
)
from pacer.phase.estimation import MLPPhaseEstimatorConfig
from pacer.plotting import PACERVisualisationConfig, PACERVisualiser
from pacer.testutils import (
    DemonstrationLoader,
    DemonstrationLoaderConfig,
    PhasePipelineConfig,
)

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
            action_pseudo_label_params=PseudoLabelParams[Action[TWO]](
                pipeline=RefinementPipeline(
                    steps=(
                        DebiasTowardsAnchorStep(debias_weight=0.5),  # lambda_{debias}
                        SidewaysAttenuationStep(shrinkage=0.5),  # rho_0
                        SpeedRegularisationStep(influence=0.5),  # eta_0
                    ),
                ),
                smoother=TemporalSmoother(smoothing_weight=0.0),  # kappa
            ),
            use_state_labels=False,
            state_trust_value_params=TrustValueParams(
                tukey_cutoff=4.685,  # c
                min_trust=0.02,  # w_min
            ),
            state_pseudo_label_params=PseudoLabelParams[State[TWO]](
                pipeline=RefinementPipeline(
                    steps=(
                        DebiasTowardsAnchorStep(debias_weight=0.1),  # lambda_{debias}
                        SidewaysAttenuationStep(shrinkage=0.1),  # rho_0
                        SpeedRegularisationStep(influence=0.1),  # eta_0
                    ),
                ),
                smoother=TemporalSmoother(smoothing_weight=0.9),  # kappa
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
            states_before_after=True,
            action_comparison=True,
            state_comparison=True,
            ribbon_action_field=True,
            action_correction_magnitude=True,
            residual_distribution=True,
            trust_heatmap=True,
            bin_occupancy=True,
            trust_vs_correction=True,
            ribbon_statistics=True,
            phase_velocity=True,
            smoothness_comparison=True,
        ),
    ).render()

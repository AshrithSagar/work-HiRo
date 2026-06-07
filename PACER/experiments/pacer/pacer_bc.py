"""
Test BC Policy vs. PACER + BC Policy.
"""
# tests/test_pacer_bc.py

from pathlib import Path

from typingkit.core import RuntimeOptions, set_global_default_runtime_options

from pacer.bc import BCTrainConfig
from pacer.datasets import DemonstrationLoader, DemonstrationLoaderConfig
from pacer.experiments import BCvsPACERBCExperiment
from pacer.pacer import PACERConfig
from pacer.pacer.consensus import (
    ConsensusConfig,
    MADResidualScaleEstimator,
    MedianScalarEstimator,
    MedianVectorEstimator,
)
from pacer.pacer.pseudolabel import (
    DebiasTowardsAnchorStep,
    PseudoLabelParams,
    PseudoLabelRefinementPipeline,
    SidewaysAttenuationStep,
    SpeedRegularisationStep,
    TemporalSmoother,
)
from pacer.pacer.trust import (
    EuclideanResidualComputer,
    MinimumTrustFloor,
    TrustPipeline,
    TrustValueParams,
    TukeyBiweightKernel,
)
from pacer.phase import PhasePipelineConfig
from pacer.phase.estimation import DTWPhaseEstimatorConfig, MLPPhaseEstimatorConfig
from pacer.plotting.legacy import PACERVisualisationConfig, PACERVisualiser
from pacer.utils import MAD_SCALE

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
                dtw_phase_estimator_config=DTWPhaseEstimatorConfig(
                    reference_demo_index=1,
                ),
                evaluate_phases=False,
            ),
            n_bins=96,  # B
            consensus_config=ConsensusConfig(
                vector_estimator=MedianVectorEstimator(),
                scalar_estimator=MedianScalarEstimator(),
                residual_scale_estimator=MADResidualScaleEstimator(),
            ),
            action_trust_value_params=TrustValueParams(
                pipeline=TrustPipeline(
                    residual_computer=EuclideanResidualComputer(),
                    scale_estimator=MADResidualScaleEstimator(
                        consistency_scale=MAD_SCALE
                    ),
                    kernel=TukeyBiweightKernel(cutoff=4.685),  # c
                    transforms=[
                        MinimumTrustFloor(minimum=0.02),  # w_min
                    ],
                ),
            ),
            action_pseudo_label_params=PseudoLabelParams(
                pipeline=PseudoLabelRefinementPipeline(
                    steps=[
                        DebiasTowardsAnchorStep(debias_weight=0.5),  # lambda_{debias}
                        SidewaysAttenuationStep(shrinkage=0.5),  # rho_0
                        SpeedRegularisationStep(influence=0.5),  # eta_0
                    ],
                ),
                smoother=TemporalSmoother(smoothing_weight=0.0),  # kappa
            ),
            use_state_labels=False,
            state_trust_value_params=TrustValueParams(
                pipeline=TrustPipeline(
                    residual_computer=EuclideanResidualComputer(),
                    scale_estimator=MADResidualScaleEstimator(
                        consistency_scale=MAD_SCALE
                    ),
                    kernel=TukeyBiweightKernel(cutoff=4.685),  # c
                    transforms=[
                        MinimumTrustFloor(minimum=0.02),  # w_min
                    ],
                ),
            ),
            state_pseudo_label_params=PseudoLabelParams(
                pipeline=PseudoLabelRefinementPipeline(
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
            show=False,
            save_dir=Path(__file__).parent / "plots_pacer_bc",
            trajectories=True,
            phases=True,
            trust_values=True,
            states_before_after=True,
            actions_before_after=True,
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
            trust_colored_trajectory=True,
            trust_colored_action_field=True,
            action_correction_vectors=True,
            phase_aligned_trajectories=True,
            ribbon_corridor=True,
            residual_vs_phase=True,
            action_angle_deviation=True,
        ),
    ).render()

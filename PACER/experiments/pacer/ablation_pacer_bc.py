"""
Ablation Study
=======
experiments/pacer/ablation_pacer_bc.py
"""
# experiments/pacer/ablation_pacer_bc.py

from __future__ import annotations

import argparse
import copy
import dataclasses
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from typingkit.core import RuntimeOptions, set_global_default_runtime_options

from pacer.bc import BCTrainConfig
from pacer.datasets import DemonstrationLoader, DemonstrationLoaderConfig
from pacer.experiments import BCExperiment, PACERBCExperiment
from pacer.pacer import PACERConfig
from pacer.pacer.analysis import (
    CorrectionMagnitudeAnalyser,
    ResidualAnalyser,
    SmoothnessAnalyser,
    TrustValueAnalysis,
)
from pacer.pacer.consensus import (
    ArcLengthLocalLinearTangentEstimator,
    CentralDifferenceTangentEstimator,
    ConsensusConfig,
    ForwardDifferenceTangentEstimator,
    GaussianTangentEstimator,
    IdentityTangentEstimator,
    MADResidualScaleEstimator,
    MeanScalarEstimator,
    MeanVectorEstimator,
    MedianScalarEstimator,
    MedianVectorEstimator,
    StandardDeviationScaleEstimator,
    UnitTangentEstimator,
)
from pacer.pacer.pseudolabel import (
    DebiasTowardsAnchorStep,
    PseudoLabelParams,
    PseudoLabelRefinementPipeline,
    SidewaysAttenuationStep,
    SpeedRegularisationStep,
    TemporalSmoother,
)
from pacer.pacer.trust.kernels import (
    AndrewsSineKernel,
    CauchyKernel,
    GaussianKernel,
    GemanMcClureKernel,
    HardThresholdKernel,
    HuberKernel,
    LogisticKernel,
    MinimumTrustFloor,
    StudentTKernel,
    TukeyBiweightKernel,
    WelschKernel,
)
from pacer.pacer.trust.legacy import (
    AngularResidualComputer,
    CosineResidualComputer,
    EuclideanAngularResidualComputer,
    EuclideanResidualComputer,
    LpResidualComputer,
    ManhattanResidualComputer,
    ProjectionResidualComputer,
    RelativeMagnitudeResidualComputer,
    TrustPipeline,
    TrustValueParams,
)
from pacer.phase import PhasePipelineConfig
from pacer.phase.estimation import DTWPhaseEstimatorConfig, MLPPhaseEstimatorConfig

set_global_default_runtime_options(RuntimeOptions(validate=True))

## ── Baseline config ──────────────────────────────────────────────────────────


def make_baseline_pacer_config() -> PACERConfig:
    return PACERConfig(
        phase_pipeline_config=PhasePipelineConfig(
            phase_estimator_choice="MLP",
            mlp_phase_estimator_config=MLPPhaseEstimatorConfig(
                hidden_dim=128, margin=1.0, lr=1e-3, epochs=240
            ),
            dtw_phase_estimator_config=DTWPhaseEstimatorConfig(reference_demo_index=1),
            evaluate_phases=False,
        ),
        n_bins=96,
        consensus_config=ConsensusConfig(
            vector_estimator=MedianVectorEstimator(),
            scalar_estimator=MedianScalarEstimator(),
            residual_scale_estimator=MADResidualScaleEstimator(),
            tangent_estimator=CentralDifferenceTangentEstimator(),
        ),
        action_trust_value_params=TrustValueParams(
            pipeline=TrustPipeline(
                residual_computer=EuclideanResidualComputer(),
                scale_estimator=MADResidualScaleEstimator(scale="normal"),
                kernel=TukeyBiweightKernel(cutoff=4.685),
                transforms=[MinimumTrustFloor(minimum=0.02)],
            )
        ),
        action_pseudo_label_params=PseudoLabelParams(
            pipeline=PseudoLabelRefinementPipeline(
                steps=[
                    DebiasTowardsAnchorStep(debias_weight=0.5),
                    SidewaysAttenuationStep(shrinkage=0.5),
                    SpeedRegularisationStep(influence=0.5),
                ]
            ),
            smoother=TemporalSmoother(smoothing_weight=0.0),
        ),
        use_state_labels=False,
        state_trust_value_params=TrustValueParams(
            pipeline=TrustPipeline(
                residual_computer=EuclideanResidualComputer(),
                scale_estimator=MADResidualScaleEstimator(scale="normal"),
                kernel=TukeyBiweightKernel(cutoff=4.685),
                transforms=[MinimumTrustFloor(minimum=0.02)],
            )
        ),
        state_pseudo_label_params=PseudoLabelParams(
            pipeline=PseudoLabelRefinementPipeline(
                steps=(
                    DebiasTowardsAnchorStep(debias_weight=0.1),
                    SidewaysAttenuationStep(shrinkage=0.1),
                    SpeedRegularisationStep(influence=0.1),
                )
            ),
            smoother=TemporalSmoother(smoothing_weight=0.9),
        ),
    )


BASELINE_BC_TRAIN_CONFIG = BCTrainConfig(hidden_dim=128, lr=1e-3, epochs=240)

## ── Ablation axes ─────────────────────────────────────────────────────────────
# group name -> {variant name: (config -> config, description)}

Mutator = Callable[[PACERConfig], PACERConfig]


def _mut(fn: Mutator) -> Mutator:
    def wrapped(cfg: PACERConfig) -> PACERConfig:
        cfg = copy.deepcopy(cfg)
        fn(cfg)
        return cfg

    return wrapped


def _replace_consensus(cfg: PACERConfig, /, **kwargs: Any) -> None:
    """ConsensusConfig is frozen -> rebuild it and reassign onto the (mutable) cfg."""
    cfg.consensus_config = dataclasses.replace(cfg.consensus_config, **kwargs)


def _replace_action_pipeline(cfg: PACERConfig, /, **kwargs: Any) -> None:
    """TrustPipeline is frozen -> rebuild it and reassign onto action_trust_value_params
    (which is itself mutable)."""
    cfg.action_trust_value_params.pipeline = dataclasses.replace(
        cfg.action_trust_value_params.pipeline, **kwargs
    )


def _replace_action_steps(cfg: PACERConfig, /, steps: list) -> None:
    """PseudoLabelRefinementPipeline is frozen -> rebuild it and reassign onto
    action_pseudo_label_params (which is itself mutable)."""
    cfg.action_pseudo_label_params.pipeline = dataclasses.replace(
        cfg.action_pseudo_label_params.pipeline, steps=steps
    )


AblationGroup = dict[str, tuple[Mutator, str]]

GROUPS: dict[str, AblationGroup] = {}

# ── Phase estimation choice ──────────────────────────────────────────────────-
GROUPS["phase"] = {
    "phase=MLP (baseline)": (
        _mut(
            lambda c: setattr(c.phase_pipeline_config, "phase_estimator_choice", "MLP")
        ),
        "Learned MLP progress scorer, trained per-run",
    ),
    "phase=DTW": (
        _mut(
            lambda c: setattr(c.phase_pipeline_config, "phase_estimator_choice", "DTW")
        ),
        "Dynamic-time-warping alignment to a reference demo",
    ),
    "phase=NormalisedTimeIndex": (
        _mut(
            lambda c: setattr(
                c.phase_pipeline_config,
                "phase_estimator_choice",
                "NORMALISED_TIME_INDEX",
            )
        ),
        "Naive t/T phase, ignores trajectory shape (sanity floor)",
    ),
    "phase=PathLength": (
        _mut(
            lambda c: setattr(
                c.phase_pipeline_config, "phase_estimator_choice", "PATH_LENGTH"
            )
        ),
        "Arc-length-normalised phase, no learning required",
    ),
}

# ── Consensus: vector/scalar location estimator ──────────────────────────────-
GROUPS["consensus_location"] = {
    "location=median (baseline)": (
        _mut(
            lambda c: _replace_consensus(
                c,
                vector_estimator=MedianVectorEstimator(),
                scalar_estimator=MedianScalarEstimator(),
            )
        ),
        "Robust median consensus across demos per phase-bin",
    ),
    "location=mean": (
        _mut(
            lambda c: _replace_consensus(
                c,
                vector_estimator=MeanVectorEstimator(),
                scalar_estimator=MeanScalarEstimator(),
            )
        ),
        "Non-robust mean consensus; expect more sensitivity to outlier demos",
    ),
}

# ── Consensus: residual scale estimator ──────────────────────────────────────-
GROUPS["scale_estimator"] = {
    "scale=MAD (baseline)": (
        _mut(
            lambda c: _replace_consensus(
                c, residual_scale_estimator=MADResidualScaleEstimator()
            )
        ),
        "Median absolute deviation, robust to outliers",
    ),
    "scale=std": (
        _mut(
            lambda c: _replace_consensus(
                c, residual_scale_estimator=StandardDeviationScaleEstimator()
            )
        ),
        "Non-robust standard deviation",
    ),
}

# ── Consensus: tangent estimator ──────────────────────────────────────────────
GROUPS["tangent"] = {
    "tangent=central-diff (baseline)": (
        _mut(
            lambda c: _replace_consensus(
                c, tangent_estimator=CentralDifferenceTangentEstimator()
            )
        ),
        "Central finite difference of consensus trajectory",
    ),
    "tangent=forward-diff": (
        _mut(
            lambda c: _replace_consensus(
                c, tangent_estimator=ForwardDifferenceTangentEstimator()
            )
        ),
        "Forward finite difference (causal, noisier)",
    ),
    "tangent=identity": (
        _mut(
            lambda c: _replace_consensus(
                c, tangent_estimator=IdentityTangentEstimator()
            )
        ),
        "No tangent estimation (disables direction-aware steps' geometry)",
    ),
    "tangent=unit": (
        _mut(lambda c: _replace_consensus(c, tangent_estimator=UnitTangentEstimator())),
        "Unit-normalised tangent direction only",
    ),
    "tangent=gaussian-smoothed": (
        _mut(
            lambda c: _replace_consensus(
                c, tangent_estimator=GaussianTangentEstimator()
            )
        ),
        "Gaussian-smoothed derivative, trades bias for lower variance",
    ),
    "tangent=arc-length-local-linear": (
        _mut(
            lambda c: _replace_consensus(
                c, tangent_estimator=ArcLengthLocalLinearTangentEstimator()
            )
        ),
        "Local-linear fit in arc-length parametrisation",
    ),
}

# ── Trust kernel ──────────────────────────────────────────────────────────────-
GROUPS["kernel"] = {
    "kernel=Tukey (baseline)": (
        _mut(
            lambda c: _replace_action_pipeline(
                c, kernel=TukeyBiweightKernel(cutoff=4.685)
            )
        ),
        "Redescending: fully rejects residuals beyond cutoff",
    ),
    "kernel=Gaussian": (
        _mut(lambda c: _replace_action_pipeline(c, kernel=GaussianKernel())),
        "Smooth non-redescending decay, never reaches exactly 0",
    ),
    "kernel=Huber": (
        _mut(lambda c: _replace_action_pipeline(c, kernel=HuberKernel())),
        "Mild down-weighting past delta, no hard rejection",
    ),
    "kernel=Welsch": (
        _mut(lambda c: _replace_action_pipeline(c, kernel=WelschKernel())),
        "Exponential redescending kernel",
    ),
    "kernel=Cauchy": (
        _mut(lambda c: _replace_action_pipeline(c, kernel=CauchyKernel())),
        "Heavy-tailed, gentle down-weighting",
    ),
    "kernel=Logistic": (
        _mut(lambda c: _replace_action_pipeline(c, kernel=LogisticKernel())),
        "Smooth sigmoid-shaped weighting",
    ),
    "kernel=HardThreshold": (
        _mut(lambda c: _replace_action_pipeline(c, kernel=HardThresholdKernel())),
        "Binary in/out cutoff, most aggressive rejection",
    ),
    "kernel=StudentT": (
        _mut(lambda c: _replace_action_pipeline(c, kernel=StudentTKernel())),
        "Heavy-tailed likelihood-motivated weighting",
    ),
    "kernel=GemanMcClure": (
        _mut(lambda c: _replace_action_pipeline(c, kernel=GemanMcClureKernel())),
        "Bounded-influence redescending kernel",
    ),
    "kernel=AndrewsSine": (
        _mut(lambda c: _replace_action_pipeline(c, kernel=AndrewsSineKernel())),
        "Sinusoidal redescending kernel",
    ),
}

# ── Residual computer (action space) ──────────────────────────────────────────-
GROUPS["residual_computer"] = {
    "residual=Euclidean (baseline)": (
        _mut(
            lambda c: _replace_action_pipeline(
                c, residual_computer=EuclideanResidualComputer()
            )
        ),
        "L2 distance to consensus action",
    ),
    "residual=Manhattan": (
        _mut(
            lambda c: _replace_action_pipeline(
                c, residual_computer=ManhattanResidualComputer()
            )
        ),
        "L1 distance, less sensitive to large single-axis errors",
    ),
    "residual=Lp": (
        _mut(
            lambda c: _replace_action_pipeline(
                c, residual_computer=LpResidualComputer()
            )
        ),
        "Generalised Lp distance",
    ),
    "residual=Angular": (
        _mut(
            lambda c: _replace_action_pipeline(
                c, residual_computer=AngularResidualComputer()
            )
        ),
        "Direction-only residual, ignores magnitude errors",
    ),
    "residual=Cosine": (
        _mut(
            lambda c: _replace_action_pipeline(
                c, residual_computer=CosineResidualComputer()
            )
        ),
        "Cosine-distance residual",
    ),
    "residual=EuclideanAngular": (
        _mut(
            lambda c: _replace_action_pipeline(
                c, residual_computer=EuclideanAngularResidualComputer()
            )
        ),
        "Combined magnitude + direction residual",
    ),
    "residual=Projection": (
        _mut(
            lambda c: _replace_action_pipeline(
                c, residual_computer=ProjectionResidualComputer()
            )
        ),
        "Residual projected onto consensus tangent",
    ),
    "residual=RelativeMagnitude": (
        _mut(
            lambda c: _replace_action_pipeline(
                c, residual_computer=RelativeMagnitudeResidualComputer()
            )
        ),
        "Residual normalised by consensus action magnitude",
    ),
}

# ── Trust floor (w_min) sensitivity ────────────────────────────────────────────
GROUPS["trust_floor"] = {
    "floor=0.00": (
        _mut(
            lambda c: _replace_action_pipeline(
                c, transforms=[MinimumTrustFloor(minimum=0.0)]
            )
        ),
        "No floor: fully-corrupted points can get exactly zero weight",
    ),
    "floor=0.02 (baseline)": (
        _mut(
            lambda c: _replace_action_pipeline(
                c, transforms=[MinimumTrustFloor(minimum=0.02)]
            )
        ),
        "Small floor prevents total exclusion of any point",
    ),
    "floor=0.10": (
        _mut(
            lambda c: _replace_action_pipeline(
                c, transforms=[MinimumTrustFloor(minimum=0.10)]
            )
        ),
        "Larger floor: even distrusted points retain meaningful weight",
    ),
    "floor=0.25": (
        _mut(
            lambda c: _replace_action_pipeline(
                c, transforms=[MinimumTrustFloor(minimum=0.25)]
            )
        ),
        "Aggressive floor, approaches uniform weighting",
    ),
}

# ── Pseudo-label refinement steps ────────────────────────-
_ALL_STEPS: Callable[[], list] = lambda: [
    DebiasTowardsAnchorStep(debias_weight=0.5),
    SidewaysAttenuationStep(shrinkage=0.5),
    SpeedRegularisationStep(influence=0.5),
]


def _steps_without(name: str) -> list:
    return [s for s in _ALL_STEPS() if type(s).__name__ != name]


GROUPS["pseudolabel_steps"] = {
    "steps=all (baseline)": (
        _mut(lambda c: _replace_action_steps(c, steps=_ALL_STEPS())),
        "Debias + sideways attenuation + speed regularisation",
    ),
    "steps=none": (
        _mut(lambda c: _replace_action_steps(c, steps=[])),
        "Raw trust-weighted actions, no geometric refinement",
    ),
    "steps=no-debias": (
        _mut(
            lambda c: _replace_action_steps(
                c, steps=_steps_without("DebiasTowardsAnchorStep")
            )
        ),
        "Removes pull toward the robust consensus anchor",
    ),
    "steps=no-sideways-attenuation": (
        _mut(
            lambda c: _replace_action_steps(
                c, steps=_steps_without("SidewaysAttenuationStep")
            )
        ),
        "Removes shrinkage of off-tangent (perpendicular) error",
    ),
    "steps=no-speed-regularisation": (
        _mut(
            lambda c: _replace_action_steps(
                c, steps=_steps_without("SpeedRegularisationStep")
            )
        ),
        "Removes blending of action magnitude toward consensus speed",
    ),
}

# ── Temporal smoothing (kappa) sweep ──────────────────────────────────────────-
GROUPS["smoothing_kappa"] = {
    f"kappa={k}": (
        _mut(
            lambda c, k=k: setattr(
                c.action_pseudo_label_params,
                "smoother",
                TemporalSmoother(smoothing_weight=k),
            )
        ),
        f"EMA smoothing weight kappa={k} on refined action labels",
    )
    for k in (0.0, 0.3, 0.6, 0.9)
}

# ── Bin resolution sweep ────────────────────────────────────────────────────────-
GROUPS["n_bins"] = {
    f"n_bins={n}": (
        _mut(lambda c, n=n: setattr(c, "n_bins", n)),
        f"Phase discretised into {n} bins",
    )
    for n in (32, 64, 96, 192)
}

# ── State-label usage ──────────────────────────────────────────────────────────-
GROUPS["state_labels"] = {
    "use_state_labels=False (baseline)": (
        _mut(lambda c: setattr(c, "use_state_labels", False)),
        "Only actions are trust-weighted and refined",
    ),
    "use_state_labels=True": (
        _mut(lambda c: setattr(c, "use_state_labels", True)),
        "States are also trust-weighted/refined, feeding WeightedBCTrainer",
    ),
}

ALL_GROUP_NAMES = list(GROUPS.keys())

## ── Metric collection ─────────────────────────────────────────────────────────


def _to_float(x: Any) -> float:
    detach = getattr(x, "detach", None)
    if callable(detach):
        x = detach()
    try:
        return float(x)
    except Exception:
        return float(np.asarray(x).mean())


def collect_row(
    *,
    group: str,
    variant: str,
    description: str,
    demonstrations,
    bc_policy_loss: float,
    pacer_bc_result,
) -> dict:
    pacer_result = pacer_bc_result.pacer_result

    residuals = ResidualAnalyser(
        demonstrations, pacer_result=pacer_result
    ).compute_action_residuals()
    trust_stats = TrustValueAnalysis(pacer_result.action_trust_values).statistics()
    low_trust_frac = TrustValueAnalysis(
        pacer_result.action_trust_values
    ).low_trust_fraction(0.25)
    correction = CorrectionMagnitudeAnalyser(
        demonstrations, pacer_result=pacer_result
    ).analyse_actions()
    smoothness_raw = SmoothnessAnalyser(actions=demonstrations.actions).analyse()
    smoothness_pseudo = SmoothnessAnalyser(
        actions=pacer_result.pseudo_labels.actions
    ).analyse()

    return {
        "group": group,
        "variant": variant,
        "description": description,
        "bc_only_loss": bc_policy_loss,
        "pacer_bc_loss": _to_float(pacer_bc_result.bc_policy_loss),
        "loss_delta_vs_bc": _to_float(pacer_bc_result.bc_policy_loss) - bc_policy_loss,
        "mean_action_residual": _to_float(residuals.mean_residual),
        "median_action_residual": _to_float(residuals.median_residual),
        "max_action_residual": _to_float(residuals.max_residual),
        "mean_correction_magnitude": _to_float(correction.mean_magnitude),
        "max_correction_magnitude": _to_float(correction.max_magnitude),
        "trust_mean": _to_float(trust_stats.mean),
        "trust_median": _to_float(trust_stats.median),
        "trust_std": _to_float(trust_stats.std),
        "trust_min": _to_float(trust_stats.minimum),
        "low_trust_fraction_lt_0.25": _to_float(low_trust_frac),
        "smoothness_raw_demos": _to_float(smoothness_raw.mean_smoothness),
        "smoothness_pseudo_labels": _to_float(smoothness_pseudo.mean_smoothness),
        "smoothness_ratio_pseudo_over_raw": (
            _to_float(smoothness_pseudo.mean_smoothness)
            / _to_float(smoothness_raw.mean_smoothness)
            if _to_float(smoothness_raw.mean_smoothness) != 0
            else float("nan")
        ),
    }


## ── Runner ────────────────────────────────────────────────────────────────────


def run_ablation(
    group_names: list[str],
    *,
    lasa_pattern: str = "GShape",
    out_dir: Path = Path("ablation_out"),
) -> pd.DataFrame:
    demonstrations = DemonstrationLoader(
        config=DemonstrationLoaderConfig(
            choice="FROM_LASA",
            LASA_pattern=lasa_pattern,
            filepath=None,
            corruptions_choice=None,
        )
    ).load()

    bc_result = BCExperiment(
        demonstrations, bc_train_config=BASELINE_BC_TRAIN_CONFIG
    ).run()
    bc_policy_loss = _to_float(bc_result.bc_policy_loss)

    rows: list[dict] = []
    for group in group_names:
        if group not in GROUPS:
            raise KeyError(f"Unknown group {group!r}. Available: {ALL_GROUP_NAMES}")
        for variant, (mutate, description) in GROUPS[group].items():
            cfg = mutate(make_baseline_pacer_config())
            pacer_bc_result = PACERBCExperiment(
                demonstrations,
                pacer_config=cfg,
                bc_train_config=BASELINE_BC_TRAIN_CONFIG,
            ).run()
            rows.append(
                collect_row(
                    group=group,
                    variant=variant,
                    description=description,
                    demonstrations=demonstrations,
                    bc_policy_loss=bc_policy_loss,
                    pacer_bc_result=pacer_bc_result,
                )
            )

    df = pd.DataFrame(rows)

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "ablation_results.csv"
    md_path = out_dir / "ablation_results.md"
    df.to_csv(csv_path, index=False)
    with open(md_path, "w") as f:
        f.write(f"# PACER ablation study ({lasa_pattern})\n\n")
        f.write(f"BC-only policy loss (reference): **{bc_policy_loss:.6f}**\n\n")
        for group in group_names:
            gdf = df[df["group"] == group].drop(columns=["group"])
            f.write(f"## {group}\n\n")
            f.write(gdf.to_markdown(index=False, floatfmt=".4f"))
            f.write("\n\n")

    print(f"Wrote {csv_path} and {md_path}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--groups", nargs="*", default=ALL_GROUP_NAMES, help="Ablation groups to run"
    )
    parser.add_argument("--lasa-pattern", default="GShape")
    parser.add_argument("--out-dir", default="ablation_out")
    parser.add_argument("--list", action="store_true", help="List groups and exit")
    args = parser.parse_args()

    if args.list:
        for g, variants in GROUPS.items():
            print(f"{g}: {list(variants.keys())}")
        raise SystemExit(0)

    run_ablation(
        args.groups, lasa_pattern=args.lasa_pattern, out_dir=Path(args.out_dir)
    )

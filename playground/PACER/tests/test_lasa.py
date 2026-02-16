# tests/test_lasa.py

import pyLasaDataset as lasa  # type: ignore[import-untyped]  # ty: ignore[unused-ignore-comment]
from pacer import console
from pacer.base import PACER, DemonstrationCorrupter
from pacer.lasa import LASADemonstrations
from pacer.plotting import full_diagnostic


def test_lasa(use_corruptions: bool = False) -> None:
    demonstrations = LASADemonstrations(lasa.DataSet.GShape).to_demonstrations()
    if use_corruptions:
        corrupter = DemonstrationCorrupter(
            demonstrations=demonstrations,
            noise_std=0.2,
            outlier_fraction=0.2,
            outlier_scale=5.0,
            bias_strength=0.2,
        )
        demonstrations = corrupter.inject_corruptions()
    pacer = PACER(demonstrations)

    phase_loss = pacer.prepare(
        phase_hidden_dim=128,
        phase_margin=1.0,  # m
        phase_lr=1e-3,
        phase_epochs=240,
        n_bins=96,  # B
        tukey_cutoff=4.685,  # c
        min_trust=0.02,  # w_min
        debias_weight=0.5,  # lambda_{debias}
        sideways_attenuation_shrinkage=0.5,  # rho_0
        speed_regularisation_influence=0.5,  # eta_0
        temporal_smoothing_weight=0.0,  # kappa
    )
    console.print(f"Phase scorer loss: {phase_loss}")

    policy_loss = pacer.train(
        policy_hidden_dim=128,
        policy_lr=1e-3,
        policy_epochs=240,
    )
    console.print(f"Policy loss: {policy_loss}")

    full_diagnostic(pacer)


if __name__ == "__main__":
    test_lasa()

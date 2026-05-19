"""
Test DemInf
"""
# tests/test_deminf.py

# pyright: reportPrivateImportUsage = false

import os
from typing import Any

import torch
from typingkit.core import RuntimeOptions, set_global_default_runtime_options

from deminf.deminf import (
    BatchScorer,
    BetaVAE,
    DemInfEstimator,
    KSGEstimator,
    MLPRepresentationEncoder,
    Scorer,
)
from pacer import console
from pacer.base import Demonstrations
from pacer.datasets import DemonstrationLoader, DemonstrationLoaderConfig

set_global_default_runtime_options(RuntimeOptions(validate=True))


def test_ksg(demonstrations: Demonstrations[Any, Any, Any, Any]) -> None:
    estimator = KSGEstimator(k=3)
    scorer = Scorer(estimator)

    scores = scorer.score_demonstrations(demonstrations)
    rankings = scorer.rank_scores(scores)
    console.print("rankings=", rankings)


def test_deminf(demonstrations: Demonstrations[Any, Any, Any, Any]) -> None:
    device = torch.device("cpu")

    # Load trained VAEs
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ckpt_dir = os.path.join(base_dir, "checkpoints")

    state_vae = BetaVAE(input_dim=demonstrations.state_dim, latent_dim=8, beta=0.005)
    state_path = os.path.join(ckpt_dir, "state_vae.pth")
    state_vae.load_state_dict(
        torch.load(state_path, map_location=device, weights_only=True)
    )
    state_vae.to(device)

    action_vae = BetaVAE(input_dim=demonstrations.action_dim, latent_dim=6, beta=0.005)
    action_path = os.path.join(ckpt_dir, "action_vae.pth")
    action_vae.load_state_dict(
        torch.load(action_path, map_location=device, weights_only=True)
    )
    action_vae.to(device)

    state_enc = MLPRepresentationEncoder(vae=state_vae)
    action_enc = MLPRepresentationEncoder(vae=action_vae)
    estimator = DemInfEstimator(state_encoder=state_enc, action_encoder=action_enc)

    scorer = BatchScorer(estimator=estimator.core)
    scores = scorer.score_demonstrations(demonstrations)
    rankings = scorer.rank_scores(scores)
    console.print("rankings=", rankings)


if __name__ == "__main__":
    demonstrations = DemonstrationLoader(
        config=DemonstrationLoaderConfig(choice="FROM_LASA", LASA_pattern="GShape")
    ).load()
    test_ksg(demonstrations)
    console.rule()
    test_deminf(demonstrations)
